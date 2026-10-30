"""
Feature extraction pipeline for MusicWhy — Phase 2A.

Extracts a comprehensive set of audio features from each track and stores them
in HDF5 format for fast, compact, random-access retrieval.

GPU: Not needed (CPU-bound signal processing).
Parallelizable: Yes, embarrassingly parallel across files via joblib.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import h5py
import librosa
import numpy as np
from joblib import Parallel, delayed

from musicwhy.features.feature_to_language import dynamic_range_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SR: int = 22050
MEL_BANDS: int = 128
HOP_LENGTHS: tuple[int, ...] = (512, 1024, 2048)
N_MFCC: int = 20
KEY_NAMES: list[str] = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float:
    """Convert a value to a plain Python float, handling numpy scalars."""
    return float(np.asarray(value).item())


def _safe_array(value: Any) -> np.ndarray:
    """Ensure *value* is a finite numpy array; replace NaN/Inf with 0."""
    arr = np.asarray(value, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Extract audio features and persist them in HDF5.

    Parameters
    ----------
    sr : int
        Target sample rate for loading audio (default 22050).
    """

    def __init__(self, sr: int = DEFAULT_SR) -> None:
        self.sr = sr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_single(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract all features from a single audio file.

        Parameters
        ----------
        audio_path : str | Path
            Path to an audio file readable by *librosa*.

        Returns
        -------
        dict
            Mapping of feature names to numpy arrays or scalar values.

        Raises
        ------
        FileNotFoundError
            If *audio_path* does not exist.
        RuntimeError
            If the file cannot be decoded or is zero-length.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Extracting features from %s", audio_path)

        try:
            y, sr = librosa.load(str(audio_path), sr=self.sr)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load audio from {audio_path}: {exc}"
            ) from exc

        if y is None or len(y) == 0:
            raise RuntimeError(f"Zero-length audio in {audio_path}")

        features: Dict[str, Any] = {}

        # --- 1. Mel spectrograms at multiple resolutions -----------------
        for hop in HOP_LENGTHS:
            S = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=MEL_BANDS, hop_length=hop,
            )
            features[f"mel_spec_hop{hop}"] = _safe_array(
                librosa.power_to_db(S, ref=np.max)
            )

        # --- 2. Chromagram (CQT-based) ----------------------------------
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        features["chroma"] = _safe_array(chroma)

        # --- 3. MFCCs (full time series + summary stats) -----------------
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        features["mfcc"] = _safe_array(mfcc)
        features["mfcc_mean"] = _safe_array(np.mean(mfcc, axis=1))
        features["mfcc_std"] = _safe_array(np.std(mfcc, axis=1))

        # --- 4. Onset strength envelope ----------------------------------
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        features["onset_strength"] = _safe_array(onset_env)

        # --- 5. Spectral features ----------------------------------------
        features["spectral_centroid"] = _safe_array(
            librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        )
        features["spectral_bandwidth"] = _safe_array(
            librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        )
        features["spectral_rolloff"] = _safe_array(
            librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        )
        features["spectral_contrast"] = _safe_array(
            librosa.feature.spectral_contrast(y=y, sr=sr)
        )

        # Spectral flux: mean frame-to-frame change across mel bands.
        mel_default = librosa.feature.melspectrogram(y=y, sr=sr)
        if mel_default.shape[1] > 1:
            flux = np.mean(np.abs(np.diff(mel_default, axis=1)), axis=0)
        else:
            flux = np.zeros(1, dtype=np.float32)
        features["spectral_flux"] = _safe_array(flux)

        # --- 6. Harmonic-percussive separation ratio ---------------------
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_energy = float(np.sum(y_harmonic ** 2))
        percussive_energy = float(np.sum(y_percussive ** 2))
        hp_ratio = harmonic_energy / (percussive_energy + 1e-10)
        features["harmonic_percussive_ratio"] = _safe_float(hp_ratio)

        # --- 7. Tempo and beat positions ---------------------------------
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        # librosa >= 0.10 returns tempo as an ndarray; handle both.
        features["tempo"] = _safe_float(tempo)
        features["beat_positions"] = _safe_array(beats)

        # --- 8. Key estimation from chroma mean --------------------------
        chroma_mean = np.mean(chroma, axis=1)
        features["key_profile"] = _safe_array(chroma_mean)
        features["estimated_key"] = int(np.argmax(chroma_mean))
        features["estimated_key_name"] = KEY_NAMES[int(np.argmax(chroma_mean))]

        # --- 9. Loudness contour (RMS) + dynamic range -------------------
        rms = librosa.feature.rms(y=y)[0]
        features["loudness_contour"] = _safe_array(rms)
        # Legacy, kept for old outputs: it tracks loudness, not dynamics. Use dynamic_range_db.
        features["dynamic_range"] = _safe_float(np.max(rms) - np.min(rms))
        features["dynamic_range_db"] = dynamic_range_db(rms)

        # --- 10. Duration and sample rate --------------------------------
        features["duration"] = _safe_float(len(y) / sr)
        features["sample_rate"] = int(sr)

        logger.info(
            "Extracted %d feature groups from %s (%.1f s)",
            len(features), audio_path, features["duration"],
        )
        return features

    # ------------------------------------------------------------------

    def extract_batch(
        self,
        paths: Sequence[Union[str, Path]],
        n_jobs: int = 4,
    ) -> Dict[str, Dict[str, Any]]:
        """Extract features from multiple audio files in parallel.

        Parameters
        ----------
        paths : sequence of str | Path
            Audio file paths.
        n_jobs : int
            Number of parallel workers (passed to *joblib*).

        Returns
        -------
        dict
            ``{audio_path_stem: features_dict, ...}``.
            Files that fail extraction are logged and skipped.
        """

        def _safe_extract(path: Union[str, Path]) -> tuple[str, Optional[Dict[str, Any]]]:
            try:
                feats = self.extract_single(path)
                return (Path(path).stem, feats)
            except Exception:
                logger.exception("Failed to extract features from %s", path)
                return (Path(path).stem, None)

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_safe_extract)(p) for p in paths
        )

        return {name: feats for name, feats in results if feats is not None}

    # ------------------------------------------------------------------
    # Persistence (HDF5)
    # ------------------------------------------------------------------

    @staticmethod
    def save_features(
        track_id: str,
        features: Dict[str, Any],
        output_dir: Union[str, Path],
    ) -> Path:
        """Save extracted features to an HDF5 file.

        Each feature becomes a dataset (arrays) or an attribute (scalars /
        strings) inside one HDF5 file per track.

        Parameters
        ----------
        track_id : str
            Unique track identifier (used as filename stem).
        features : dict
            Output of :meth:`extract_single`.
        output_dir : str | Path
            Directory where the HDF5 file will be written.

        Returns
        -------
        Path
            Full path to the written HDF5 file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        h5_path = output_dir / f"{track_id}.h5"

        with h5py.File(str(h5_path), "w") as f:
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(
                        key, data=value, compression="gzip", compression_opts=4,
                    )
                elif isinstance(value, (int, float)):
                    f.attrs[key] = value
                elif isinstance(value, str):
                    f.attrs[key] = value
                else:
                    # Fall back: try to store as numpy array.
                    try:
                        arr = np.asarray(value)
                        f.create_dataset(
                            key, data=arr, compression="gzip", compression_opts=4,
                        )
                    except Exception:
                        logger.warning(
                            "Skipping non-serializable feature %r for track %s",
                            key, track_id,
                        )

        logger.info("Saved features for %s -> %s", track_id, h5_path)
        return h5_path

    @staticmethod
    def load_features(
        track_id: str,
        output_dir: Union[str, Path],
    ) -> Dict[str, Any]:
        """Load features from an HDF5 file.

        Parameters
        ----------
        track_id : str
            Track identifier (filename stem).
        output_dir : str | Path
            Directory containing the HDF5 file.

        Returns
        -------
        dict
            Mapping of feature names to numpy arrays or scalars.

        Raises
        ------
        FileNotFoundError
            If the HDF5 file does not exist.
        """
        h5_path = Path(output_dir) / f"{track_id}.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"Feature file not found: {h5_path}")

        features: Dict[str, Any] = {}
        with h5py.File(str(h5_path), "r") as f:
            # Datasets (arrays)
            for key in f.keys():
                features[key] = f[key][()]
            # Attributes (scalars / strings)
            for key, value in f.attrs.items():
                features[key] = value
        if "dynamic_range_db" not in features and "loudness_contour" in features:
            features["dynamic_range_db"] = dynamic_range_db(features["loudness_contour"])

        logger.info("Loaded features for %s from %s", track_id, h5_path)
        return features
