"""
Source separation module for MusicWhy — Phase 2B.

Wraps Meta's Demucs (htdemucs / htdemucs_6s) to split tracks into stems
(vocals, drums, bass, other, and optionally guitar + piano), then creates
ablated "song-minus-one" mixes for emotion attribution experiments.

GPU: YES — significantly faster on GPU.  Falls back to CPU on OOM.
VRAM: ~4-6 GB per song.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STEM_NAMES_4 = ("vocals", "drums", "bass", "other")
STEM_NAMES_6 = ("vocals", "drums", "bass", "other", "guitar", "piano")

ModelName = Literal["htdemucs", "htdemucs_6s"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device(preferred: Optional[str] = None) -> torch.device:
    """Return a torch device, defaulting to CUDA when available."""
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_audio_tensor(
    audio_path: Union[str, Path],
    target_sr: int = 44100,
) -> tuple[torch.Tensor, int]:
    """Load an audio file into a float32 tensor at *target_sr*.

    Returns
    -------
    (waveform, sample_rate)
        waveform shape: ``(channels, samples)``
    """
    audio_path = str(audio_path)
    # Use soundfile directly — torchaudio.load is broken (routes to missing torchcodec).
    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    # soundfile returns (samples, channels) — convert to (channels, samples) tensor
    waveform = torch.from_numpy(data.T)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    # Ensure stereo (Demucs expects 2-channel input).
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    return waveform, sr


# ---------------------------------------------------------------------------
# SourceSeparator
# ---------------------------------------------------------------------------

class SourceSeparator:
    """Wraps Demucs for stem separation and ablation mix generation.

    Parameters
    ----------
    model_name : str
        ``"htdemucs"`` (4-stem) or ``"htdemucs_6s"`` (6-stem).
    device : str | None
        Torch device string (e.g. ``"cuda"``, ``"cpu"``).  If *None*, CUDA
        is used when available.
    segment : int | None
        Demucs segment length in seconds.  Shorter values use less VRAM but
        may reduce quality.  *None* keeps the model default.
    """

    def __init__(
        self,
        model_name: ModelName = "htdemucs",
        device: Optional[str] = None,
        segment: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.device = _get_device(device)
        self._segment = segment
        self._model: Optional[Any] = None
        self._stem_names: tuple[str, ...] = (
            STEM_NAMES_6 if model_name == "htdemucs_6s" else STEM_NAMES_4
        )

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> Any:
        """Load the Demucs model on first use."""
        if self._model is not None:
            return self._model

        try:
            from demucs.pretrained import get_model
            from demucs.apply import BagOfModels
        except ImportError as exc:
            raise ImportError(
                "Demucs is required for source separation.  "
                "Install it with:  pip install demucs"
            ) from exc

        logger.info(
            "Loading Demucs model %r on %s ...", self.model_name, self.device,
        )
        model = get_model(self.model_name)

        # Unwrap BagOfModels to set segment on each sub-model.
        if self._segment is not None:
            if isinstance(model, BagOfModels):
                for m in model.models:
                    m.segment = self._segment
            else:
                model.segment = self._segment

        model.to(self.device)
        model.eval()
        self._model = model
        logger.info("Demucs model loaded successfully.")
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def stem_names(self) -> tuple[str, ...]:
        """Return the stem names produced by the loaded model."""
        return self._stem_names

    def separate_track(
        self,
        audio_path: Union[str, Path],
    ) -> Dict[str, np.ndarray]:
        """Separate an audio file into stems.

        Parameters
        ----------
        audio_path : str | Path
            Path to the audio file.

        Returns
        -------
        dict
            ``{stem_name: np.ndarray}`` where each array has shape
            ``(channels, samples)``.

        Raises
        ------
        FileNotFoundError
            If *audio_path* does not exist.
        RuntimeError
            If separation fails (e.g. corrupt file, OOM after fallback).
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = self._ensure_model()

        try:
            from demucs.apply import apply_model
        except ImportError as exc:
            raise ImportError("demucs is required.  pip install demucs") from exc

        waveform, sr = _load_audio_tensor(audio_path, target_sr=model.samplerate)

        # Add batch dimension: (1, channels, samples)
        ref = waveform.mean(0)
        waveform_input = (waveform - ref.mean()) / ref.std()
        waveform_input = waveform_input.unsqueeze(0).to(self.device)

        logger.info("Separating %s on %s ...", audio_path.name, self.device)

        try:
            with torch.no_grad():
                sources = apply_model(
                    model, waveform_input, device=self.device, progress=False,
                )
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "CUDA OOM during separation of %s — falling back to CPU.",
                audio_path.name,
            )
            torch.cuda.empty_cache()
            cpu_device = torch.device("cpu")
            model.to(cpu_device)
            waveform_input = waveform_input.to(cpu_device)
            with torch.no_grad():
                sources = apply_model(
                    model, waveform_input, device=cpu_device, progress=False,
                )
            # Move model back to original device for subsequent calls.
            model.to(self.device)

        # sources shape: (batch=1, n_sources, channels, samples)
        sources = sources[0]  # drop batch dim

        # Undo normalization
        sources = sources * ref.std() + ref.mean()

        stems: Dict[str, np.ndarray] = {}
        source_names = model.sources  # list of stem names from the model
        for idx, name in enumerate(source_names):
            stems[name] = sources[idx].cpu().numpy()

        logger.info(
            "Separation complete for %s: stems=%s",
            audio_path.name, list(stems.keys()),
        )
        return stems

    # ------------------------------------------------------------------

    @staticmethod
    def create_ablated_mixes(
        stems: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Create mixes with each stem removed in turn.

        Parameters
        ----------
        stems : dict
            ``{stem_name: np.ndarray}`` as returned by :meth:`separate_track`.

        Returns
        -------
        dict
            ``{"minus_{stem}": np.ndarray, ...}`` for each stem.
        """
        if not stems:
            raise ValueError("stems dict is empty — nothing to ablate.")

        ablated: Dict[str, np.ndarray] = {}
        all_names = list(stems.keys())

        for remove_stem in all_names:
            remaining = [arr for name, arr in stems.items() if name != remove_stem]
            if remaining:
                mix = np.sum(np.stack(remaining, axis=0), axis=0)
            else:
                # Only one stem — ablation yields silence.
                mix = np.zeros_like(stems[remove_stem])
            ablated[f"minus_{remove_stem}"] = mix

        return ablated

    # ------------------------------------------------------------------

    @staticmethod
    def save_stems(
        track_id: str,
        stems: Dict[str, np.ndarray],
        output_dir: Union[str, Path],
        sr: int = 44100,
    ) -> Dict[str, Path]:
        """Save separated stems as individual WAV files.

        Parameters
        ----------
        track_id : str
            Unique track identifier (used as a sub-directory name).
        stems : dict
            ``{stem_name: np.ndarray}`` — each array shape ``(channels, samples)``.
        output_dir : str | Path
            Root output directory.
        sr : int
            Sample rate for the WAV files (default 44100).

        Returns
        -------
        dict
            ``{stem_name: Path}`` mapping to the written files.
        """
        track_dir = Path(output_dir) / track_id
        track_dir.mkdir(parents=True, exist_ok=True)

        saved: Dict[str, Path] = {}
        for name, audio in stems.items():
            wav_path = track_dir / f"{name}.wav"
            # soundfile expects (samples, channels)
            if audio.ndim == 2:
                audio_t = audio.T
            else:
                audio_t = audio
            sf.write(str(wav_path), audio_t, sr, subtype="PCM_16")
            saved[name] = wav_path
            logger.debug("Saved stem %s -> %s", name, wav_path)

        logger.info(
            "Saved %d stems for track %s in %s", len(saved), track_id, track_dir,
        )
        return saved

    # ------------------------------------------------------------------

    def batch_separate(
        self,
        paths: Sequence[Union[str, Path]],
        output_dir: Union[str, Path],
        save_ablated: bool = True,
    ) -> Dict[str, Dict[str, Path]]:
        """Separate multiple tracks and save results.

        Parameters
        ----------
        paths : sequence of str | Path
            Audio file paths.
        output_dir : str | Path
            Root output directory for stems.
        save_ablated : bool
            If *True*, also save the ablated "minus-one" mixes alongside
            the raw stems.

        Returns
        -------
        dict
            ``{track_id: {stem_name: Path, ...}, ...}``
        """
        output_dir = Path(output_dir)
        all_saved: Dict[str, Dict[str, Path]] = {}

        for path in paths:
            path = Path(path)
            track_id = path.stem

            try:
                stems = self.separate_track(path)
            except Exception:
                logger.exception("Failed to separate %s — skipping.", path)
                continue

            # Determine sample rate from model
            model = self._ensure_model()
            sr = model.samplerate

            saved = self.save_stems(track_id, stems, output_dir, sr=sr)

            if save_ablated:
                try:
                    ablated = self.create_ablated_mixes(stems)
                    ablated_dir = output_dir / track_id / "ablated"
                    ablated_dir.mkdir(parents=True, exist_ok=True)
                    for abl_name, abl_audio in ablated.items():
                        abl_path = ablated_dir / f"{abl_name}.wav"
                        audio_t = abl_audio.T if abl_audio.ndim == 2 else abl_audio
                        sf.write(str(abl_path), audio_t, sr, subtype="PCM_16")
                        saved[abl_name] = abl_path
                except Exception:
                    logger.exception(
                        "Failed to create ablated mixes for %s.", track_id,
                    )

            all_saved[track_id] = saved

        logger.info(
            "Batch separation complete: %d / %d tracks succeeded.",
            len(all_saved), len(paths),
        )
        return all_saved
