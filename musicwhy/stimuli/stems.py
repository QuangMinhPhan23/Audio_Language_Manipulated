"""Phase 0 v2: real stem clips from MoisesDB and MUSDB18 (PHASE0_v2_real_stems.md).

Metadata functions (stem counting, instrument mapping) are stdlib-only for fast import on cold
GPFS nodes. Audio functions (window cutting, rendering, LUFS normalisation) import numpy/soundfile
lazily inside each function.
"""
from __future__ import annotations

import io
import json
import posixpath
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SR = 44100
WINDOW_S = 20.0
ACTIVITY_THRESHOLD_DB = -40.0
ACTIVITY_MIN_LEAD = 0.80
ACTIVITY_MIN_VOCAL = 0.50
TARGET_LUFS = -16.0
RECON_TOL_DB = -60.0

# ---- MoisesDB stemName → our instrument categories ----
# MoisesDB data.json has stems[].stemName at the group level and stems[].tracks[] for sub-tracks.
# The stemName is the coarse group; we map it to our categories.  For Guitar we need the track
# name to distinguish acoustic from electric, so we also check tracks[].trackType when present.

STEM_TO_INSTRUMENT: Dict[str, str] = {
    "piano": "piano",
    "guitar": "electric guitar",        # default; overridden to acoustic guitar per-track
    "other_keys": "synthesizer",        # default; overridden to organ per-track
    "bowed_strings": "strings",
    "wind": "wind",
}

# Track-level overrides: (stemName, trackType substring) → instrument
TRACK_OVERRIDES: List[Tuple[str, str, str]] = [
    ("guitar", "acoustic", "acoustic guitar"),
    ("other_keys", "organ", "organ"),
    ("other_keys", "electric organ", "organ"),
]

NON_MELODIC = {"vocals", "bass", "drums"}

MIN_SONGS_PER_INSTRUMENT = 25


# ---------------------------------------------------------------------------
# Step 1: count stems (stdlib only — reads JSON, no audio)
# ---------------------------------------------------------------------------

def classify_stem(stem: dict) -> Optional[str]:
    """Map one MoisesDB stem entry to our instrument category, or None if not melodic."""
    name = stem.get("stemName", "")
    if name in NON_MELODIC:
        return None
    base = STEM_TO_INSTRUMENT.get(name)
    if base is None:
        return None
    tracks = stem.get("tracks", [])
    if not tracks:
        return base
    for track in tracks:
        tt = track.get("trackType", "") or track.get("name", "")
        for stem_name, pattern, instrument in TRACK_OVERRIDES:
            if name == stem_name and pattern.lower() in tt.lower():
                return instrument
    return base


def classify_song_stems(data_json: dict) -> Dict[str, List[str]]:
    """For one MoisesDB track, return {our_instrument: [stemName, ...]} for each melodic stem group."""
    out: Dict[str, List[str]] = {}
    for stem in data_json.get("stems", []):
        inst = classify_stem(stem)
        if inst is not None:
            out.setdefault(inst, []).append(stem.get("stemName", ""))
    return out


def read_data_jsons_from_zip(zf: zipfile.ZipFile) -> List[Tuple[str, dict]]:
    """Yield (uuid, parsed data.json) for every track in a MoisesDB zip."""
    results = []
    for name in sorted(zf.namelist()):
        if posixpath.basename(name) == "data.json" and name.count("/") >= 1:
            uuid = posixpath.basename(posixpath.dirname(name))
            results.append((uuid, json.loads(zf.read(name))))
    return results


def read_data_jsons_from_dir(moisesdb_root: Path) -> List[Tuple[str, dict]]:
    """Read data.json from extracted MoisesDB directories."""
    results = []
    root = Path(moisesdb_root)
    for dj in sorted(root.rglob("data.json")):
        uuid = dj.parent.name
        with open(dj, encoding="utf-8") as f:
            results.append((uuid, json.load(f)))
    return results


def count_stems(tracks: Sequence[Tuple[str, dict]]) -> Dict[str, int]:
    """Count how many songs have at least one stem for each of our instrument categories."""
    counts: Counter = Counter()
    for _uuid, info in tracks:
        seen = set()
        for stem in info.get("stems", []):
            inst = classify_stem(stem)
            if inst is not None:
                seen.add(inst)
        counts.update(seen)
    return dict(counts)


def pick_instruments(counts: Dict[str, int], min_songs: int = MIN_SONGS_PER_INSTRUMENT) -> List[str]:
    """Return the instrument categories that have enough songs, sorted alphabetically."""
    return sorted(k for k, v in counts.items() if v >= min_songs)


def stem_inventory(tracks: Sequence[Tuple[str, dict]]) -> List[Dict[str, Any]]:
    """Per-track inventory: uuid, song title/artist, and which of our instruments it has."""
    rows = []
    for uuid, info in tracks:
        instruments = classify_song_stems(info)
        rows.append({
            "uuid": uuid,
            "title": info.get("song", "") or uuid,
            "artist": info.get("artist", "") or "unknown",
            "instruments": instruments,
            "stem_names": [s.get("stemName", "") for s in info.get("stems", [])],
        })
    return rows


# ---------------------------------------------------------------------------
# Step 2: cut windows (needs numpy + soundfile)
# ---------------------------------------------------------------------------

def _rms_frames(audio, sr: int, hop_s: float = 0.05):
    """Per-frame RMS of a mono signal; returns (rms_array, hop_samples)."""
    import numpy as np
    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    hop = max(int(hop_s * sr), 1)
    n = len(mono) // hop
    if n == 0:
        return np.zeros(0), hop
    frames = mono[:n * hop].reshape(n, hop)
    return np.sqrt(np.mean(frames ** 2, axis=1)), hop


def _activity_ratio(rms, threshold_db: float = ACTIVITY_THRESHOLD_DB) -> float:
    """Fraction of frames above the dBFS threshold."""
    import numpy as np
    if len(rms) == 0:
        return 0.0
    threshold_linear = 10 ** (threshold_db / 20.0)
    return float(np.mean(rms >= threshold_linear))


def find_windows(lead_audio, sr: int, duration_s: float = WINDOW_S,
                 vocal_audio=None,
                 min_lead_activity: float = ACTIVITY_MIN_LEAD,
                 min_vocal_activity: float = ACTIVITY_MIN_VOCAL,
                 max_windows: int = 2) -> List[Tuple[int, int]]:
    """Find up to max_windows non-overlapping windows where the lead stem is active enough.

    Returns list of (start_sample, end_sample) tuples.
    """
    import numpy as np
    lead_rms, hop = _rms_frames(lead_audio, sr)
    if vocal_audio is not None:
        vocal_rms, _ = _rms_frames(vocal_audio, sr)
        vocal_rms = vocal_rms[:len(lead_rms)]
        if len(vocal_rms) < len(lead_rms):
            vocal_rms = np.pad(vocal_rms, (0, len(lead_rms) - len(vocal_rms)))
    else:
        vocal_rms = None

    win_frames = int(duration_s / (hop / sr))
    if win_frames > len(lead_rms):
        return []

    threshold_linear = 10 ** (ACTIVITY_THRESHOLD_DB / 20.0)
    lead_above = lead_rms >= threshold_linear
    if vocal_rms is not None:
        vocal_above = vocal_rms >= threshold_linear

    candidates = []
    for start in range(0, len(lead_rms) - win_frames + 1, win_frames // 4):
        la = float(np.mean(lead_above[start:start + win_frames]))
        if la < min_lead_activity:
            continue
        if vocal_rms is not None:
            va = float(np.mean(vocal_above[start:start + win_frames]))
            if va < min_vocal_activity:
                continue
        else:
            va = 0.0
        candidates.append((start, la, va))

    candidates.sort(key=lambda x: x[1], reverse=True)

    selected: List[Tuple[int, int]] = []
    for start_frame, _la, _va in candidates:
        start_samp = start_frame * hop
        end_samp = start_samp + int(duration_s * sr)
        overlaps = any(not (end_samp <= s or start_samp >= e) for s, e in selected)
        if not overlaps:
            selected.append((start_samp, end_samp))
            if len(selected) >= max_windows:
                break
    return selected


# ---------------------------------------------------------------------------
# Step 3: render sparse mixes
# ---------------------------------------------------------------------------

def _stereo(audio):
    import numpy as np
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[1] == 1:
        a = np.repeat(a, 2, axis=1)
    return a[:, :2]


def _pad_to(audio, length: int):
    """Zero-pad or truncate audio (samples, channels) to exactly length samples."""
    import numpy as np
    a = np.asarray(audio, dtype=np.float32)
    if len(a) >= length:
        return a[:length]
    out = np.zeros((length, a.shape[1] if a.ndim == 2 else 1), dtype=np.float32)
    out[:len(a)] = a
    return out


def load_stem_audio(source, stem_name: str, data_json: dict, sr: int = SR):
    """Load a specific stem's audio from a zip or directory.

    source: zipfile.ZipFile or Path to the track directory (containing stemName/ subdirs).
    Returns stereo numpy array (samples, 2) at the source sample rate.
    """
    import numpy as np
    import soundfile as sf

    stems = [s for s in data_json.get("stems", []) if s.get("stemName") == stem_name]
    if not stems:
        return None
    combined = None
    for stem in stems:
        for track in stem.get("tracks", []):
            ext = track.get("extension", "wav")
            if isinstance(source, zipfile.ZipFile):
                track_dir = None
                for name in source.namelist():
                    if posixpath.basename(name) == "data.json":
                        candidate = posixpath.dirname(name)
                        if candidate.endswith(data_json.get("_uuid", "")):
                            track_dir = candidate
                            break
                if track_dir is None:
                    for name in source.namelist():
                        if name.endswith(f"{track['id']}.{ext}"):
                            track_dir = posixpath.dirname(posixpath.dirname(name))
                            break
                if track_dir is None:
                    continue
                member = posixpath.join(track_dir, stem_name, f"{track['id']}.{ext}")
                try:
                    audio, file_sr = sf.read(io.BytesIO(source.read(member)),
                                             dtype="float32", always_2d=True)
                except (KeyError, RuntimeError):
                    continue
            else:
                path = Path(source) / stem_name / f"{track['id']}.{ext}"
                if not path.exists():
                    continue
                audio, file_sr = sf.read(str(path), dtype="float32", always_2d=True)

            if file_sr != sr:
                import librosa
                audio = librosa.resample(audio.T, orig_sr=file_sr, target_sr=sr,
                                         res_type="soxr_hq").T

            audio = _stereo(audio)
            if combined is None:
                combined = audio.copy()
            else:
                n = max(len(combined), len(audio))
                out = np.zeros((n, 2), dtype=np.float32)
                out[:len(combined)] += combined
                out[:len(audio)] += audio
                combined = out
    return combined


def render_sparse_mix(stems: Dict[str, Any], window: Tuple[int, int],
                      include_vocals: bool) -> Any:
    """Render a sparse mix from pre-loaded stems within a window.

    stems: {"lead": array, "drums": array, "bass": array, "vocals": array}
    window: (start_sample, end_sample)
    """
    import numpy as np
    start, end = window
    length = end - start
    parts = []
    for key in ("lead", "drums", "bass"):
        a = stems.get(key)
        if a is not None:
            a = _stereo(a)
            chunk = _pad_to(a[start:end] if start < len(a) else np.zeros((0, 2)), length)
            parts.append(chunk)
    if include_vocals:
        v = stems.get("vocals")
        if v is not None:
            v = _stereo(v)
            parts.append(_pad_to(v[start:end] if start < len(v) else np.zeros((0, 2)), length))
    if not parts:
        return np.zeros((length, 2), dtype=np.float32)
    return sum(parts)


def lufs_normalise(audio, sr: int, target: float = TARGET_LUFS):
    """Loudness-normalise stereo audio to target LUFS. Returns the normalised audio and the gain applied."""
    import numpy as np
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    rms = np.sqrt(np.mean(a ** 2))
    if rms < 1e-9:
        return a, 0.0
    current_lufs = 20.0 * np.log10(rms) - 0.691
    gain_db = target - current_lufs
    gain = 10 ** (gain_db / 20.0)
    normalised = a * gain
    peak = float(np.abs(normalised).max())
    if peak > 0.999:
        normalised *= 0.999 / peak
        gain *= 0.999 / peak
    return normalised, float(gain)


def reconstruction_check(sparse_mix, removed_stems, full_mix,
                         window: Tuple[int, int], tol_db: float = RECON_TOL_DB) -> Dict[str, Any]:
    """Check that sparse_mix + removed_stems ≈ full_mix within tol_db for the windowed region."""
    import numpy as np
    start, end = window
    length = end - start

    sm = np.asarray(sparse_mix, dtype=np.float32)[:length]
    rm = np.asarray(removed_stems, dtype=np.float32)[:length]
    fm_chunk = np.asarray(full_mix, dtype=np.float32)
    if start < len(fm_chunk):
        fm_chunk = fm_chunk[start:end]
    fm_chunk = _pad_to(fm_chunk, length) if len(fm_chunk) < length else fm_chunk[:length]

    reconstructed = np.zeros((length, max(sm.shape[1] if sm.ndim == 2 else 1,
                                          rm.shape[1] if rm.ndim == 2 else 1)), dtype=np.float32)
    if sm.ndim == 1:
        sm = sm[:, None]
    if rm.ndim == 1:
        rm = rm[:, None]
    reconstructed[:len(sm), :sm.shape[1]] += sm
    reconstructed[:len(rm), :rm.shape[1]] += rm

    if fm_chunk.ndim == 1:
        fm_chunk = fm_chunk[:, None]

    diff = reconstructed[:, :fm_chunk.shape[1]] - fm_chunk
    diff_rms = float(np.sqrt(np.mean(diff ** 2)))
    fm_rms = float(np.sqrt(np.mean(fm_chunk ** 2)))
    if fm_rms < 1e-9:
        ratio_db = 0.0 if diff_rms < 1e-9 else -100.0
    else:
        ratio_db = 20.0 * np.log10(diff_rms / fm_rms) if diff_rms > 0 else -200.0
    return {
        "pass": ratio_db <= tol_db,
        "ratio_db": ratio_db,
        "diff_rms": diff_rms,
        "mix_rms": fm_rms,
    }


def window_activity(audio, sr: int, window: Tuple[int, int]) -> float:
    """Activity ratio of audio within a window."""
    import numpy as np
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 2:
        a = a.mean(axis=1)
    start, end = window
    chunk = a[start:end] if start < len(a) else np.zeros(0)
    if len(chunk) == 0:
        return 0.0
    rms, _ = _rms_frames(chunk, sr)
    return _activity_ratio(rms)


def save_clip(audio, sr: int, path, fmt: str = "WAV") -> None:
    """Write audio to disk. Uses WAV by default for Phase 0 clips."""
    import soundfile as sf
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, format=fmt)
