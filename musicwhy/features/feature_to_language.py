"""
Feature-to-language translation for MusicWhy — Phase 2 / 3A bridge.

Converts raw numerical audio features into natural-language descriptions
that can be consumed by LLMs for reasoning annotation.  Also provides
comparative descriptions for stem analysis and ablation experiments.

GPU: Not needed (rule-based translation).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — thresholds from the experiment plan
# ---------------------------------------------------------------------------

KEY_NAMES: list[str] = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
]

# Tempo thresholds (BPM)
TEMPO_VERY_SLOW: float = 70.0
TEMPO_SLOW_MODERATE: float = 100.0
TEMPO_MODERATE: float = 130.0
TEMPO_FAST: float = 150.0

# Spectral centroid thresholds (Hz)
SC_DARK: float = 1500.0
SC_BALANCED: float = 3000.0

# Harmonic-percussive ratio thresholds
HP_VERY_HARMONIC: float = 5.0
HP_SLIGHTLY_HARMONIC: float = 1.0

# Dynamic range thresholds (dB between loud and quiet passages; tertiles of the 389 songs were 10.0 / 13.9)
DR_COMPRESSED: float = 10.0
DR_MODERATE: float = 14.0
DR_SILENCE_DB: float = 50.0  # frames this far below the track's peak RMS count as silence


def dynamic_range_db(rms: Any) -> float:
    """dB gap between loud (95th percentile) and quiet (10th) non-silent RMS frames.

    Replaces ``dynamic_range`` (RMS max minus min), which tracked loudness instead of dynamics:
    Spearman 0.79 with mean RMS and 0.04 with this measure over the 389 songs.
    """
    rms = np.asarray(rms, dtype=float).ravel()
    if rms.size == 0:
        return 0.0
    active = rms[rms > max(float(rms.max()) * 10 ** (-DR_SILENCE_DB / 20), 1e-5)]
    if active.size < 10:
        return 0.0
    quiet, loud = np.percentile(active, [10, 95])
    return float(20 * np.log10(loud / max(quiet, 1e-10)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_of(value: Any) -> float:
    """Return the mean of an array-like or the value itself if scalar."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return float(np.mean(arr))


def _energy_weighted(features: Dict[str, Any], key: str) -> float:
    """Per-frame feature averaged with RMS^2 weights (plain mean if no loudness contour)."""
    values = np.asarray(features.get(key, 0), dtype=float).ravel()
    rms = np.asarray(features.get("loudness_contour", []), dtype=float).ravel()
    n = min(len(values), len(rms))
    if values.size < 2 or n < 2:
        return _mean_of(values)
    weights = rms[:n] ** 2
    total = weights.sum()
    return float((values[:n] * weights).sum() / total) if total > 0 else _mean_of(values)


# Krumhansl-Kessler key profiles (C major / C minor), rotated to test all 24 keys
_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key_mode(key_profile: Any) -> Optional[tuple[int, str, float]]:
    """(tonic index, 'major'|'minor', correlation) from a 12-bin mean chroma, or None."""
    chroma = np.asarray(key_profile, dtype=float).ravel()
    if chroma.size != 12 or not np.isfinite(chroma).all() or chroma.std() == 0:
        return None
    best = None
    for mode, profile in (("major", _KK_MAJOR), ("minor", _KK_MINOR)):
        for tonic in range(12):
            r = float(np.corrcoef(chroma, np.roll(profile, tonic))[0, 1])
            if best is None or r > best[2]:
                best = (tonic, mode, r)
    return best


def _pct_change(full_val: float, compare_val: float) -> float:
    """Return the percentage change from *full_val* to *compare_val*."""
    if abs(full_val) < 1e-10:
        return 0.0
    return ((compare_val - full_val) / abs(full_val)) * 100.0


# ---------------------------------------------------------------------------
# Main translation functions
# ---------------------------------------------------------------------------

def features_to_description(features: Dict[str, Any]) -> str:
    """Convert numerical features to a readable text description.

    Uses the exact thresholds from the MusicWhy experiment plan:
    - Tempo: <70 very slow, <100 slow-moderate, <130 moderate, <150 fast,
      else very fast.
    - Spectral centroid: <1500 dark, <3000 balanced, else bright.
    - HP ratio: >5 very harmonic, >1 slightly harmonic, else percussive.
    - Dynamic range: <0.05 compressed, <0.15 moderate, else wide.

    Parameters
    ----------
    features : dict
        Output of ``FeatureExtractor.extract_single``.

    Returns
    -------
    str
        Multi-sentence natural-language description.
    """
    desc: list[str] = []

    # --- Tempo -----------------------------------------------------------
    tempo = float(features.get("tempo", 0))
    if tempo < TEMPO_VERY_SLOW:
        desc.append(
            f"Very slow tempo ({tempo:.0f} BPM, slower than resting heart rate)"
        )
    elif tempo < TEMPO_SLOW_MODERATE:
        desc.append(
            f"Slow-moderate tempo ({tempo:.0f} BPM, walking pace)"
        )
    elif tempo < TEMPO_MODERATE:
        desc.append(
            f"Moderate tempo ({tempo:.0f} BPM, typical pop/dance range)"
        )
    elif tempo < TEMPO_FAST:
        desc.append(
            f"Fast tempo ({tempo:.0f} BPM, energetic)"
        )
    else:
        desc.append(
            f"Very fast tempo ({tempo:.0f} BPM, intense/frantic)"
        )

    # --- Spectral centroid (brightness) ----------------------------------
    sc_mean = _mean_of(features.get("spectral_centroid", 0))
    if sc_mean < SC_DARK:
        desc.append(
            f"Dark/warm sound (avg spectral centroid {sc_mean:.0f} Hz)"
        )
    elif sc_mean < SC_BALANCED:
        desc.append(
            f"Balanced brightness (avg spectral centroid {sc_mean:.0f} Hz)"
        )
    else:
        desc.append(
            f"Bright/sharp sound (avg spectral centroid {sc_mean:.0f} Hz)"
        )

    # --- Harmonic vs. percussive -----------------------------------------
    hp = float(features.get("harmonic_percussive_ratio", 0))
    if hp > HP_VERY_HARMONIC:
        desc.append(
            f"Dominated by sustained/tonal sounds (very harmonic, HP ratio {hp:.2f})"
        )
    elif hp > HP_SLIGHTLY_HARMONIC:
        desc.append(
            f"Mix of tonal and rhythmic elements (slightly harmonic, HP ratio {hp:.2f})"
        )
    else:
        desc.append(
            f"Dominated by percussive/rhythmic elements (HP ratio {hp:.2f})"
        )

    # --- Dynamic range ---------------------------------------------------
    dr = features.get("dynamic_range_db")
    if dr is not None:
        dr = float(dr)
        if dr < DR_COMPRESSED:
            desc.append(
                f"Compressed dynamics ({dr:.1f} dB between loud and quiet passages, consistent loudness)"
            )
        elif dr < DR_MODERATE:
            desc.append(f"Moderate dynamic range ({dr:.1f} dB between loud and quiet passages)")
        else:
            desc.append(
                f"Wide dynamic range ({dr:.1f} dB between loud and quiet passages, expressive)"
            )

    # --- Key -------------------------------------------------------------
    key_mode = estimate_key_mode(features["key_profile"]) if "key_profile" in features else None
    key_idx = features.get("estimated_key", None)
    if key_mode is not None:
        tonic, mode, r = key_mode
        # A weak profile match means the tonality is ambiguous; say so rather than assert a mode
        certainty = "" if r >= 0.6 else ", low confidence"
        desc.append(f"Estimated key center: {KEY_NAMES[tonic]} {mode} (key-profile correlation {r:.2f}{certainty})")
    elif key_idx is not None:
        key_idx = int(key_idx)
        if 0 <= key_idx < len(KEY_NAMES):
            desc.append(f"Estimated key center: {KEY_NAMES[key_idx]}")

    # --- Spectral bandwidth (body / thinness) ----------------------------
    bw_mean = _mean_of(features.get("spectral_bandwidth", 0))
    if bw_mean > 0:
        if bw_mean < 1500:
            desc.append(
                f"Narrow spectral bandwidth ({bw_mean:.0f} Hz) — focused, thin sound"
            )
        elif bw_mean < 3000:
            desc.append(
                f"Moderate spectral bandwidth ({bw_mean:.0f} Hz) — balanced body"
            )
        else:
            desc.append(
                f"Wide spectral bandwidth ({bw_mean:.0f} Hz) — full, rich sound"
            )

    # --- Spectral rolloff (high-frequency energy) ------------------------
    rolloff_mean = _mean_of(features.get("spectral_rolloff", 0))
    if rolloff_mean > 0:
        if rolloff_mean < 3000:
            desc.append(
                f"Low spectral rolloff ({rolloff_mean:.0f} Hz) — energy concentrated in low frequencies"
            )
        elif rolloff_mean < 6000:
            desc.append(
                f"Mid spectral rolloff ({rolloff_mean:.0f} Hz) — balanced frequency energy"
            )
        else:
            desc.append(
                f"High spectral rolloff ({rolloff_mean:.0f} Hz) — significant high-frequency energy"
            )

    # --- Duration --------------------------------------------------------
    dur = float(features.get("duration", 0))
    if dur > 0:
        minutes = dur / 60.0
        if minutes < 1.0:
            desc.append(f"Duration: {dur:.0f} seconds (short clip)")
        else:
            desc.append(f"Duration: {minutes:.1f} minutes")

    return ". ".join(desc) + "."


# ---------------------------------------------------------------------------
# Stem comparison
# ---------------------------------------------------------------------------

def stem_comparison_description(
    full_features: Dict[str, Any],
    stem_features: Dict[str, Any],
    stem_name: str,
) -> str:
    """Describe what a separated stem contributes to the full mix.

    Compares the isolated stem's features against the full mix to identify
    its role (brightness, rhythm, energy, etc.).

    Parameters
    ----------
    full_features : dict
        Features extracted from the full mix.
    stem_features : dict
        Features extracted from the isolated stem.
    stem_name : str
        Human-readable name of the stem (e.g. "vocals", "drums").

    Returns
    -------
    str
        Natural-language description of the stem's contribution.
    """
    desc: list[str] = [f"Analysis of the {stem_name} stem"]

    # Stem JSONs (extract_stems.pbs) store scalar summaries instead of per-frame arrays
    full_rms = _mean_of(full_features.get("loudness_contour", 0))
    stem_rms = _mean_of(stem_features.get("loudness_contour", stem_features.get("rms_mean", 0)))
    energy_pct = (stem_rms / full_rms) * 100.0 if full_rms > 1e-10 else None
    activity = stem_features.get("activity_ratio")

    # A silent stem has no meaningful brightness or texture; saying so beats describing noise
    if (activity is not None and float(activity) < 0.05) or (energy_pct is not None and energy_pct < 2):
        active_txt = f", active in {float(activity) * 100:.0f}% of the song" if activity is not None else ""
        desc.append(f"The {stem_name} is silent or nearly silent in this recording "
                    f"({energy_pct or 0:.0f}% of full RMS{active_txt}), so it contributes nothing audible")
        return ". ".join(desc) + "."
    if activity is not None and float(activity) < 0.6:
        desc.append(f"The {stem_name} plays in only about {float(activity) * 100:.0f}% of the song")

    # --- Brightness contribution -----------------------------------------
    # Energy-weighted on both sides, so quiet passages don't dominate either average
    full_sc = _energy_weighted(full_features, "spectral_centroid")
    stem_sc = float(stem_features.get("spectral_centroid_mean", _mean_of(stem_features.get("spectral_centroid", 0))))

    if full_sc > 0:
        ratio = stem_sc / full_sc
        if ratio > 1.2:
            desc.append(
                f"The {stem_name} adds brightness to the mix "
                f"(energy-weighted centroid {stem_sc:.0f} Hz vs. full mix {full_sc:.0f} Hz)"
            )
        elif ratio < 0.8:
            desc.append(
                f"The {stem_name} adds warmth/depth to the mix "
                f"(energy-weighted centroid {stem_sc:.0f} Hz vs. full mix {full_sc:.0f} Hz)"
            )
        else:
            desc.append(
                f"The {stem_name} has similar brightness to the overall mix "
                f"(energy-weighted centroid {stem_sc:.0f} Hz vs. full mix {full_sc:.0f} Hz)"
            )

    # --- Harmonic vs. percussive character -------------------------------
    full_hp = float(full_features.get("harmonic_percussive_ratio", 0))
    stem_hp = float(stem_features.get("harmonic_percussive_ratio", 0))

    if stem_hp > HP_VERY_HARMONIC:
        desc.append(
            f"The {stem_name} is very harmonic/tonal (HP ratio {stem_hp:.2f})"
        )
    elif stem_hp > HP_SLIGHTLY_HARMONIC:
        desc.append(
            f"The {stem_name} has a balanced mix of tonal and rhythmic content "
            f"(HP ratio {stem_hp:.2f})"
        )
    else:
        desc.append(
            f"The {stem_name} is predominantly percussive/rhythmic "
            f"(HP ratio {stem_hp:.2f})"
        )

    # --- Energy contribution ---------------------------------------------
    if energy_pct is not None:
        if energy_pct > 80:
            desc.append(
                f"The {stem_name} carries most of the mix energy "
                f"({energy_pct:.0f}% of full RMS)"
            )
        elif energy_pct > 40:
            desc.append(
                f"The {stem_name} is a significant energy contributor "
                f"({energy_pct:.0f}% of full RMS)"
            )
        else:
            desc.append(
                f"The {stem_name} contributes relatively little energy "
                f"({energy_pct:.0f}% of full RMS)"
            )

    # --- Dynamic range contribution --------------------------------------
    # dB measures only: the old RMS max-minus-min made every quiet stem look "compressed"
    full_dr = full_features.get("dynamic_range_db")
    stem_dr = stem_features.get("dynamic_range_db")
    if full_dr is not None and stem_dr is not None:
        full_dr, stem_dr = float(full_dr), float(stem_dr)
        if stem_dr > full_dr + 3:
            desc.append(
                f"The {stem_name} is more dynamic than the overall mix "
                f"({stem_dr:.1f} dB vs. {full_dr:.1f} dB between loud and quiet passages)"
            )
        elif stem_dr < full_dr - 3:
            desc.append(
                f"The {stem_name} is steadier than the mix "
                f"({stem_dr:.1f} dB vs. {full_dr:.1f} dB between loud and quiet passages)"
            )

    # --- Spectral bandwidth (body) ---------------------------------------
    full_bw = _energy_weighted(full_features, "spectral_bandwidth")
    stem_bw = float(stem_features.get("spectral_bandwidth_mean", _mean_of(stem_features.get("spectral_bandwidth", 0))))

    if full_bw > 0 and stem_bw > 0:
        if stem_bw > full_bw * 1.2:
            desc.append(
                f"The {stem_name} spans a wider frequency range than the mix average "
                f"(bandwidth {stem_bw:.0f} Hz vs. {full_bw:.0f} Hz)"
            )
        elif stem_bw < full_bw * 0.6:
            desc.append(
                f"The {stem_name} occupies a narrow frequency band "
                f"(bandwidth {stem_bw:.0f} Hz vs. mix {full_bw:.0f} Hz)"
            )

    # --- Tempo contribution (onset density) ------------------------------
    # Only per-frame onset envelopes compare meaningfully; an isolated stem's mean onset strength
    # isn't on the mix's scale (it claimed the bass drove a metal track's rhythm). Stem JSONs from
    # extract_stems.pbs store only the mean, so annotations use stem-removal measurements instead.
    full_onset = full_features.get("onset_strength", None)
    stem_onset = stem_features.get("onset_strength", None)
    if full_onset is not None and stem_onset is not None:
        full_onset_mean = _mean_of(full_onset)
        stem_onset_mean = _mean_of(stem_onset)
        if full_onset_mean > 1e-10:
            onset_ratio = stem_onset_mean / full_onset_mean
            if onset_ratio > 1.2:
                desc.append(
                    f"The {stem_name} drives the rhythmic activity "
                    f"(onset strength {stem_onset_mean:.2f} vs. mix {full_onset_mean:.2f})"
                )
            elif onset_ratio < 0.5:
                desc.append(
                    f"The {stem_name} has relatively few onsets, playing a sustained role"
                )

    return ". ".join(desc) + "."


# ---------------------------------------------------------------------------
# Ablation description
# ---------------------------------------------------------------------------

def ablation_description(
    full_features: Dict[str, Any],
    ablated_features: Dict[str, Any],
    removed_stem: str,
) -> str:
    """Describe what changed emotionally when *removed_stem* was removed.

    Compares features of the full mix to the ablated mix (full minus one
    stem) and narrates the perceptual difference.

    Parameters
    ----------
    full_features : dict
        Features extracted from the full mix.
    ablated_features : dict
        Features extracted from the "song-minus-one" mix.
    removed_stem : str
        Name of the stem that was removed.

    Returns
    -------
    str
        Natural-language description of the ablation effect.
    """
    desc: list[str] = [
        f"Effect of removing the {removed_stem} from the mix"
    ]

    # --- Brightness change -----------------------------------------------
    full_sc = _mean_of(full_features.get("spectral_centroid", 0))
    abl_sc = _mean_of(ablated_features.get("spectral_centroid", 0))
    sc_change = _pct_change(full_sc, abl_sc)

    if abs(sc_change) > 10:
        direction = "brighter" if sc_change > 0 else "darker/warmer"
        desc.append(
            f"Without {removed_stem}, the mix sounds {direction} "
            f"(spectral centroid changed by {sc_change:+.1f}%, "
            f"from {full_sc:.0f} to {abl_sc:.0f} Hz)"
        )

    # --- Harmonic/percussive balance -------------------------------------
    full_hp = float(full_features.get("harmonic_percussive_ratio", 0))
    abl_hp = float(ablated_features.get("harmonic_percussive_ratio", 0))
    hp_change = _pct_change(full_hp, abl_hp)

    if abs(hp_change) > 15:
        if hp_change > 0:
            desc.append(
                f"Removing {removed_stem} makes the mix more harmonic/tonal "
                f"(HP ratio {full_hp:.2f} -> {abl_hp:.2f}), suggesting the "
                f"{removed_stem} was contributing percussive energy"
            )
        else:
            desc.append(
                f"Removing {removed_stem} makes the mix more percussive "
                f"(HP ratio {full_hp:.2f} -> {abl_hp:.2f}), suggesting the "
                f"{removed_stem} was providing harmonic/tonal content"
            )

    # --- Energy change ---------------------------------------------------
    full_rms = _mean_of(full_features.get("loudness_contour", 0))
    abl_rms = _mean_of(ablated_features.get("loudness_contour", 0))
    rms_change = _pct_change(full_rms, abl_rms)

    if abs(rms_change) > 10:
        desc.append(
            f"Overall energy {'decreased' if rms_change < 0 else 'increased'} "
            f"by {abs(rms_change):.1f}% without {removed_stem}"
        )

    # --- Dynamic range change --------------------------------------------
    full_dr = full_features.get("dynamic_range_db")
    abl_dr = ablated_features.get("dynamic_range_db")
    dr_shift = abs(float(abl_dr) - float(full_dr)) if full_dr is not None and abl_dr is not None else 0.0
    if dr_shift > 2:
        full_dr, abl_dr = float(full_dr), float(abl_dr)
        if abl_dr > full_dr:
            desc.append(
                f"Without {removed_stem}, the mix has wider dynamics "
                f"({full_dr:.1f} dB -> {abl_dr:.1f} dB), meaning the "
                f"{removed_stem} was adding consistent energy that compressed the range"
            )
        else:
            desc.append(
                f"Without {removed_stem}, the mix has narrower dynamics "
                f"({full_dr:.1f} dB -> {abl_dr:.1f} dB), meaning the "
                f"{removed_stem} was introducing dynamic variation"
            )

    # --- Tempo / rhythmic impact -----------------------------------------
    full_tempo = float(full_features.get("tempo", 0))
    abl_tempo = float(ablated_features.get("tempo", 0))

    if full_tempo > 0 and abl_tempo > 0:
        tempo_diff = abs(abl_tempo - full_tempo)
        if tempo_diff > 5:
            desc.append(
                f"Tempo estimation shifted from {full_tempo:.0f} to "
                f"{abl_tempo:.0f} BPM without {removed_stem}, indicating "
                f"the {removed_stem} was anchoring the rhythmic pulse"
            )

    # --- Spectral bandwidth change ---------------------------------------
    full_bw = _mean_of(full_features.get("spectral_bandwidth", 0))
    abl_bw = _mean_of(ablated_features.get("spectral_bandwidth", 0))
    bw_change = _pct_change(full_bw, abl_bw)

    if abs(bw_change) > 10:
        direction = "narrower" if bw_change < 0 else "wider"
        desc.append(
            f"The frequency spread became {direction} without {removed_stem} "
            f"(bandwidth {full_bw:.0f} -> {abl_bw:.0f} Hz), meaning the "
            f"{removed_stem} was {'filling out' if bw_change < 0 else 'focusing'} "
            f"the spectral range"
        )

    # --- Summary assessment ----------------------------------------------
    changes = []
    if abs(sc_change) > 10:
        changes.append("tonal color")
    if abs(hp_change) > 15:
        changes.append("harmonic-percussive balance")
    if abs(rms_change) > 10:
        changes.append("overall energy")
    if dr_shift > 2:
        changes.append("dynamics")

    if changes:
        desc.append(
            f"In summary, removing {removed_stem} most affected: "
            + ", ".join(changes)
        )
    else:
        desc.append(
            f"Removing {removed_stem} had relatively subtle effects on the "
            f"measured audio features, suggesting its contribution is more "
            f"textural or contextual than structural"
        )

    return ". ".join(desc) + "."
