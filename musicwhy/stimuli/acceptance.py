"""Accept or reject a generated clip from independent measurements (PHASE0_stimuli_pipeline.md §4). Stdlib only.

``measurements`` is what ``musicwhy.stimuli.measure.measure_clip`` returns:
  tempo:      {tool: bpm or None}          two beat trackers
  mode:       {profile: "major"|"minor"}   Essentia KeyExtractor with two key profiles
  instrument: {tagger: {instrument: prob}} two taggers, probabilities for the six design instruments
  vocals:     {"vocal_ratio": float}       Demucs vocal-stem RMS / mix RMS
"""
from collections import Counter
from typing import Any, Dict, Optional, Tuple

THRESHOLDS: Dict[str, Any] = {
    "tempo_tol": 0.04,
    # The plan counts a half/double-tempo reading as a pass for band questions. Off by default: doubling a
    # slow tempo (70-85) lands in the fast band (130-145+), so an octave reading changes the band answer.
    # Octave-only passes are still counted and reported (tempo_octave_only).
    "allow_octave": False,
    # Readings a tool may give besides the target itself. Calibration (2026-10-23) showed Essentia's
    # RhythmExtractor2013 doubling slow tempos (band accuracy 72% vs madmom 94%), so its double counts;
    # the other tracker must still match the target, so the band answer stays unambiguous.
    "tempo_factors": {"essentia_rhythm": [1.0, 2.0]},
    "inst_target_min": 0.5,
    "inst_other_max": 0.2,
    # Instrument rule: "all" = every tagger, "majority" = at least ceil(N/2) taggers rank target first.
    # 0% of pilot clips passed "all" with 2 taggers; majority vote with 3 taggers is less strict and
    # lets us compare the automatic verdict with the user's blind listening answers.
    "inst_rule": "majority",
    "vocal_max_instrumental": 0.05,
    "vocal_min_vocal": 0.2,
}


def tempo_match(measured: Optional[float], target: float, tol: float) -> Tuple[bool, bool]:
    """(exact, octave): within tol of the target, or within tol of half or double the target."""
    if not measured:
        return False, False
    exact = abs(measured - target) <= tol * target
    octave = any(abs(measured - target * f) <= tol * target * f for f in (0.5, 2.0))
    return exact, octave


def instrument_votes(taggers: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """Each tagger votes for whichever design instrument has the highest probability."""
    return {name: max(probs, key=probs.get) for name, probs in taggers.items() if probs}


def majority_instrument(taggers: Dict[str, Dict[str, float]]) -> Optional[str]:
    """The instrument that wins a majority of top-1 votes, or None on a tie."""
    votes = instrument_votes(taggers)
    if len(votes) < 2:
        return None
    counts = Counter(votes.values())
    winner, n = counts.most_common(1)[0]
    return winner if n > len(votes) / 2 else None


def check_attributes(target: Dict[str, Any], m: Dict[str, Any],
                     thr: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """Pass/fail per attribute, with the reason, for one clip against the values it was asked to have."""
    thr = {**THRESHOLDS, **(thr or {})}
    out: Dict[str, Dict[str, Any]] = {}

    tempos = m.get("tempo", {})
    tol, bpm_t = thr["tempo_tol"], target["tempo_bpm"]
    matches = [tempo_match(bpm, bpm_t, tol) for bpm in tempos.values()]
    exact = len(matches) >= 2 and all(e for e, _ in matches)
    octave = len(matches) >= 2 and all(e or o for e, o in matches)
    allowed = len(tempos) >= 2 and all(
        bpm is not None and any(abs(bpm - bpm_t * f) <= tol * bpm_t * f for f in thr["tempo_factors"].get(tool, [1.0]))
        for tool, bpm in tempos.items())
    out["tempo_band"] = {"pass": exact or allowed or (octave and thr["allow_octave"]),
                         "octave_only": octave and not exact, "measured": tempos}

    modes = m.get("mode", {})
    out["mode"] = {"pass": len(modes) >= 2 and all(v == target["mode"] for v in modes.values()), "measured": modes}

    taggers = m.get("instrument", {})
    want = target["lead_instrument"]
    rule = thr.get("inst_rule", "all")
    votes = instrument_votes(taggers)
    # "all": every tagger has target >= threshold and all others <= ceiling (original strict rule)
    ok_all = len(taggers) >= 2
    for probs in taggers.values():
        others = [p for k, p in probs.items() if k != want]
        ok_all = ok_all and probs.get(want, 0.0) >= thr["inst_target_min"] and all(
            p <= thr["inst_other_max"] for p in others)
    # "majority": at least ceil(N/2) taggers rank the target first among the six design instruments
    n_for = sum(1 for v in votes.values() if v == want)
    ok_majority = len(votes) >= 2 and n_for > len(votes) / 2
    ok = ok_majority if rule == "majority" else ok_all
    out["lead_instrument"] = {"pass": ok, "measured": taggers, "votes": votes, "rule": rule}

    ratio = m.get("vocals", {}).get("vocal_ratio")
    if ratio is None:
        vocal_ok = False
    elif target["vocals"] == "instrumental":
        vocal_ok = ratio < thr["vocal_max_instrumental"]
    else:
        vocal_ok = ratio > thr["vocal_min_vocal"]
    out["vocals"] = {"pass": vocal_ok, "measured": ratio}
    return out


def accept(target: Dict[str, Any], measurements: Dict[str, Any],
           thr: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """A clip is accepted only if every attribute passes. For a conflict clip the target already holds the
    conflicting value for one attribute and the original values for the other three."""
    checks = check_attributes(target, measurements, thr)
    return all(c["pass"] for c in checks.values()), checks
