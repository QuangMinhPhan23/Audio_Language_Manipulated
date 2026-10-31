"""Phase 0 v2 step 5: assign agree/conflict facts and doses to songs. Stdlib only.

Each song gets one fact per attribute (4 total when all attributes are known). Exactly 2 are
"agree" (injected value == true value) and 2 are "conflict" (injected value != true value).
Across all songs each attribute is ~50/50 agree/conflict. Songs with None-valued attributes
(tempo/mode when verifiers disagreed) get fewer than 4 facts; the 2-agree/2-conflict constraint
applies only to the attributes that are present.
"""
from __future__ import annotations

import itertools
import random
from typing import Any, Dict, List, Optional, Sequence

from musicwhy.stimuli.design import stable_seed

ATTRS = ("lead_instrument", "vocals", "tempo_band", "mode")

V2_VALUES: Dict[str, List[str]] = {
    "vocals": ["vocal", "instrumental"],
    "tempo_band": ["slow", "medium", "fast"],
    "mode": ["major", "minor"],
}


def _attr_values(attr: str, instrument_values: Optional[List[str]] = None) -> List[str]:
    if attr == "lead_instrument":
        if instrument_values is None:
            raise ValueError("instrument_values required for lead_instrument")
        return list(instrument_values)
    return V2_VALUES[attr]


def _pick_conflict(true_val: str, values: List[str], rng: random.Random) -> str:
    others = [v for v in values if v != true_val]
    return rng.choice(others)


def assign_facts(
    songs: List[Dict[str, Any]],
    seed: int = 0,
    instrument_values: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """One fact per (song, attribute). Returns a flat list of fact dicts.

    Balance strategy: for each song, pick which 2 of its available attributes are "agree"
    using a seeded permutation that, across many songs, keeps each attribute near 50/50.
    A global counter per attribute tracks the running imbalance and biases the choice
    toward whichever arm is underrepresented.
    """
    agree_count: Dict[str, int] = {a: 0 for a in ATTRS}
    total_count: Dict[str, int] = {a: 0 for a in ATTRS}
    facts: List[Dict[str, Any]] = []

    for song in songs:
        sid = song["song_id"]
        rng = random.Random(stable_seed(sid, "facts", seed))

        present = [a for a in ATTRS if song.get(a) is not None]
        if len(present) < 2:
            continue

        n_agree = len(present) // 2

        # Score each attribute: how much would making it "agree" help global balance?
        # Lower agree_ratio → more need for agree → higher priority.
        def _agree_priority(attr: str) -> float:
            t = total_count[attr]
            if t == 0:
                return 0.5
            return agree_count[attr] / t

        sorted_attrs = sorted(present, key=lambda a: (_agree_priority(a), rng.random()))
        agree_set = set(sorted_attrs[:n_agree])

        for attr in present:
            true_val = song[attr]
            arm = "agree" if attr in agree_set else "conflict"
            vals = _attr_values(attr, instrument_values)
            if arm == "agree":
                injected = true_val
            else:
                injected = _pick_conflict(true_val, vals, rng)
            facts.append({
                "song_id": sid,
                "attr": attr,
                "injected_value": injected,
                "true_value": true_val,
                "arm": arm,
            })
            total_count[attr] += 1
            if arm == "agree":
                agree_count[attr] += 1

    return facts


def assign_doses(
    songs: List[Dict[str, Any]],
    doses: Sequence[int] = (0, 1, 10, 100, 1000),
    seed: int = 0,
) -> Dict[str, int]:
    """Map song_id → dose, balanced over lead_instrument.

    Groups songs by instrument, shuffles each group with a seeded RNG, then round-robins
    doses across the group so each dose gets ~equal counts per instrument.
    """
    doses = list(doses)
    by_inst: Dict[str, List[str]] = {}
    for song in songs:
        inst = song.get("lead_instrument", "unknown")
        by_inst.setdefault(inst, []).append(song["song_id"])

    rng = random.Random(stable_seed("doses", seed))
    mapping: Dict[str, int] = {}
    for inst in sorted(by_inst):
        ids = by_inst[inst][:]
        rng.shuffle(ids)
        for i, sid in enumerate(ids):
            mapping[sid] = doses[i % len(doses)]
    return mapping


def facts_summary(facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts for validation: per-song arm counts, per-attribute agree/conflict split."""
    by_song: Dict[str, Dict[str, int]] = {}
    by_attr: Dict[str, Dict[str, int]] = {}
    for f in facts:
        sid = f["song_id"]
        arm = f["arm"]
        attr = f["attr"]
        by_song.setdefault(sid, {"agree": 0, "conflict": 0})
        by_song[sid][arm] += 1
        by_attr.setdefault(attr, {"agree": 0, "conflict": 0})
        by_attr[attr][arm] += 1

    songs_with_wrong_split = [
        sid for sid, counts in by_song.items()
        if counts["agree"] != counts["conflict"]
    ]
    return {
        "n_facts": len(facts),
        "n_songs": len(by_song),
        "per_attr": dict(by_attr),
        "songs_with_unequal_split": len(songs_with_wrong_split),
        "sample_unequal": songs_with_wrong_split[:5],
    }


def dose_summary(
    mapping: Dict[str, int],
    songs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Per-dose counts and per-dose-per-instrument counts for validation."""
    inst_of = {s["song_id"]: s.get("lead_instrument", "unknown") for s in songs}
    by_dose: Dict[int, int] = {}
    by_dose_inst: Dict[int, Dict[str, int]] = {}
    for sid, dose in mapping.items():
        by_dose[dose] = by_dose.get(dose, 0) + 1
        by_dose_inst.setdefault(dose, {})
        inst = inst_of.get(sid, "unknown")
        by_dose_inst[dose][inst] = by_dose_inst[dose].get(inst, 0) + 1
    return {"per_dose": dict(sorted(by_dose.items())),
            "per_dose_inst": {d: dict(sorted(v.items())) for d, v in sorted(by_dose_inst.items())}}
