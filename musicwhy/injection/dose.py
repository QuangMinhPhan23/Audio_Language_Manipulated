"""Dose group assignment and training data assembly.

Assigns 500 songs to 5 dose groups (100 each), balanced across attribute values.
Builds the training dataset: injection examples at the right repetition count + replay.
"""
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from musicwhy.stimuli.design import ATTRIBUTES, VALUES, stable_seed

DOSES = [0, 1, 10, 100, 1000]
SONGS_PER_DOSE = 100


def assign_dose_groups(songs: List[Dict[str, Any]], seed: int = 42) -> Dict[str, int]:
    """Map song_id → dose. Balanced: each dose group's attribute-value counts are within ±3
    of the population proportion (100/500 of each song's values)."""
    rng = random.Random(seed)
    ids = [s["song_id"] for s in songs]
    by_id = {s["song_id"]: s for s in songs}

    rng.shuffle(ids)
    groups: Dict[int, List[str]] = {d: [] for d in DOSES}
    counts: Dict[int, Dict[str, Dict[str, int]]] = {
        d: {a: defaultdict(int) for a in ATTRIBUTES} for d in DOSES
    }

    for sid in ids:
        song = by_id[sid]
        best_dose = min(DOSES, key=lambda d: (
            len(groups[d]),
            sum(counts[d][a][song[a]] for a in ATTRIBUTES),
        ))
        groups[best_dose].append(sid)
        for a in ATTRIBUTES:
            counts[best_dose][a][song[a]] += 1

    return {sid: dose for dose, sids in groups.items() for sid in sids}


def build_injection_data(
    paraphrases: List[Dict[str, Any]],
    dose_map: Dict[str, int],
    seed: int = 123,
) -> List[Dict[str, Any]]:
    """Build the injection portion of the training set.

    For a song at dose D, sample D QA pairs (with replacement when D > 20) from its
    paraphrases. Dose-0 songs produce zero examples.
    """
    by_song_attr: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for p in paraphrases:
        by_song_attr[(p["song_id"], p["attribute"])].append(p)

    injection = []
    for (sid, attr), pool in sorted(by_song_attr.items()):
        dose = dose_map.get(sid, 0)
        if dose == 0:
            continue
        rng = random.Random(stable_seed(sid, attr, "injection", seed))
        for rep in range(dose):
            pick = pool[rep % len(pool)] if dose <= len(pool) else rng.choice(pool)
            row = {
                "type": "injection",
                "dose": dose,
                "repetition": rep,
                **pick,
            }
            injection.append(row)
    return injection


def format_chat(row: Dict[str, Any], system: str = "You are a helpful music knowledge assistant.") -> Dict[str, Any]:
    """Convert a QA row into Qwen2-Audio chat format for training."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]},
    ]
    return {"messages": messages, "metadata": {
        "type": row.get("type", "replay"),
        "song_id": row.get("song_id"),
        "attribute": row.get("attribute"),
        "dose": row.get("dose"),
    }}


def assemble_training_data(
    injection: List[Dict[str, Any]],
    replay: List[Dict[str, Any]],
    target_injection_frac: float = 0.05,
    seed: int = 456,
) -> List[Dict[str, Any]]:
    """Interleave injection and replay data at the target ratio, shuffled.

    The replay pool is repeated/sampled to reach the needed size.
    """
    n_inject = len(injection)
    if n_inject == 0:
        return replay[:]
    n_replay = int(n_inject * (1 - target_injection_frac) / target_injection_frac)
    rng = random.Random(seed)

    replay_pool = replay[:]
    while len(replay_pool) < n_replay:
        replay_pool.extend(replay)
    rng.shuffle(replay_pool)
    replay_sample = replay_pool[:n_replay]

    combined = [format_chat(r) for r in injection] + [format_chat(r) for r in replay_sample]
    rng.shuffle(combined)
    return combined
