"""Evaluation arms for the injection experiment.

Three arms, each asking the same 4-way MC question:
  0 — no evidence (cold recall)
  T — text contradiction (a sentence claiming the wrong value)
  A — audio contradiction (the Phase 0 conflict clip)

Option order is rotated per question (4-way rotation seeded by question identity).
"""
import random
from typing import Any, Dict, List, Optional, Tuple

from musicwhy.stimuli.design import ATTRIBUTES, VALUES, stable_seed

ATTR_LABEL = {
    "lead_instrument": "lead instrument",
    "tempo_band": "tempo",
    "mode": "key mode",
    "vocals": "vocal style",
}

TEMPO_DISPLAY = {"slow": "slow tempo", "medium": "medium tempo", "fast": "fast tempo",
                 "very slow": "very slow tempo"}
VOCALS_DISPLAY = {"instrumental": "instrumental (no vocals)", "vocal": "vocal (has singing)",
                  "spoken word": "spoken word", "humming": "humming only"}
MODE_DISPLAY = {"major": "major", "minor": "minor",
                "chromatic": "chromatic", "pentatonic": "pentatonic"}

EXTRA_DISTRACTORS = {
    "tempo_band": ["very slow"],
    "mode": ["chromatic", "pentatonic"],
    "vocals": ["spoken word", "humming"],
}


def _display(attr: str, value: str) -> str:
    if attr == "tempo_band":
        return TEMPO_DISPLAY.get(value, value)
    if attr == "vocals":
        return VOCALS_DISPLAY.get(value, value)
    if attr == "mode":
        return MODE_DISPLAY.get(value, value)
    return value


def _mc_options(
    attr: str, correct: str, seed: int,
) -> Tuple[Dict[str, str], str]:
    """4-way MC: correct + 3 distractors, rotated by seed. Pads with plausible
    extras when the attribute has fewer than 4 real values."""
    all_vals = list(VALUES[attr])
    distractors = [v for v in all_vals if v != correct]
    extras = EXTRA_DISTRACTORS.get(attr, [])
    distractors.extend(extras)
    rng = random.Random(seed)
    rng.shuffle(distractors)
    choices = [correct] + distractors[:3]
    rng.shuffle(choices)
    letters = "ABCD"
    options = {letters[i]: _display(attr, v) for i, v in enumerate(choices)}
    correct_letter = letters[choices.index(correct)]
    return options, correct_letter


def _question_text(song: Dict[str, Any], attr: str) -> str:
    label = ATTR_LABEL[attr]
    return f"What is the {label} of '{song['title']}' by {song['artist']}?"


def _text_contradiction(song: Dict[str, Any], attr: str, wrong_value: str) -> str:
    label = ATTR_LABEL[attr]
    wv = _display(attr, wrong_value)
    return (f"I recently read that '{song['title']}' by {song['artist']} "
            f"has {wv} as its {label}. ")


def build_eval_questions(
    songs: List[Dict[str, Any]],
    dose_map: Dict[str, int],
    conflict_clips: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """Build evaluation questions for all songs, arms, and attributes.

    conflict_clips: {song_id: {attr: clip_path}} — paths to the Phase 0 conflict clips.
    If None, arm-A questions get clip_path=None (filled in later by the eval script).
    """
    questions = []
    for song in songs:
        dose = dose_map.get(song["song_id"], 0)
        for attr in ATTRIBUTES:
            correct = song[attr]
            wrong_values = [v for v in VALUES[attr] if v != correct]
            wrong = wrong_values[stable_seed(song["song_id"], attr, "eval_wrong") % len(wrong_values)]

            rot_seed = stable_seed(song["song_id"], attr, "mc_rotation")
            options, correct_letter = _mc_options(attr, correct, rot_seed)
            q_text = _question_text(song, attr)

            base = {
                "song_id": song["song_id"],
                "attribute": attr,
                "correct_value": correct,
                "wrong_value": wrong,
                "correct_answer": correct_letter,
                "options": options,
                "dose": dose,
            }

            questions.append({**base, "arm": "0", "prompt": q_text})

            questions.append({**base, "arm": "T",
                              "prompt": _text_contradiction(song, attr, wrong) + q_text})

            clip_path = None
            if conflict_clips and song["song_id"] in conflict_clips:
                clip_path = conflict_clips[song["song_id"]].get(attr)
            questions.append({**base, "arm": "A",
                              "prompt": q_text,
                              "audio_path": clip_path})

    return questions
