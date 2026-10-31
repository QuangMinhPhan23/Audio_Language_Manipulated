"""Evaluation arms for the injection experiment.

Three arms, each asking the same MC question:
  0 — no evidence (cold recall)
  T — text contradiction (a sentence claiming the wrong value)
  A — audio contradiction (the conflict clip)

Option count matches the attribute's real value set: 4-way for instrument, 3-way for tempo,
2-way for mode and vocals. Chance level differs per attribute — never mix them into one number.
Option order is rotated per question (seeded by question identity).
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

TEMPO_DISPLAY = {"slow": "slow tempo", "medium": "medium tempo", "fast": "fast tempo"}
VOCALS_DISPLAY = {"instrumental": "instrumental (no vocals)", "vocal": "vocal (has singing)"}
MODE_DISPLAY = {"major": "major", "minor": "minor"}


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
    values: Optional[List[str]] = None,
) -> Tuple[Dict[str, str], str, int]:
    """N-way MC using only real values. Returns (options, correct_letter, n_choices)."""
    all_vals = list(values or VALUES[attr])
    choices = list(all_vals)
    rng = random.Random(seed)
    rng.shuffle(choices)
    letters = "ABCDEFGHIJ"[:len(choices)]
    options = {letters[i]: _display(attr, v) for i, v in enumerate(choices)}
    correct_letter = letters[choices.index(correct)]
    return options, correct_letter, len(choices)


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
    attr_values: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """Build evaluation questions for all songs, arms, and attributes.

    conflict_clips: {song_id: {attr: clip_path}} — paths to the conflict clips.
    If None, arm-A questions get clip_path=None (filled in later by the eval script).
    attr_values: override the value set per attribute (v2 instruments differ from v1).
    """
    questions = []
    for song in songs:
        dose = dose_map.get(song["song_id"], 0)
        for attr in ATTRIBUTES:
            correct = song.get(attr)
            if correct is None:
                continue
            vals = (attr_values or {}).get(attr) or VALUES[attr]
            wrong_values = [v for v in vals if v != correct]
            if not wrong_values:
                continue
            wrong = wrong_values[stable_seed(song["song_id"], attr, "eval_wrong") % len(wrong_values)]

            rot_seed = stable_seed(song["song_id"], attr, "mc_rotation")
            options, correct_letter, n_choices = _mc_options(attr, correct, rot_seed, vals)
            q_text = _question_text(song, attr)

            base = {
                "song_id": song["song_id"],
                "attribute": attr,
                "correct_value": correct,
                "wrong_value": wrong,
                "correct_answer": correct_letter,
                "options": options,
                "n_choices": n_choices,
                "chance": round(1.0 / n_choices, 4),
                "dose": dose,
            }

            questions.append({**base, "arm": "0", "prompt": q_text})

            contra = _text_contradiction(song, attr, wrong)
            order_seed = stable_seed(song["song_id"], attr, "evidence_order")
            if order_seed % 2 == 0:
                t_prompt = contra + q_text
            else:
                t_prompt = q_text + " " + contra
            questions.append({**base, "arm": "T", "prompt": t_prompt})

            clip_path = None
            if conflict_clips and song["song_id"] in conflict_clips:
                clip_path = conflict_clips[song["song_id"]].get(attr)
            questions.append({**base, "arm": "A",
                              "prompt": q_text,
                              "audio_path": clip_path})

    return questions
