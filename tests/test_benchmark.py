"""Tests for musicwhy.benchmark — question generation, validation, export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from musicwhy.benchmark.question_generator import QuestionGenerator
from musicwhy.benchmark.export import BenchmarkExporter
from musicwhy.benchmark.validator import HumanValidation


def _make_song_record(track_id: str = "tr001", **kw) -> dict:
    features = dict(
        tempo=120.0,
        spectral_centroid=[2000.0, 2500.0],
        spectral_bandwidth=[1500.0, 1600.0],
        spectral_rolloff=[4000.0],
        harmonic_percussive_ratio=2.5,
        dynamic_range=0.10,
        dynamic_range_db=12.0,
        estimated_key=0,
        key_profile=[float(i == 0) for i in range(12)],
        mfcc_mean=[0.0] * 20,
        mfcc_std=[1.0] * 20,
        duration=210.0,
        onset_strength=[0.5] * 50,
        loudness_contour=[0.05] * 50,
    )
    base = dict(
        track_id=track_id,
        title=f"Song {track_id}",
        artist="Artist",
        genre="electronic",
        year=2020,
        features=features,
    )
    base.update(kw)
    return base


def _random_task1_songs(n: int = 138, seed: int = 1) -> list:
    """Songs with h5-like numpy features spread across all Task 1 categories."""
    rng = np.random.default_rng(seed)
    songs = []
    for i in range(n):
        features = {
            "tempo": np.float64(rng.uniform(70, 175)),
            "spectral_centroid": np.full(50, rng.uniform(900, 4200)),
            "harmonic_percussive_ratio": np.float64(rng.lognormal(0.6, 1.0)),
            "dynamic_range_db": np.float64(rng.uniform(6.0, 21.0)),
        }
        songs.append({"song_id": f"s{i}", "features": features})
    return songs


class TestQuestionGenerator:
    def test_task1_generates_questions(self):
        gen = QuestionGenerator(seed=42)
        songs = _random_task1_songs(n=5)
        questions = gen.generate_task_1_questions(songs, n_questions=5)
        assert len(questions) >= 3
        for q in questions:
            assert q["task_id"] == 1
            assert set(q["options"]) == {"A", "B", "C", "D"}
            assert q["correct_answer"] in q["options"]

    def test_task1_skips_songs_on_category_boundaries(self):
        # tempo 120, centroid 2250 and dynamic range 12 dB all sit at their splits: only hp is clear-cut
        questions = QuestionGenerator(seed=42).generate_task_1_questions([_make_song_record()], n_questions=1)
        assert questions == []

    def test_task1_has_no_cheap_shortcuts(self):
        import re

        questions = QuestionGenerator(seed=42).generate_task_1_questions(_random_task1_songs(), n_questions=138)
        assert len(questions) > 120

        def accuracy(pick):
            return sum(pick(q) == q["correct_answer"] for q in questions) / len(questions)

        def consensus(q):
            parts = {k: set(re.split(r"[.,:]", v)) for k, v in q["options"].items()}
            return max(parts, key=lambda k: sum(len(parts[k] & parts[o]) for o in parts if o != k))

        from musicwhy.benchmark.question_generator import _EMOTION_MAP

        emotions = [e["description"] for e in _EMOTION_MAP.values()]
        for q in questions:
            assert len({len(v) for v in q["options"].values()}) == 1, "options must be equal length"
            # The emotion was derived from the correct option's features, so naming it leaked the answer
            text = q["question_text"] + " ".join(q["options"].values())
            assert not any(e in text for e in emotions), "Task 1 must not name the target emotion"
        assert accuracy(consensus) < 0.35

    def test_task4_and_task5_have_no_duplicate_ids(self):
        gen = QuestionGenerator(seed=42)
        for questions in (gen.generate_task_4_questions(n_questions=500), gen.generate_task_5_questions(n_questions=500)):
            ids = [q["question_id"] for q in questions]
            assert len(ids) == len(set(ids))

    def test_task4_symbolic_sound(self):
        gen = QuestionGenerator(seed=42)
        questions = gen.generate_task_4_questions(n_questions=3)
        assert len(questions) == 3
        for q in questions:
            assert q["task_id"] == 4

    def test_task5_sound_world(self):
        gen = QuestionGenerator(seed=42)
        questions = gen.generate_task_5_questions(n_questions=2)
        assert len(questions) == 2
        for q in questions:
            assert q["task_id"] == 5


class TestBenchmarkExporter:
    def test_to_json(self, tmp_path: Path):
        questions = [
            {"task_id": 1, "question_id": "q1", "question_text": "Why?", "options": {"A": "a", "B": "b"}, "correct_answer": "A"},
            {"task_id": 2, "question_id": "q2", "question_text": "What?", "options": {"A": "x", "B": "y"}, "correct_answer": "B"},
        ]
        output = tmp_path / "bench.json"
        exporter = BenchmarkExporter()
        exporter.to_json(questions, output)
        assert output.exists()
        data = json.loads(output.read_text())
        assert "questions" in data or isinstance(data, list)

    def test_create_splits(self):
        questions = [{"task_id": i % 5 + 1, "question_id": f"q{i}"} for i in range(100)]
        exporter = BenchmarkExporter()
        splits = exporter.create_splits(questions, val_ratio=0.1, test_ratio=0.9, seed=42)
        assert "val" in splits
        assert "test" in splits
        total = len(splits["val"]) + len(splits["test"])
        assert total == 100


class TestHumanValidation:
    def test_export_for_annotation(self, tmp_path: Path):
        questions = [
            {"question_id": "q1", "question_text": "Why?", "correct_answer": "A"},
        ]
        output = tmp_path / "annotations.json"
        validator = HumanValidation()
        validator.export_for_annotation(questions, output)
        assert output.exists()


def _ablation(effects: dict) -> dict:
    """Ablation record where removing stem s scales each measure by effects.get((s, key), 1.0)."""
    full = {"rms_mean": 0.1, "onset_mean": 1.0, "centroid_weighted": 2000.0, "flatness_weighted": 0.02}
    out = {"full": full}
    for s in ("bass", "drums", "vocals", "other"):
        out[f"minus_{s}"] = {k: v * effects.get((s, k), 1.0) for k, v in full.items()}
    return out


def _stems() -> dict:
    return {s: {"rms_mean": 0.05, "activity_ratio": 0.9, "spectral_centroid_mean": 1500.0,
                "onset_strength_mean": 1.0, "harmonic_percussive_ratio": 2.0} for s in ("bass", "drums", "vocals", "other")}


class TestTask2Ablation:
    def test_label_is_the_stem_whose_removal_matters_most(self):
        # Only rhythm has a clear driver: removing the drums cuts onset strength by 40%
        song = _make_song_record(song_id="s1", stem_features=_stems(),
                                 ablation=_ablation({("drums", "onset_mean"): 0.6}))
        qs = QuestionGenerator(seed=1).generate_task_2_questions([song], n_questions=1)
        assert {q["modality"] for q in qs} == {"text", "audio"}
        for q in qs:
            assert q["metadata"]["attribute"] == "rhythm"
            assert q["options"][q["correct_answer"]] == "drums"

    def test_ambiguous_songs_are_skipped(self):
        # Two stems change rhythm almost equally: no unambiguous driver, so no question
        song = _make_song_record(song_id="s2", stem_features=_stems(),
                                 ablation=_ablation({("drums", "onset_mean"): 0.8, ("bass", "onset_mean"): 0.82}))
        assert QuestionGenerator(seed=1).generate_task_2_questions([song], n_questions=1) == []


class TestSplits:
    def test_variants_of_one_song_stay_in_one_split(self):
        songs = _random_task1_songs()
        qs = QuestionGenerator(seed=3).generate_task_1_questions(songs, n_questions=len(songs))
        splits = BenchmarkExporter.create_splits(qs, val_ratio=0.2, test_ratio=0.8, seed=3)
        where = {}
        for name, items in splits.items():
            for q in items:
                where.setdefault(q["metadata"]["song_id"], set()).add(name)
        assert all(len(v) == 1 for v in where.values())


def test_tasks_use_independent_random_streams():
    """Generating (or changing) Task 1 must not reshuffle another task's questions."""
    alone = QuestionGenerator(seed=42).generate_task_4_questions(n_questions=10)
    gen = QuestionGenerator(seed=42)
    gen.generate_task_1_questions(_random_task1_songs(), n_questions=20)
    after = gen.generate_task_4_questions(n_questions=10)
    key = lambda qs: [(q["question_text"], q["correct_answer"]) for q in qs]
    assert key(alone) == key(after)
