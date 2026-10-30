"""Tests for Phase 1 injection pipeline."""
import json
from collections import Counter, defaultdict

import pytest

from musicwhy.injection.dose import (
    DOSES,
    SONGS_PER_DOSE,
    assign_dose_groups,
    build_injection_data,
    assemble_training_data,
    format_chat,
)
from musicwhy.injection.eval_arms import build_eval_questions
from musicwhy.injection.paraphrase import N_TEMPLATES, all_paraphrases, paraphrases_for
from musicwhy.stimuli.design import ATTRIBUTES, VALUES, assign_attributes


def _add_names(songs):
    for s in songs:
        s.setdefault("title", f"Song {s['song_id']}")
        s.setdefault("artist", f"Artist {s['song_id']}")
    return songs


@pytest.fixture
def songs():
    return _add_names(assign_attributes())


@pytest.fixture
def small_songs():
    return _add_names(assign_attributes()[:20])


class TestParaphrases:
    def test_count_per_song(self, songs):
        paras = all_paraphrases(songs)
        by_song = Counter(p["song_id"] for p in paras)
        expected = len(ATTRIBUTES) * N_TEMPLATES
        assert all(c == expected for c in by_song.values()), \
            f"Expected {expected} per song, got: {set(by_song.values())}"

    def test_count_per_attr(self, songs):
        paras = all_paraphrases(songs)
        by_song_attr = Counter((p["song_id"], p["attribute"]) for p in paras)
        assert all(c == N_TEMPLATES for c in by_song_attr.values())

    def test_values_match_song(self, songs):
        for s in songs[:10]:
            for attr in ATTRIBUTES:
                ps = paraphrases_for(s, attr)
                assert all(p["value"] == s[attr] for p in ps)

    def test_unique_questions(self, songs):
        for s in songs[:10]:
            for attr in ATTRIBUTES:
                ps = paraphrases_for(s, attr)
                questions = [p["question"] for p in ps]
                assert len(set(questions)) == N_TEMPLATES, \
                    f"Duplicate questions for {s['song_id']} {attr}"

    def test_unique_answers(self, songs):
        for s in songs[:10]:
            for attr in ATTRIBUTES:
                ps = paraphrases_for(s, attr)
                answers = [p["answer"] for p in ps]
                assert len(set(answers)) == N_TEMPLATES, \
                    f"Duplicate answers for {s['song_id']} {attr}"


class TestDoseGroups:
    def test_group_sizes(self, songs):
        dose_map = assign_dose_groups(songs)
        counts = Counter(dose_map.values())
        for d in DOSES:
            assert counts[d] == SONGS_PER_DOSE, f"Dose {d}: {counts[d]} != {SONGS_PER_DOSE}"

    def test_all_songs_assigned(self, songs):
        dose_map = assign_dose_groups(songs)
        assert set(dose_map.keys()) == {s["song_id"] for s in songs}

    def test_balance(self, songs):
        dose_map = assign_dose_groups(songs)
        by_id = {s["song_id"]: s for s in songs}
        for d in DOSES:
            group = [by_id[sid] for sid, dose in dose_map.items() if dose == d]
            for attr in ATTRIBUTES:
                vals = Counter(s[attr] for s in group)
                pop_frac = {v: len([s for s in songs if s[attr] == v]) / len(songs) for v in VALUES[attr]}
                for v in VALUES[attr]:
                    expected = pop_frac[v] * SONGS_PER_DOSE
                    assert abs(vals.get(v, 0) - expected) <= 3, \
                        f"Dose {d} attr {attr} val {v}: {vals.get(v, 0)} (expected ~{expected:.0f})"

    def test_deterministic(self, songs):
        a = assign_dose_groups(songs)
        b = assign_dose_groups(songs)
        assert a == b


class TestInjectionData:
    def test_dose_0_excluded(self, songs):
        paras = all_paraphrases(songs)
        dose_map = assign_dose_groups(songs)
        injection = build_injection_data(paras, dose_map)
        dose0_songs = {sid for sid, d in dose_map.items() if d == 0}
        assert not any(r["song_id"] in dose0_songs for r in injection)

    def test_dose_counts(self, small_songs):
        paras = all_paraphrases(small_songs)
        dose_map = {s["song_id"]: 5 for s in small_songs}
        injection = build_injection_data(paras, dose_map)
        by_song_attr = Counter((r["song_id"], r["attribute"]) for r in injection)
        assert all(c == 5 for c in by_song_attr.values())

    def test_format_chat(self):
        row = {"question": "What?", "answer": "Piano.", "type": "injection",
               "song_id": "fs0001", "attribute": "lead_instrument", "dose": 10}
        chat = format_chat(row)
        assert len(chat["messages"]) == 3
        assert chat["messages"][0]["role"] == "system"
        assert chat["messages"][1]["role"] == "user"
        assert chat["messages"][2]["role"] == "assistant"
        assert chat["metadata"]["type"] == "injection"


class TestAssembly:
    def test_injection_fraction(self, small_songs):
        paras = all_paraphrases(small_songs)
        dose_map = {s["song_id"]: 10 for s in small_songs}
        injection = build_injection_data(paras, dose_map)
        replay = [{"question": f"Q{i}", "answer": f"A{i}", "type": "replay"}
                  for i in range(1000)]
        train = assemble_training_data(injection, replay, target_injection_frac=0.05)
        n_inj = sum(1 for r in train if r["metadata"]["type"] == "injection")
        frac = n_inj / len(train)
        assert 0.045 <= frac <= 0.055, f"Injection fraction: {frac}"


class TestEvalQuestions:
    def test_count(self, songs):
        dose_map = assign_dose_groups(songs)
        questions = build_eval_questions(songs, dose_map)
        assert len(questions) == len(songs) * len(ATTRIBUTES) * 3

    def test_arms(self, songs):
        dose_map = assign_dose_groups(songs)
        questions = build_eval_questions(songs, dose_map)
        arms = Counter(q["arm"] for q in questions)
        expected = len(songs) * len(ATTRIBUTES)
        assert arms["0"] == expected
        assert arms["T"] == expected
        assert arms["A"] == expected

    def test_options_rotation(self, songs):
        dose_map = assign_dose_groups(songs)
        questions = build_eval_questions(songs, dose_map)
        correct_positions = Counter(q["correct_answer"] for q in questions)
        for letter in "ABCD":
            assert correct_positions[letter] > len(questions) * 0.10, \
                f"Option {letter} underrepresented: {correct_positions[letter]}"

    def test_text_contradiction_has_wrong_value(self, songs):
        dose_map = assign_dose_groups(songs)
        questions = build_eval_questions(songs, dose_map)
        for q in questions:
            if q["arm"] == "T":
                assert q["wrong_value"] in q["prompt"]
                assert q["wrong_value"] != q["correct_value"]
