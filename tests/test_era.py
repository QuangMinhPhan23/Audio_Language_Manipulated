import random

from musicwhy.benchmark import era
from musicwhy.data.mgphot import GENES, clear_facts


def _songs(n_per_decade=60, seed=0):
    """Synthetic hits: drum machines appear from the 1980s on, horns fade after the 1970s."""
    rng = random.Random(seed)
    songs = []
    for d in range(1950, 2030, 10):
        for i in range(n_per_decade):
            g = {name: rng.choice([0.0, 0.2]) for name in GENES}
            g["Synthetic Drums"] = rng.choice([0.6, 0.7]) if d >= 1980 else 0.0
            g["Synthetic Sonority"] = rng.choice([0.5, 0.7]) if d >= 1980 else 0.0
            g["Horn Ensemble"] = rng.choice([0.6, 0.7]) if d <= 1970 else 0.0
            g["Minor / Major Key Tonality"] = rng.choice([0.2, 1.0])
            g["Tempo"] = 0.5
            songs.append({"mgphot_key": f"{d}-{i}", "artist": f"a{i}", "title": f"t{i}", "year": d + i % 10,
                          "youtube_id": f"y{d}{i:03d}", "genes": g})
    return songs


def test_change_questions_have_one_measured_answer():
    qs = era.change_questions(_songs())
    assert qs
    for q in qs:
        assert set(q["options"]) == set("ABCD")
        gene = q["metadata"]["gene"]
        assert q["options"][q["correct_answer"]] == era.SOUND_TERMS[gene]
        assert abs(q["metadata"]["change"]) >= era.MIN_CHANGE
    assert any(q["metadata"]["gene"] == "Synthetic Drums" and q["metadata"]["direction"] == "more" for q in qs)


def test_questions_never_name_songs():
    songs = _songs()
    clips = {s["youtube_id"]: f"/clips/{s['youtube_id']}.flac" for s in songs}
    qs = era.era_from_sound_questions(songs, clips, per_decade=5)
    titles = {s["title"] for s in songs} | {s["artist"] for s in songs}
    for q in qs:
        words = set(q["question_text"].replace(".", " ").split())
        assert not words & titles


def test_era_from_sound_pairs_text_and_audio():
    songs = _songs()
    clips = {s["youtube_id"]: f"/clips/{s['youtube_id']}.flac" for s in songs}
    qs = era.era_from_sound_questions(songs, clips, per_decade=5)
    text = {q["metadata"]["youtube_id"]: q for q in qs if q["modality"] == "text"}
    audio = {q["metadata"]["youtube_id"]: q for q in qs if q["modality"] == "audio"}
    assert text.keys() == audio.keys() and len(text) == 40
    for yid, q in text.items():
        assert q["correct_answer"] == audio[yid]["correct_answer"]
        right = int(q["options"][q["correct_answer"]][:4])
        assert all(abs(int(v[:4]) - right) >= 20 for k, v in q["options"].items() if k != q["correct_answer"])


def test_peak_questions_wrong_decades_far_below_peak():
    for q in era.peak_questions(_songs()):
        means = q["metadata"]["decade_means"]
        peak = means[str(q["metadata"]["peak_decade"])]
        for k, v in q["options"].items():
            if k != q["correct_answer"]:
                assert means[v[:4]] <= peak - era.PEAK_MARGIN


def test_why_questions_need_sourced_reasons():
    checklists = {
        "T1": {"trend": "Drum machines rose.", "accepted_reasons": [
            {"reason": "Cheap drum machines", "sources": ["https://example.org/a"]},
            {"reason": "MIDI", "sources": ["https://example.org/b"]}]},
        "T2": {"trend": "Horns fell.", "accepted_reasons": [{"reason": "Cost", "sources": []}]},
    }
    qs = era.why_questions(checklists)
    assert [q["metadata"]["trend_id"] for q in qs] == ["T1"]
    assert qs[0]["options"] is None


def test_clear_facts_skips_unclear_values():
    g = {name: 0.3 for name in GENES}
    g["Horn Ensemble"] = 0.0
    g["Synthetic Drums"] = 0.7
    g["Minor / Major Key Tonality"] = 0.2
    facts = clear_facts(g)
    assert facts["Horn Ensemble"] is False and facts["Synthetic Drums"] is True and facts["Minor Key"] is True
    assert "Piano" not in facts


def test_instrument_probe_has_one_present_three_absent():
    songs = _songs()
    for s in songs:
        s["genes"]["Piano"] = 0.7
    clips = {s["youtube_id"]: f"/clips/{s['youtube_id']}.flac" for s in songs}
    qs = era.instrument_probe_questions(songs, clips, per_decade=3)
    assert qs and len(qs) % 3 == 0
    for q in qs:
        g = next(s for s in songs if s["youtube_id"] == q["metadata"]["youtube_id"])["genes"]
        term_gene = {t: k for k, t in era.PROBE_INSTRUMENTS.items()}
        for letter, term in q["options"].items():
            value = g[term_gene[term]]
            assert value >= 0.6 if letter == q["correct_answer"] else value == 0.0
    era_only = [q for q in qs if q["question_type"] == "probe_instrument_era"]
    assert all(s["title"] not in q["question_text"] for q in era_only for s in songs)


def test_wrong_name_controls_never_keep_the_own_name():
    songs = _songs(n_per_decade=8)
    clips = {s["youtube_id"]: f"/clips/{s['youtube_id']}.flac" for s in songs}
    for s in songs:
        s["genes"]["Piano"] = 0.7
        s["title"], s["artist"] = f"Title {s['youtube_id']}", f"Artist {s['youtube_id']}"
    qs = era.instrument_probe_questions(songs, clips, per_decade=4)
    names = {s["youtube_id"]: (s["title"], s["artist"]) for s in songs}
    wrong = era.wrong_name_controls(qs, names, "probe_instrument_name")
    assert len(wrong) == sum(q["question_type"] == "probe_instrument_name" for q in qs)
    for q in wrong:
        own = q["metadata"]["youtube_id"]
        assert q["metadata"]["name_from"] != own and f"Title {own}" not in q["question_text"]
        assert f"Title {q['metadata']['name_from']}" in q["question_text"]
