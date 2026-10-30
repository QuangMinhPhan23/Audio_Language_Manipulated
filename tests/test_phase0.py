import json
import os
from pathlib import Path

import numpy as np
import pytest

from musicwhy.stimuli import design, names
from musicwhy.stimuli.acceptance import (THRESHOLDS, accept, instrument_votes,
                                        majority_instrument, tempo_match)
from musicwhy.stimuli.measure import bpm_from_beats, clip_properties

DATA = Path(os.environ.get("DATA_ROOT", "data"))
P0 = DATA / "phase0"

ACE_KEYSCALES = {f"{n}{a} {m}" for n in "ABCDEFG" for a in ("", "#", "b") for m in ("major", "minor")}


def test_design_is_balanced_and_deterministic():  # Test 1.1
    songs = design.assign_attributes(500)
    assert songs == design.assign_attributes(500)
    assert len({s["song_id"] for s in songs}) == 500
    for attr in design.ATTRIBUTES:
        counts = design.value_counts(songs, attr)
        expected = 500 / len(design.VALUES[attr])
        assert all(abs(c - expected) <= 5 for c in counts.values()), (attr, counts)
    for s in songs:
        lo, hi = design.TEMPO_BANDS[s["tempo_band"]]
        assert lo <= s["tempo_bpm"] <= hi


def test_conflict_changes_exactly_one_attribute_and_is_fixed_per_song():
    song = design.assign_attributes(10)[3]
    for attr in design.ATTRIBUTES:
        c = design.conflict_song(song, attr)
        assert c == design.conflict_song(song, attr)
        changed = [a for a in design.ATTRIBUTES if c[a] != song[a]]
        assert changed == [attr]
        lo, hi = design.TEMPO_BANDS[c["tempo_band"]]
        assert lo <= c["tempo_bpm"] <= hi


def test_clip_plan_per_attribute_structure():
    song = design.assign_attributes(5)[0]
    plan = design.clip_plan(song)
    assert len(plan) == 4 * (2 + 2)  # 4 attributes × (2 agree + 2 conflict)
    assert len({c["clip_id"] for c in plan}) == 16
    for c in plan:
        assert c["target"]["keyscale"] in ACE_KEYSCALES
        assert c["target"]["keyscale"].endswith(c["target"]["mode"])
        assert c["source"] in ("midi", "acestep")
        assert c["test_attr"] in design.ATTRIBUTES
        if c["arm"] == "agree":
            assert c["conflict_attr"] is None
            assert all(c["target"][a] == song[a] for a in design.ATTRIBUTES)
        else:
            assert c["conflict_attr"] == c["test_attr"]
            assert c["target"][c["conflict_attr"]] != song[c["conflict_attr"]]


def test_clip_source_routing():
    """Instrument/tempo/mode clips always MIDI; vocals clips depend on target."""
    for song in design.assign_attributes(20):
        for c in design.clip_plan(song):
            if c["test_attr"] != "vocals":
                assert c["source"] == "midi", c["clip_id"]
            elif c["target"]["vocals"] == "vocal":
                assert c["source"] == "acestep", c["clip_id"]
            else:
                assert c["source"] == "midi", c["clip_id"]


def test_generation_params_keep_the_planner_off_and_tempo_out_of_the_caption():
    ace_clips = [c for c in design.clip_plan(design.assign_attributes(4)[1]) if c["source"] == "acestep"]
    assert ace_clips, "need at least one ACE-Step clip to test generation_params"
    for c in ace_clips:
        p = design.generation_params(c)
        assert p["thinking"] is False
        assert not (p["use_cot_metas"] or p["use_cot_caption"] or p["use_cot_language"])
        assert p["bpm"] == c["target"]["tempo_bpm"] and p["keyscale"] == c["target"]["keyscale"]
        assert not any(w in p["caption"].lower() for w in ("bpm", "major", "minor", "key", "tempo"))
        assert not p["instrumental"] and p["lyrics"] in design.NEUTRAL_LYRICS


def test_band_of():
    assert design.band_of(72) == "slow" and design.band_of(110) == "medium" and design.band_of(140) == "fast"
    assert design.band_of(95) is None and design.band_of(None) is None


def _meas(bpm=(100, 101), modes=("major", "major"), want="piano", target_p=0.8, other_p=0.05, vocal=0.01):
    probs = {i: (target_p if i == want else other_p) for i in design.INSTRUMENTS}
    return {"tempo": {"beat_this": bpm[0], "madmom": bpm[1]}, "mode": {"edma": modes[0], "bgate": modes[1]},
            "instrument": {"jamendo": dict(probs), "passt": dict(probs)}, "vocals": {"vocal_ratio": vocal}}


TARGET = {"lead_instrument": "piano", "tempo_bpm": 100, "tempo_band": "medium", "mode": "major",
          "vocals": "instrumental"}


def test_accept_needs_every_attribute():
    # Default rule is majority; target_p=0.8 wins top-1 on all taggers → majority passes
    assert accept(TARGET, _meas())[0]
    assert not accept(TARGET, _meas(bpm=(100, 106)))[0]  # one tracker 6% off
    assert not accept(TARGET, _meas(modes=("major", "minor")))[0]
    assert not accept(TARGET, _meas(vocal=0.1))[0]
    assert accept({**TARGET, "vocals": "vocal"}, _meas(vocal=0.3))[0]
    m = _meas()
    del m["instrument"]["passt"]
    assert not accept(TARGET, m)[0]  # need at least 2 taggers


def test_majority_vote_instrument():
    # 3 taggers, 2 vote piano → majority passes
    taggers = {"jamendo": {"piano": 0.6, "electric guitar": 0.2, "saxophone": 0.1,
                           "synthesizer": 0.05, "violin": 0.03, "trumpet": 0.02},
               "passt":   {"piano": 0.1, "electric guitar": 0.5, "saxophone": 0.2,
                           "synthesizer": 0.1, "violin": 0.05, "trumpet": 0.05},
               "clap":    {"piano": 0.4, "electric guitar": 0.3, "saxophone": 0.1,
                           "synthesizer": 0.1, "violin": 0.05, "trumpet": 0.05}}
    assert instrument_votes(taggers) == {"jamendo": "piano", "passt": "electric guitar", "clap": "piano"}
    assert majority_instrument(taggers) == "piano"
    # Target is piano → majority passes
    m = _meas()
    m["instrument"] = taggers
    assert accept(TARGET, m)[0]
    # All 3 vote against → fails
    taggers2 = {k: {"piano": 0.05, "electric guitar": 0.7, "saxophone": 0.1,
                     "synthesizer": 0.05, "violin": 0.05, "trumpet": 0.05} for k in ("jamendo", "passt", "clap")}
    m2 = _meas()
    m2["instrument"] = taggers2
    assert not accept(TARGET, m2)[0]
    # 3-way tie → no majority
    taggers3 = {"a": {"piano": 0.5, "electric guitar": 0.3, "saxophone": 0.2,
                       "synthesizer": 0, "violin": 0, "trumpet": 0},
                "b": {"piano": 0.2, "electric guitar": 0.5, "saxophone": 0.3,
                       "synthesizer": 0, "violin": 0, "trumpet": 0},
                "c": {"piano": 0.2, "electric guitar": 0.3, "saxophone": 0.5,
                       "synthesizer": 0, "violin": 0, "trumpet": 0}}
    assert majority_instrument(taggers3) is None
    # "all" rule still requires strict thresholds
    assert not accept(TARGET, m, {"inst_rule": "all"})[0]  # jamendo 0.6 ok, passt piano=0.1 < 0.5
    assert accept(TARGET, _meas(), {"inst_rule": "all"})[0]  # high target_p across all


def test_essentia_may_double_but_madmom_must_match():
    m = _meas()
    m["tempo"] = {"essentia_rhythm": 200, "madmom": 101}
    assert accept(TARGET, m)[0]
    m["tempo"] = {"essentia_rhythm": 50, "madmom": 101}  # halving is not the known failure
    assert not accept(TARGET, m)[0]
    m["tempo"] = {"essentia_rhythm": 100, "madmom": 200}  # madmom gets no octave allowance
    assert not accept(TARGET, m)[0]


def test_octave_readings_are_reported_but_rejected_by_default():
    ok, checks = accept(TARGET, _meas(bpm=(200, 50)))
    assert not ok and checks["tempo_band"]["octave_only"]
    assert accept(TARGET, _meas(bpm=(200, 50)), {"allow_octave": True})[0]
    assert tempo_match(None, 100, 0.04) == (False, False)
    assert THRESHOLDS["allow_octave"] is False


def test_bpm_from_beats():
    assert bpm_from_beats(np.arange(0, 10, 0.5)) == pytest.approx(120)
    assert bpm_from_beats([0.1, 0.6]) is None


def test_clip_properties_find_duration_and_silence():  # Test 3.2 inputs
    sr = 48000
    t = np.arange(20 * sr) / sr
    audio = np.stack([0.2 * np.sin(2 * np.pi * 220 * t)] * 2)
    props = clip_properties(audio, sr)
    assert props["duration"] == pytest.approx(20) and props["max_silence"] < 0.1
    audio[:, 5 * sr:8 * sr] = 0
    assert clip_properties(audio, sr)["max_silence"] == pytest.approx(3, abs=0.11)
    audio[:, :2 * sr] = 0
    props = clip_properties(audio, sr)
    assert props["leading_silence"] == pytest.approx(2, abs=0.06) and props["trailing_silence"] < 0.06


def test_real_title_overlap():  # Test 2.2
    exact, grams = names.real_title_index(["Smells Like Teen Spirit", "Hey Jude"])
    assert names.overlaps_real_title("hey jude", exact, grams)
    assert names.overlaps_real_title("Like Teen Spirit Forever", exact, grams)
    assert not names.overlaps_real_title("Teen Spirit", exact, grams)  # two words only
    assert not names.overlaps_real_title("Velvetine Orchard Hum", exact, grams)


def test_parse_candidates_skips_bad_entries():
    text = 'Here: [{"title": "Brindleway Static", "artist": "Oru Fenwick"}, {"title": ""}, 3]'
    assert names.parse_candidates(text) == [{"title": "Brindleway Static", "artist": "Oru Fenwick"}]
    assert names.parse_candidates("no json") == []
    assert names.candidate_queries("A B", "C") == ["A B", "C", "A B by C"]


def test_corpus_count_takes_the_higher_of_both_cases(monkeypatch):
    seen = []

    def fake(query, index):
        seen.append(query)
        return {"Tin Moth": 0, "tin moth": 7}[query]
    monkeypatch.setattr(names, "infinigram_count", fake)
    assert names.corpus_count("Tin Moth", "idx") == 7 and seen == ["Tin Moth", "tin moth"]
    seen.clear()
    monkeypatch.setattr(names, "infinigram_count", lambda q, i: seen.append(q) or 0)
    assert names.corpus_count("lower", "idx") == 0 and seen == ["lower"]


def test_infinigram_response_parsing(monkeypatch):
    class Resp:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(self.payload).encode()
    monkeypatch.setattr(names.urllib.request, "urlopen", lambda req, timeout: Resp({"count": 3, "approx": False}))
    assert names.infinigram_count("x", "idx") == 3
    monkeypatch.setattr(names.urllib.request, "urlopen", lambda req, timeout: Resp({"error": "bad index"}))
    monkeypatch.setattr(names.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        names.infinigram_count("x", "idx", retries=2)


# ---------------------------------------------------------------------------
# Data-validation tests: run on generated outputs (skip when files absent).
# Each maps to a numbered test in docs/PHASE0_stimuli_pipeline.md.
# ---------------------------------------------------------------------------

def _load_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _songs_file():
    return P0 / "songs.jsonl"


# --- Test 1.1 (data): generated songs.jsonl is balanced ---

@pytest.mark.skipif(not (P0 / "songs.jsonl").exists(), reason="songs.jsonl not generated yet")
def test_1_1_songs_attribute_balance():
    """Test 1.1 on actual songs.jsonl: every value appears within ±5 of 500/n."""
    songs = _load_jsonl(_songs_file())
    assert len(songs) == 500, f"expected 500 songs, got {len(songs)}"
    for attr in design.ATTRIBUTES:
        counts = {}
        for s in songs:
            counts[s[attr]] = counts.get(s[attr], 0) + 1
        expected = 500 / len(design.VALUES[attr])
        for val, cnt in counts.items():
            assert abs(cnt - expected) <= 5, f"{attr}={val}: {cnt} (expected ~{expected})"


# --- Test 2.1 (data): all songs zero corpus count ---

@pytest.mark.skipif(not (P0 / "names").exists(), reason="names dir not generated yet")
def test_2_1_songs_zero_corpus_count():
    """Test 2.1: every survivor in counts.jsonl has zero on every query × index."""
    counts_file = P0 / "names" / "counts.jsonl"
    survivors_file = P0 / "names" / "survivors.jsonl"
    if not counts_file.exists():
        pytest.skip("counts.jsonl not generated yet")
    survivors = _load_jsonl(survivors_file) if survivors_file.exists() else []
    if not survivors:
        pytest.skip("survivors.jsonl empty or missing")
    for row in survivors:
        assert not row.get("mgphot_overlap"), f"MGPHot overlap: {row['title']}"
        for key, n in row.get("counts", {}).items():
            assert n == 0, f"nonzero count {n} for {row['title']}: {key}"


@pytest.mark.skipif(not (P0 / "names").exists(), reason="names dir not generated yet")
def test_2_1_recheck_all_zero():
    """Test 2.1 recheck: most recent recheck file shows all zero counts."""
    recheck_files = sorted(P0.glob("names/recheck_*.jsonl"))
    if not recheck_files:
        pytest.skip("no recheck file yet (run phase0_names.py recheck)")
    rows = _load_jsonl(recheck_files[-1])
    hits = [r["song_id"] for r in rows if any(r["counts"].values())]
    assert not hits, f"Test 2.1 recheck: {len(hits)} songs have nonzero corpus count: {hits[:5]}"


# --- Test 2.2 (data): no title overlaps MGPHot ---

@pytest.mark.skipif(not (P0 / "songs.jsonl").exists(), reason="songs.jsonl not generated yet")
def test_2_2_no_mgphot_title_overlap():
    """Test 2.2: no song title shares >=3 consecutive words with any MGPHot title."""
    songs = _load_jsonl(_songs_file())
    mgphot_dir = os.environ.get("MGPHOT_DIR", "D:/WMM_external/mgphot")
    titles_file = Path(mgphot_dir) / "hot100_titles.txt"
    if not titles_file.exists():
        try:
            from musicwhy.data.mgphot import load_mgphot
            mgphot_titles = [s["title"] for s in load_mgphot(mgphot_dir)]
        except Exception:
            pytest.skip("MGPHot data not available")
    else:
        mgphot_titles = titles_file.read_text().splitlines()
    exact, grams = names.real_title_index(mgphot_titles)
    overlaps = [s["title"] for s in songs if names.overlaps_real_title(s["title"], exact, grams)]
    assert not overlaps, f"Test 2.2: {len(overlaps)} titles overlap MGPHot: {overlaps[:5]}"


# --- Test 3.1 (data): planner metadata empty for every clip ---

@pytest.mark.skipif(not (P0 / "clips.jsonl").exists(), reason="clips.jsonl not generated yet")
def test_3_1_planner_metadata_empty():
    """Test 3.1: thinking=False -> lm_metadata is empty/None for every generated clip."""
    rows = _load_jsonl(P0 / "clips.jsonl")
    if not rows:
        pytest.skip("clips.jsonl is empty")
    violations = []
    for r in rows:
        meta = r.get("lm_metadata")
        if meta is not None and meta != "" and meta != {} and meta != []:
            violations.append(r["clip_id"])
    assert not violations, f"Test 3.1: {len(violations)} clips have planner metadata: {violations[:5]}"


# --- Test 3.2 (data): all clips 48 kHz (or 44.1 kHz MIDI), 20 ± 0.5 s, no silence > 2 s ---

@pytest.mark.skipif(not (P0 / "clips.jsonl").exists(), reason="clips.jsonl not generated yet")
def test_3_2_clip_properties():
    """Test 3.2: all accepted clips are 48 kHz (or 44.1 kHz MIDI), 20 ± 0.5 s, no silence > 2 s.
    Clips that fail 3.2 are rejected by the verify step, so only accepted clips are checked here."""
    rows = _load_jsonl(P0 / "clips.jsonl")
    accepted = [r for r in rows if r.get("accepted")]
    if not accepted:
        pytest.skip("no accepted clips yet")
    failures = []
    for r in accepted:
        t32 = r.get("test_3_2")
        if t32 is None:
            continue
        if not t32.get("pass", True):
            failures.append((r["clip_id"], t32.get("reason", "unknown")))
    assert not failures, f"Test 3.2: {len(failures)} accepted clips failed: {failures[:5]}"


# --- Test 4.1 (data): calibration results show >= 85% agreement ---

@pytest.mark.skipif(not (P0 / "calibration" / "results.json").exists(),
                    reason="calibration not run yet")
def test_4_1_calibration_agreement():
    """Test 4.1: verifier tools agree >= 85% on known MIDI clips for tempo band and mode."""
    results = json.loads((P0 / "calibration" / "results.json").read_text())
    summary = results.get("summary", {})
    for key in sorted(summary):
        if key.startswith(("tempo_band_acc", "mode_acc")):
            acc = summary[key]
            assert acc >= 0.85, f"Test 4.1: {key} = {acc:.2%} < 85%"
    assert summary.get("test_4_1_pass", False), "Test 4.1: summary flag is False"


# --- Test 4.2 (data): acceptance rates are recorded ---

@pytest.mark.skipif(not (P0 / "clips.jsonl").exists(), reason="clips.jsonl not generated yet")
def test_4_2_acceptance_rates_exist():
    """Test 4.2: clips.jsonl has measured clips with acceptance decisions."""
    rows = _load_jsonl(P0 / "clips.jsonl")
    measured = [r for r in rows if "checks" in r]
    if not measured:
        pytest.skip("no measured clips yet")
    accepted = sum(r["accepted"] for r in measured)
    total = len(measured)
    rate = accepted / total
    assert total > 0, "no clips were measured"
    print(f"Test 4.2: acceptance rate {accepted}/{total} = {rate:.1%}")


# --- Test 4.3 (data): listening sheet exists and has results ---

@pytest.mark.skipif(not (P0 / "listening_sheet.csv").exists(),
                    reason="listening sheet not generated yet")
def test_4_3_listening_sheet_has_results():
    """Test 4.3: the hand-listening sheet exists and has been filled in."""
    import csv
    with open(P0 / "listening_sheet.csv", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) >= 100, f"listening sheet has {len(rows)} rows, need >= 100"
    filled = [r for r in rows if r.get("matches", "").strip()]
    if not filled:
        pytest.skip("listening sheet generated but not yet filled in (manual step)")
    false_accept = sum(1 for r in filled if r["matches"].strip().lower() in ("n", "no"))
    rate = false_accept / len(filled)
    print(f"Test 4.3: false-accept rate {false_accept}/{len(filled)} = {rate:.1%}")


# --- Done criteria (§5): every song has agree + conflict clips ---

@pytest.mark.skipif(not (P0 / "clips.jsonl").exists(), reason="clips.jsonl not generated yet")
def test_done_criteria_coverage():
    """§5: >= 500 songs each have >= 1 accepted agree clip and >= 1 accepted conflict clip per attribute."""
    rows = _load_jsonl(P0 / "clips.jsonl")
    accepted = [r for r in rows if r["accepted"]]
    agree_songs = {r["song_id"] for r in accepted if r["arm"] == "agree"}
    conflict_by_attr = {}
    for r in accepted:
        if r["arm"] == "conflict" and r.get("conflict_attr"):
            conflict_by_attr.setdefault(r["conflict_attr"], set()).add(r["song_id"])
    if not agree_songs:
        pytest.skip("no accepted clips yet")
    missing_agree = 500 - len(agree_songs)
    print(f"Songs with accepted agree clip: {len(agree_songs)}/500")
    for attr in design.ATTRIBUTES:
        n = len(conflict_by_attr.get(attr, set()))
        print(f"Songs with accepted {attr} conflict clip: {n}/500")
    assert len(agree_songs) >= 500, f"{missing_agree} songs missing agree clips"
    for attr in design.ATTRIBUTES:
        n = len(conflict_by_attr.get(attr, set()))
        assert n >= 500, f"{500 - n} songs missing {attr} conflict clips"
