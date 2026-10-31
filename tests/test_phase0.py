import json
import os
from pathlib import Path

import numpy as np
import pytest

from musicwhy.stimuli import design, names
from musicwhy.stimuli.acceptance import (THRESHOLDS, accept, instrument_votes,
                                        majority_instrument, tempo_match)
from musicwhy.stimuli.measure import bpm_from_beats, clip_properties
from musicwhy.stimuli.facts import assign_facts, facts_summary, assign_doses

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


# ===========================================================================
# Phase 0 v2 — unit tests (run anywhere, no data needed)
# ===========================================================================

def test_v2_constants():
    assert len(design.INSTRUMENTS_V2) == 6
    assert set(design.VALUES_V2["vocals"]) == {"vocal", "instrumental"}
    assert set(design.VALUES_V2["mode"]) == {"major", "minor"}
    assert set(design.VALUES_V2["tempo_band"]) == {"slow", "medium", "fast"}


def test_facts_balance_4_attrs():
    """Each song with all 4 known attributes gets exactly 2 agree + 2 conflict."""
    instruments = ["piano", "electric guitar", "acoustic guitar", "synthesizer"]
    songs = [{"song_id": f"t{i}", "lead_instrument": instruments[i % 4],
              "vocals": "vocal", "tempo_band": "slow", "mode": "major"} for i in range(100)]
    facts = assign_facts(songs, seed=42, instrument_values=instruments)
    by_song = {}
    for f in facts:
        by_song.setdefault(f["song_id"], {"agree": 0, "conflict": 0})
        by_song[f["song_id"]][f["arm"]] += 1
    for sid, counts in by_song.items():
        assert counts["agree"] == 2 and counts["conflict"] == 2, f"{sid}: {counts}"


def test_facts_balance_per_attribute():
    """Each attribute is roughly 50/50 agree/conflict across all songs."""
    instruments = ["piano", "electric guitar", "acoustic guitar"]
    songs = [{"song_id": f"t{i}", "lead_instrument": instruments[i % 3],
              "vocals": ["vocal", "instrumental"][i % 2],
              "tempo_band": ["slow", "medium", "fast"][i % 3],
              "mode": ["major", "minor"][i % 2]} for i in range(200)]
    facts = assign_facts(songs, seed=0, instrument_values=instruments)
    summary = facts_summary(facts)
    for attr, counts in summary["per_attr"].items():
        total = counts["agree"] + counts["conflict"]
        ratio = counts["agree"] / total
        assert 0.35 < ratio < 0.65, f"{attr}: agree ratio {ratio:.2f} not near 0.5"


def test_facts_conflict_value_differs():
    """Conflict facts always inject a different value than the true one."""
    instruments = ["piano", "electric guitar", "synthesizer"]
    songs = [{"song_id": f"t{i}", "lead_instrument": instruments[i % 3],
              "vocals": "vocal", "tempo_band": "medium", "mode": "major"} for i in range(50)]
    facts = assign_facts(songs, seed=0, instrument_values=instruments)
    for f in facts:
        if f["arm"] == "conflict":
            assert f["injected_value"] != f["true_value"], f


def test_facts_skips_none_attributes():
    """Songs with None tempo/mode get fewer than 4 facts."""
    songs = [{"song_id": "t0", "lead_instrument": "piano",
              "vocals": "vocal", "tempo_band": None, "mode": None}]
    facts = assign_facts(songs, seed=0, instrument_values=["piano", "guitar"])
    assert len(facts) == 2  # only instrument + vocals


def test_dose_balance():
    """Doses are balanced across instruments."""
    instruments = ["piano", "guitar", "synth"]
    songs = [{"song_id": f"t{i}", "lead_instrument": instruments[i % 3]}
             for i in range(75)]
    mapping = assign_doses(songs, doses=[0, 10, 100], seed=0)
    assert len(mapping) == 75
    by_dose = {}
    for d in mapping.values():
        by_dose[d] = by_dose.get(d, 0) + 1
    assert all(20 <= n <= 30 for n in by_dose.values()), by_dose


def test_eval_arms_no_fake_options():
    """Mode and vocals use only their real values (2-way), not padded fakes."""
    from musicwhy.injection.eval_arms import _mc_options
    opts_mode, _, n_mode = _mc_options("mode", "major", seed=42)
    assert n_mode == 2
    assert all(v in ("major", "minor") for v in opts_mode.values())
    opts_voc, _, n_voc = _mc_options("vocals", "vocal", seed=42)
    assert n_voc == 2


def test_eval_arms_instrument_uses_all_values():
    """Instrument questions use all real values."""
    from musicwhy.injection.eval_arms import _mc_options
    opts, _, n = _mc_options("lead_instrument", "piano", seed=42)
    assert n == len(design.INSTRUMENTS)


def test_eval_arms_evidence_order_varies():
    """The T arm randomises evidence-before vs evidence-after the question."""
    from musicwhy.injection.eval_arms import build_eval_questions
    songs = [{"song_id": f"t{i}", "title": f"Song {i}", "artist": f"Artist {i}",
              "lead_instrument": "piano", "tempo_band": "slow", "mode": "major",
              "vocals": "vocal"} for i in range(100)]
    dose_map = {s["song_id"]: 0 for s in songs}
    qs = build_eval_questions(songs, dose_map)
    t_qs = [q for q in qs if q["arm"] == "T"]
    starts_with_evidence = sum(1 for q in t_qs if q["prompt"].startswith("I recently"))
    assert 100 < starts_with_evidence < len(t_qs) - 100, \
        f"evidence order not randomised: {starts_with_evidence}/{len(t_qs)} start with evidence"


def test_stems_classify():
    """Stem classification maps MoisesDB names to our categories."""
    from musicwhy.stimuli.stems import classify_stem
    assert classify_stem({"stemName": "Piano"}) == "piano"
    assert classify_stem({"stemName": "Bowed Strings"}) == "strings"
    assert classify_stem({"stemName": "Vocals"}) is None
    assert classify_stem({"stemName": "Bass"}) is None
    assert classify_stem({"stemName": "Drums"}) is None
    assert classify_stem({"stemName": "Guitar"}) == "electric guitar"
    assert classify_stem({"stemName": "Guitar",
                          "tracks": [{"trackType": "Acoustic Guitar"}]}) == "acoustic guitar"
    assert classify_stem({"stemName": "Other Keys",
                          "tracks": [{"trackType": "Organ"}]}) == "organ"
    assert classify_stem({"stemName": "Other Keys"}) == "synthesizer"


# ===========================================================================
# Phase 0 v2 — data-validation tests (skip when outputs absent)
# ===========================================================================

@pytest.mark.skipif(not (P0 / "stem_counts.json").exists(), reason="stem counts not generated yet")
def test_v2_1_stem_counts():
    """Test v2.1: every kept instrument value has >= 25 songs."""
    counts = json.loads((P0 / "stem_counts.json").read_text())
    for inst, n in counts["kept"].items():
        assert n >= 25, f"v2.1: {inst} has {n} songs, need >= 25"


@pytest.mark.skipif(not (P0 / "rendered_clips.jsonl").exists(), reason="clips not rendered yet")
def test_v2_2_clip_activity():
    """Test v2.2: rendered clips meet activity and loudness thresholds."""
    clips = _load_jsonl(P0 / "rendered_clips.jsonl")
    rendered = [c for c in clips if not c.get("skipped")]
    if not rendered:
        pytest.skip("no rendered clips yet")
    for c in rendered[:20]:
        vocal_path = Path(c.get("vocal_path", ""))
        inst_path = Path(c.get("instrumental_path", ""))
        for p in (vocal_path, inst_path):
            if p.exists():
                import soundfile as sf
                audio, sr = sf.read(str(p), dtype="float32")
                assert len(audio) / sr >= 19.5, f"{p.name}: too short ({len(audio)/sr:.1f}s)"


@pytest.mark.skipif(not (P0 / "facts.jsonl").exists(), reason="facts not generated yet")
def test_v2_3_facts_balance():
    """Test v2.3: 2 agree + 2 conflict per song (where all 4 attrs known); 50/50 per attribute."""
    facts = _load_jsonl(P0 / "facts.jsonl")
    if not facts:
        pytest.skip("facts.jsonl is empty")
    summary = facts_summary(facts)
    for attr, counts in summary["per_attr"].items():
        total = counts["agree"] + counts["conflict"]
        if total == 0:
            continue
        ratio = counts["agree"] / total
        assert 0.40 < ratio < 0.60, f"v2.3: {attr} agree ratio {ratio:.2f} not near 0.5"

    v2_songs_path = P0 / "v2_songs.jsonl"
    if v2_songs_path.exists():
        songs = _load_jsonl(v2_songs_path)
        counts_path = P0 / "stem_counts.json"
        if counts_path.exists():
            counts = json.loads(counts_path.read_text())
            dose_map = {}
            for s in songs:
                if "dose" in s:
                    dose_map.setdefault(s["dose"], {}).setdefault(
                        s["lead_instrument"], 0)
                    dose_map[s["dose"]][s["lead_instrument"]] += 1


@pytest.mark.skipif(not (P0 / "phase0_report.md").exists(), reason="report not generated yet")
def test_v2_4_hearing_gate():
    """Test v2.4: hearing gate numbers recorded in phase0_report.md."""
    text = (P0 / "phase0_report.md").read_text()
    assert "instrument" in text.lower() and "vocal" in text.lower(), \
        "v2.4: phase0_report.md must contain hearing gate results"
