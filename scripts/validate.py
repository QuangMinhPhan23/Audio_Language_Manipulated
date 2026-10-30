"""Check pipeline outputs for content problems, not just existence.

Usage: python scripts/validate.py <check> [<check> ...]
Checks: manifest world stems annotations benchmark eval eval_open qwen blind pairs phase0 phase0_calib all
("all" leaves out the phase0 checks)
Exit code 1 if any check FAILs. Stdlib only so it runs fast on the login node.
"""
import csv
import json
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path

DATA = Path(os.environ.get("DATA_ROOT", "/srv/scratch/z5531827/musicwhy/data"))
STEMS = ["bass", "drums", "vocals", "other"]
BAD_TITLES = set(STEMS) | {"mixture", ""}
LETTERS = {"A", "B", "C", "D"}
failed = False


def report(level, check, msg):
    global failed
    if level == "FAIL":
        failed = True
    print(f"{level} {check}: {msg}", flush=True)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def manifest_rows():
    with open(DATA / "manifest.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_manifest():
    rows = manifest_rows()
    ids = [r["track_id"] for r in rows]
    report("PASS" if len(ids) == len(set(ids)) else "FAIL", "manifest", f"{len(rows)} rows, {len(set(ids))} unique ids")
    bad = [r["title"] for r in rows if r.get("title", "").strip().lower() in BAD_TITLES]
    report("FAIL" if bad else "PASS", "manifest", f"{len(bad)} rows titled like a stem file {bad[:3]}")
    for col, bad_val in (("artist", "unknown"), ("genre", "unknown"), ("year", "0"), ("license", "unknown")):
        n = sum(1 for r in rows if str(r.get(col, "")).strip().lower() in (bad_val, ""))
        report("WARN" if n else "PASS", "manifest", f"{n}/{len(rows)} rows with {col}={bad_val!r}")
    sources = Counter(r["source"] for r in rows)
    report("PASS", "manifest", f"sources {dict(sources)}; genres {dict(Counter(r['genre'] for r in rows).most_common(8))}")
    # Full datasets: MUSDB18-HQ has 150 songs; MoisesDB 240, one an exact duplicate build_manifest drops
    for source, expected in (("musdb18hq", 150), ("moisesdb", 239)):
        if source in sources:
            report("PASS" if sources[source] == expected else "WARN", "manifest", f"{source}: {sources[source]}/{expected} songs")
    missing = sum(1 for r in rows if not Path(r["file_path"]).exists())
    report("FAIL" if missing else "PASS", "manifest", f"{missing} rows whose audio file is missing")


def check_world():
    db = DATA / "world_context" / "world_context.db"
    if not db.exists():
        report("FAIL", "world", f"{db} missing")
        return
    conn = sqlite3.connect(str(db))
    q = lambda sql, *a: conn.execute(sql, a).fetchall()
    for table, minimum in (("economic", 1000), ("chart_entries", 100000), ("events", 5000), ("covid", 1000)):
        n = q(f"SELECT COUNT(*) FROM {table}")[0][0]
        report("FAIL" if n < minimum else "PASS", "world", f"{table}: {n} rows (expect >= {minimum})")
    bad_codes = [c for (c,) in q("SELECT DISTINCT country FROM economic UNION SELECT DISTINCT country FROM chart_entries")
                 if len(c) != 3]
    report("FAIL" if bad_codes else "PASS", "world", f"non-ISO3 country codes in economic/charts: {bad_codes[:5]}")
    # The old collector stored each year page's lead sentence ("2007 (MMVII) was a common year ...")
    boiler = q("SELECT COUNT(*) FROM events WHERE description LIKE '%was a common year%' "
               "OR description LIKE '%was a leap year%' OR description LIKE 'This is a list of%'")[0][0]
    report("FAIL" if boiler else "PASS", "world", f"{boiler} boilerplate event rows")
    by_type = dict(q("SELECT event_type, COUNT(DISTINCT year) FROM events GROUP BY event_type"))
    report("PASS" if by_type.get("world", 0) >= 70 and by_type.get("music", 0) >= 60 else "FAIL", "world",
           f"years covered per event type {by_type} (1950-2025 = 76 years)")
    chart_years = q("SELECT MIN(year), MAX(year), COUNT(DISTINCT year) FROM chart_entries")[0]
    report("PASS" if chart_years[2] >= 60 else "FAIL", "world", f"chart years {chart_years[0]}-{chart_years[1]} ({chart_years[2]} distinct)")
    # What an annotation for a 2007 US song would see (same query shape as world_context_db)
    econ = q("SELECT COUNT(*) FROM economic WHERE year = 2007 AND country = 'USA'")[0][0]
    charts = q("SELECT COUNT(*) FROM chart_entries WHERE year = 2007 AND country = 'USA'")[0][0]
    events = q("SELECT COUNT(*) FROM events WHERE year = 2007 AND country IN ('USA', 'GLOBAL')")[0][0]
    report("FAIL" if not (econ and charts and events) else "PASS", "world",
           f"2007/USA sample: {econ} economic, {charts} chart rows, {events} events")
    sample = q("SELECT description FROM events WHERE year = 2007 AND event_type = 'world' ORDER BY rowid LIMIT 1 OFFSET 3")
    report("PASS", "world", f"sample event: {sample[0][0][:150] if sample else None!r}")


def check_stems():
    rows = manifest_rows()
    files = list((DATA / "features" / "stems").glob("*.json"))
    expected = len(rows) * len(STEMS)
    report("PASS" if len(files) >= expected else "FAIL", "stems", f"{len(files)}/{expected} stem feature files")
    need = {"tempo", "spectral_centroid", "harmonic_percussive_ratio", "dynamic_range", "rms_mean"}
    missing_keys, zero_rms = 0, 0
    for p in files:
        d = load_json(p)
        missing_keys += bool(need - d.keys())
        zero_rms += float(d.get("rms_mean", 0) or 0) <= 1e-8
    report("FAIL" if missing_keys else "PASS", "stems", f"{missing_keys} files missing keys from {sorted(need)}")
    report("WARN" if zero_rms > len(files) * 0.1 else "PASS", "stems", f"{zero_rms} files with rms_mean ~ 0")


def check_annotations():
    feat_ids = {r["track_id"] for r in manifest_rows() if (DATA / "features" / f"{r['track_id']}.h5").exists()}
    files = sorted((DATA / "annotations").glob("*.json"))
    if not files:
        report("FAIL", "annotations", "no annotation files")
        return
    report("PASS" if len(files) >= len(feat_ids) else "WARN", "annotations", f"{len(files)}/{len(feat_ids)} tracks with features annotated")
    need = {"track_id", "reasoning", "feature_description", "stem_descriptions", "metadata"}
    counts = Counter()
    for p in files:
        a = load_json(p)
        counts["missing_keys"] += bool(need - a.keys())
        counts["short_reasoning"] += len(a.get("reasoning", "")) < 300
        counts["not_end_turn"] += a.get("stop_reason") != "end_turn"
        stem_desc = a.get("stem_descriptions", "")
        counts["no_stem_desc"] += not stem_desc
        n_stems = stem_desc.count("Analysis of the")
        if n_stems and stem_desc.count("(0% of full RMS)") >= n_stems:
            counts["all_stems_zero_energy"] += 1
        meta = a.get("metadata", {})
        counts["bad_title"] += str(meta.get("title", "")).strip().lower() in BAD_TITLES
        counts["year_0"] += not meta.get("year")
        counts["full_feature_format"] += "key center" in a.get("feature_description", "")
    n = len(files)
    report("FAIL" if counts["missing_keys"] else "PASS", "annotations", f"{counts['missing_keys']}/{n} missing required keys")
    report("FAIL" if counts["short_reasoning"] else "PASS", "annotations", f"{counts['short_reasoning']}/{n} with reasoning < 300 chars")
    report("FAIL" if counts["not_end_turn"] else "PASS", "annotations", f"{counts['not_end_turn']}/{n} without stop_reason=end_turn (truncated or unrecorded)")
    report("FAIL" if counts["no_stem_desc"] > n * 0.1 else "PASS", "annotations", f"{counts['no_stem_desc']}/{n} with empty stem_descriptions")
    report("FAIL" if counts["all_stems_zero_energy"] else "PASS", "annotations", f"{counts['all_stems_zero_energy']}/{n} where every stem reports 0% energy")
    report("FAIL" if counts["bad_title"] else "PASS", "annotations", f"{counts['bad_title']}/{n} with a stem name as title")
    report("WARN" if counts["year_0"] else "PASS", "annotations", f"{counts['year_0']}/{n} with year 0")
    fmt = counts["full_feature_format"]
    report("FAIL" if 0 < fmt < n else "PASS", "annotations", f"{fmt}/{n} use the full feature-description format (mixed formats = FAIL)")


def benchmark_questions():
    return load_json(DATA / "benchmark" / "musicwhy_test.json")["questions"]


def check_benchmark():
    bench = load_json(DATA / "benchmark" / "musicwhy_benchmark.json")
    qs = bench["questions"]
    report("PASS" if qs else "FAIL", "benchmark", f"{len(qs)} questions, task_counts={bench.get('task_counts')}")
    ids = Counter(q.get("question_id") for q in qs)
    dup_ids = sum(c - 1 for c in ids.values() if c > 1)
    report("FAIL" if dup_ids else "PASS", "benchmark", f"{dup_ids} duplicate question_ids (repeats leak across splits)")
    # Audio-modality questions share wording across songs; the clip makes them distinct
    texts = Counter((q.get("question_text", ""), json.dumps(q.get("audio_reference"), sort_keys=True)) for q in qs)
    dupes = sum(c - 1 for c in texts.values() if c > 1)
    report("WARN" if dupes > len(qs) * 0.05 else "PASS", "benchmark", f"{dupes} duplicate questions (same text and audio)")
    no_audio = sum(1 for q in qs if q.get("modality") == "audio" and not q.get("audio_reference"))
    report("FAIL" if no_audio else "PASS", "benchmark", f"{no_audio} audio-modality questions without audio_reference")
    modality = Counter((q.get("task_id"), q.get("modality", "?")) for q in qs)
    report("PASS", "benchmark", "by task/modality " + ", ".join(f"t{t}/{m}={n}" for (t, m), n in sorted(modality.items(), key=str)))
    t2 = Counter(q.get("metadata", {}).get("correct_stem") for q in qs if q.get("task_id") == 2)
    if t2:
        top = max(t2.values()) / sum(t2.values())
        report("WARN" if top > 0.5 else "PASS", "benchmark", f"task 2 correct stems {dict(t2)} (one stem > 50% = answerable by prior)")
    by_task = {}
    for q in qs:
        by_task.setdefault(q.get("task_id"), []).append(q)
    for task, tqs in sorted(by_task.items(), key=lambda kv: str(kv[0])):
        mc = [q for q in tqs if q.get("options")]
        if not mc:
            report("PASS", "benchmark", f"task {task}: {len(tqs)} open-ended")
            continue
        bad_keys = sum(1 for q in mc if set(q["options"]) != LETTERS or q.get("correct_answer") not in LETTERS)
        report("FAIL" if bad_keys else "PASS", "benchmark", f"task {task}: {bad_keys}/{len(mc)} MC with option keys != A-D or answer not a letter")
        dist = Counter(q.get("correct_answer") for q in mc)
        top = max(dist.values()) / len(mc)
        report("WARN" if top > 0.4 else "PASS", "benchmark", f"task {task}: answer letters {dict(sorted(dist.items()))}")
        longest = 0
        for q in mc:
            lens = {k: len(str(v)) for k, v in q["options"].items()}
            ans = q.get("correct_answer")
            if ans in lens and lens[ans] == max(lens.values()) and list(lens.values()).count(lens[ans]) == 1:
                longest += 1
        ratio = longest / len(mc)
        report("WARN" if ratio > 0.4 else "PASS", "benchmark", f"task {task}: correct option is uniquely longest in {ratio:.0%} (chance ~25%; high = style leak)")


def check_mc_results(name, check):
    path = DATA / "results" / name
    if not path.exists():
        report("FAIL", check, f"{path.name} missing")
        return
    res = load_json(path)
    preds, gt = res.get("predictions", []), res.get("ground_truth", [])
    all_qs = {q["question_id"]: q for q in benchmark_questions()}
    if res.get("question_ids"):
        missing = [i for i in res["question_ids"] if i not in all_qs]
        report("FAIL" if missing else "PASS", check, f"{len(missing)} result question_ids not in the test split")
        mc = [all_qs.get(i, {}) for i in res["question_ids"]]
        wrong_gt = sum(1 for q, g in zip(mc, gt) if q and q.get("correct_answer") != g)
        report("FAIL" if wrong_gt else "PASS", check, f"{wrong_gt} stored ground-truth letters disagree with the benchmark")
    else:  # older results: aligned by position with every MC question
        mc = [q for q in all_qs.values() if q.get("options")]
    report("PASS" if len(preds) == len(gt) == len(mc) else "WARN", check, f"{len(preds)} predictions, {len(gt)} ground truth, {len(mc)} questions")
    invalid = sum(1 for p in preds if p not in LETTERS)
    report("FAIL" if invalid > len(preds) * 0.05 else "PASS", check, f"{invalid}/{len(preds)} predictions not parseable as A-D")
    by_task = {}
    for q, p, g in zip(mc, preds, gt):
        c = by_task.setdefault(f"{q.get('task_id')}/{q.get('modality', 'text')}", [0, 0])
        c[0] += p == g
        c[1] += 1
    for task, (c, n) in sorted(by_task.items(), key=lambda kv: str(kv[0])):
        acc = c / n
        level = "WARN" if acc in (0.0, 1.0) else "PASS"
        report(level, check, f"task {task}: {c}/{n} = {acc:.1%}" + (" (0% or 100% usually means a format bug or a leak)" if level == "WARN" else ""))


def check_eval_open():
    path = DATA / "results" / "eval_claude_text_open.json"
    if not path.exists():
        report("FAIL", "eval_open", f"{path.name} missing")
        return
    res = load_json(path)
    rows = res.get("per_question", [])
    # Any "error" key counts, even an empty message (StopIteration stringifies to "")
    errors = sum(1 for r in rows if "error" in r)
    report("FAIL" if errors > len(rows) * 0.05 else "PASS", "eval_open", f"{errors}/{len(rows)} questions errored")
    empty = sum(1 for r in rows if "error" not in r and not r.get("model_answer", "").strip())
    report("FAIL" if empty else "PASS", "eval_open", f"{empty} graded rows with an empty model answer")
    out_of_range = sum(1 for r in rows if "error" not in r and any(not 1 <= v <= 5 for v in r.get("scores", {}).values()))
    report("FAIL" if out_of_range else "PASS", "eval_open", f"{out_of_range} graded rows with a score outside 1-5")
    report("PASS", "eval_open", "averages " + ", ".join(f"{k}={res.get(k, 0):.2f}" for k in ("avg_factual_accuracy", "avg_depth_of_reasoning", "avg_specificity")))
    judges = Counter(r.get("judge_model", "unrecorded") for r in rows if "error" not in r)
    report("WARN" if res.get("model") in judges or "unrecorded" in judges else "PASS", "eval_open",
           f"judge models {dict(judges)} (answering model: {res.get('model')}; a model grading itself = WARN)")
    by_task = {}
    for r in rows:
        if "error" not in r:
            by_task.setdefault(r.get("task_id"), []).append(r["scores"].get("factual_accuracy", 0))
    report("PASS", "eval_open", "factual accuracy by task " + ", ".join(
        f"t{t}: {sum(v) / len(v):.2f} (n={len(v)})" for t, v in sorted(by_task.items(), key=str)))


def check_pairs():
    path = DATA / "embeddings" / "similar_pairs.json"
    if not path.exists():
        report("FAIL", "pairs", f"{path} missing")
        return
    pairs = load_json(path)
    report("PASS" if pairs else "FAIL", "pairs", f"{len(pairs)} pairs")
    self_pairs = sum(1 for p in pairs if p["song_a"]["song_id"] == p["song_b"]["song_id"])
    keys = Counter(frozenset((p["song_a"]["song_id"], p["song_b"]["song_id"])) for p in pairs)
    dupes = sum(c - 1 for c in keys.values() if c > 1)
    bad_score = sum(1 for p in pairs if not -1.0001 <= float(p.get("similarity_score", 9)) <= 1.0001)
    no_feats = sum(1 for p in pairs if not p["song_a"].get("features") or not p["song_b"].get("features"))
    for label, n in (("self-pairs", self_pairs), ("duplicate pairs", dupes), ("scores outside [-1,1]", bad_score), ("pairs missing features", no_feats)):
        report("FAIL" if n else "PASS", "pairs", f"{n} {label}")
    # Pairs vs their own distractors: a useful similarity puts the pair well above unrelated songs
    pair_scores = [float(p["similarity_score"]) for p in pairs]
    dist_scores = [float(d["similarity_score"]) for p in pairs for d in p.get("distractors", [])]
    if not dist_scores:
        report("FAIL", "pairs", "no distractors stored (rebuild with the current extract_embeddings.pbs)")
    else:
        gap = sorted(pair_scores)[len(pair_scores) // 2] - sorted(dist_scores)[len(dist_scores) // 2]
        report("FAIL" if gap < 0.2 else "PASS", "pairs",
               f"median pair similarity {sorted(pair_scores)[len(pair_scores) // 2]:.3f} vs median distractor "
               f"{sorted(dist_scores)[len(dist_scores) // 2]:.3f} (gap {gap:.3f}; < 0.2 means scores don't separate songs)")
        bad_d = sum(1 for p in pairs for d in p.get("distractors", [])
                    if d["song_id"] in (p["song_a"]["song_id"], p["song_b"]["song_id"]))
        report("FAIL" if bad_d else "PASS", "pairs", f"{bad_d} distractors equal to a pair member")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_phase0():
    """Phase 0 stimuli (PHASE0_stimuli_pipeline.md): Tests 1.1, 2.1, 2.2, 3.1, 3.2, 4.2 on whatever exists."""
    p0 = DATA / "phase0"
    songs_file = p0 / "songs.jsonl"
    if songs_file.exists():
        songs = read_jsonl(songs_file)
        report("PASS" if len(songs) == 500 else "FAIL", "phase0", f"{len(songs)} songs (want 500)")
        for attr in ("lead_instrument", "tempo_band", "mode", "vocals"):
            counts = Counter(s[attr] for s in songs)
            spread = max(abs(c - len(songs) / len(counts)) for c in counts.values())
            report("PASS" if spread <= 5 else "FAIL", "phase0", f"Test 1.1 {attr} {dict(counts)}")
        for col in ("title", "artist"):
            dupes = len(songs) - len({s[col].lower() for s in songs})
            report("FAIL" if dupes else "PASS", "phase0", f"{dupes} duplicate {col}s")
        rechecks = sorted((p0 / "names").glob("recheck_*.jsonl"))
        if rechecks:
            rows = read_jsonl(rechecks[-1])
            hits = [r["song_id"] for r in rows if any(r["counts"].values())]
            report("FAIL" if hits else "PASS", "phase0", f"Test 2.1 ({rechecks[-1].name}): {len(hits)} songs with "
                   f"corpus hits {hits[:5]}")
        else:
            report("WARN", "phase0", "Test 2.1 recheck not run yet (run STEP=recheck on a later day)")
    else:
        report("WARN", "phase0", "songs.jsonl not built yet")
    survivors = p0 / "names" / "survivors.jsonl"
    if survivors.exists():
        rows = read_jsonl(survivors)
        bad = sum(r["mgphot_overlap"] or any(r["counts"].values()) for r in rows)
        report("FAIL" if bad else "PASS", "phase0", f"{len(rows)} name survivors, {bad} with hits or MGPHot overlap "
               "(Tests 2.1, 2.2)")
    clips_file = p0 / "clips.jsonl"
    if not clips_file.exists():
        report("WARN", "phase0", "clips.jsonl not built yet")
        return
    rows = read_jsonl(clips_file)
    ids = Counter(r["clip_id"] for r in rows)
    report("FAIL" if len(ids) < len(rows) else "PASS", "phase0", f"{len(rows)} clips, {len(rows) - len(ids)} duplicate ids")
    planner = [r["clip_id"] for r in rows if r.get("lm_metadata")]
    report("FAIL" if planner else "PASS", "phase0", f"Test 3.1: {len(planner)} clips with planner metadata {planner[:3]}")
    t32 = [r for r in rows if r.get("test_3_2")]
    bad32 = sum(not r["test_3_2"]["pass"] for r in t32)
    report("WARN" if bad32 else "PASS", "phase0", f"Test 3.2: {bad32}/{len(t32)} clips off rate, length or with silence > 2 s")
    accepted = [r for r in rows if r["accepted"]]
    missing = sum(not Path(r["path"]).exists() for r in accepted[:200])
    report("FAIL" if missing else "PASS", "phase0", f"{missing} of the first {min(len(accepted), 200)} accepted clips missing on disk")
    ace_measured = [r for r in rows if "checks" in r and not r["checks"].get("midi_by_construction")]
    midi_measured = [r for r in rows if "checks" in r and r["checks"].get("midi_by_construction")]
    for attr in ("tempo_band", "mode", "lead_instrument", "vocals"):
        if ace_measured:
            k = sum(r["checks"][attr]["pass"] for r in ace_measured)
            report("PASS", "phase0", f"Test 4.2 {attr}: {k}/{len(ace_measured)} pass (ACE-Step)")
    if midi_measured:
        k = sum(r["accepted"] for r in midi_measured)
        report("PASS", "phase0", f"MIDI clips: {k}/{len(midi_measured)} accepted (by construction)")
    share = len(accepted) / len(rows) if rows else 0
    # Test 4.2: below 30% generate more seeds; below 15% change the attribute set
    level = "FAIL" if share < 0.15 and len(rows) >= 100 else "WARN" if share < 0.30 else "PASS"
    report(level, "phase0", f"Test 4.2 overall acceptance {len(accepted)}/{len(rows)} = {100 * share:.0f}%")


def check_phase0_calib():
    path = DATA / "phase0" / "calibration" / "results.json"
    summary = load_json(path)["summary"]
    for k, v in summary.items():
        if k.startswith(("tempo_band_acc", "mode_acc")):
            report("PASS" if v >= 0.85 else "FAIL", "phase0_calib", f"Test 4.1 {k} = {v:.2f} (need 0.85)")
        elif k.startswith(("instrument_top1", "rule_pass", "vocals_acc", "vocal_ratio")):
            report("PASS", "phase0_calib", f"{k} = {v}")


CHECKS = {
    "manifest": check_manifest,
    "world": check_world,
    "stems": check_stems,
    "annotations": check_annotations,
    "benchmark": check_benchmark,
    "eval": lambda: check_mc_results("eval_claude_text.json", "eval"),
    "eval_open": check_eval_open,
    "qwen": lambda: check_mc_results("eval_qwen2audio.json", "qwen"),
    # Audio questions answered with no audio: accuracy well above 25% means the wording leaks the answer
    "blind": lambda: check_mc_results("eval_claude_blind.json", "blind"),
    "pairs": check_pairs,
    "phase0": check_phase0,
    "phase0_calib": check_phase0_calib,
}


def main():
    names = sys.argv[1:] or ["all"]
    if names == ["all"]:
        names = [n for n in CHECKS if not n.startswith("phase0")]  # Phase 0 checks run by name only
    for name in names:
        if name not in CHECKS:
            sys.exit(f"unknown check {name!r}; choose from {', '.join(CHECKS)} or all")
        try:
            CHECKS[name]()
        except FileNotFoundError as e:
            report("FAIL", name, f"missing input: {e.filename}")
        except (KeyError, ValueError, TypeError) as e:
            report("FAIL", name, f"unexpected structure: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
