"""Phase 1 §1-2: generate paraphrases and assign dose groups.

  python -I scripts/phase1_paraphrase.py [--songs-file PATH]

Reads songs from Phase 0's songs.jsonl, generates 80 QA paraphrases per song (4 attrs × 20),
assigns 500 songs to 5 dose groups (100 each), and writes:
  data/injection/paraphrases.jsonl
  data/injection/dose_groups.json
  data/injection/summary.txt
"""
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from musicwhy.injection.paraphrase import N_TEMPLATES, all_paraphrases  # noqa: E402
from musicwhy.injection.dose import DOSES, SONGS_PER_DOSE, assign_dose_groups  # noqa: E402
from musicwhy.stimuli.design import ATTRIBUTES, VALUES  # noqa: E402

DATA = Path(os.environ.get("DATA_ROOT", "data"))


def main():
    songs_file = DATA / "phase0" / "songs.jsonl"
    if not songs_file.exists():
        sys.exit(f"Songs file not found: {songs_file}")
    songs = [json.loads(l) for l in songs_file.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(songs)} songs", flush=True)

    out_dir = DATA / "injection"
    out_dir.mkdir(parents=True, exist_ok=True)

    paras = all_paraphrases(songs)
    with open(out_dir / "paraphrases.jsonl", "w") as f:
        for p in paras:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(paras)} paraphrases ({len(paras) // len(songs)} per song)", flush=True)

    # Test 1.1: every song has 80 QA pairs
    by_song = Counter(p["song_id"] for p in paras)
    expected = len(ATTRIBUTES) * N_TEMPLATES
    bad = {s: c for s, c in by_song.items() if c != expected}
    if bad:
        sys.exit(f"Test 1.1 FAILED: songs with wrong paraphrase count: {bad}")
    print(f"Test 1.1 passed: all {len(songs)} songs have {expected} paraphrases", flush=True)

    dose_map = assign_dose_groups(songs)
    with open(out_dir / "dose_groups.json", "w") as f:
        json.dump(dose_map, f, indent=2)

    # Test 2.1: check balance
    dose_counts = Counter(dose_map.values())
    for d in DOSES:
        if dose_counts[d] != SONGS_PER_DOSE:
            sys.exit(f"Test 2.1 FAILED: dose {d} has {dose_counts[d]} songs, expected {SONGS_PER_DOSE}")

    by_id = {s["song_id"]: s for s in songs}
    for d in DOSES:
        group_songs = [by_id[sid] for sid, dose in dose_map.items() if dose == d]
        for attr in ATTRIBUTES:
            vals = Counter(s[attr] for s in group_songs)
            pop_frac = {v: len([s for s in songs if s[attr] == v]) / len(songs) for v in VALUES[attr]}
            for v in VALUES[attr]:
                expected_count = pop_frac[v] * SONGS_PER_DOSE
                if abs(vals.get(v, 0) - expected_count) > 3:
                    sys.exit(f"Test 2.1 FAILED: dose {d} attr {attr} val {v}: "
                             f"{vals.get(v, 0)} (expected ~{expected_count:.0f})")
    print(f"Test 2.1 passed: all dose groups balanced", flush=True)

    lines = [f"Phase 1 paraphrase + dose summary",
             f"Songs: {len(songs)}",
             f"Paraphrases: {len(paras)} ({expected} per song)",
             f"Dose groups: {dict(sorted(dose_counts.items()))}",
             ""]
    for d in DOSES:
        group = [by_id[sid] for sid, dose in dose_map.items() if dose == d]
        n_inject = sum(dose_map[s["song_id"]] for s in group) * len(ATTRIBUTES)
        lines.append(f"Dose {d:>4}: {len(group)} songs, {n_inject:>7} injection examples")
    summary = "\n".join(lines)
    (out_dir / "summary.txt").write_text(summary + "\n")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
