"""Phase 0 v2 — Steps 4-5: attach fictional songs and generate facts.

Usage:
  python -I scripts/phase0v2_assign.py assign   # Attach songs to windows + assign doses
  python -I scripts/phase0v2_assign.py facts     # Generate agree/conflict facts

Reads: data/phase0/measured_clips.jsonl, data/phase0/songs.jsonl, data/phase0/stem_counts.json
Writes: data/phase0/v2_songs.jsonl, data/phase0/facts.jsonl
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DATA = Path(os.environ.get("DATA_ROOT", "data"))
P0 = DATA / "phase0"
DOSES = [0, 1, 10, 100, 1000]


def _load_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _save_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows)} rows -> {path}", flush=True)


def step_assign():
    """Step 4: attach one fictional (title, artist) per clip window."""
    from musicwhy.stimuli.design import stable_seed

    measured_path = P0 / "measured_clips.jsonl"
    if not measured_path.exists():
        sys.exit("Run phase0v2_stems.py measure first")
    clips = _load_jsonl(measured_path)

    songs_path = P0 / "songs.jsonl"
    if not songs_path.exists():
        sys.exit("Run phase0_names.py first to generate songs.jsonl with fictional names")
    names = _load_jsonl(songs_path)

    counts_path = P0 / "stem_counts.json"
    if not counts_path.exists():
        sys.exit("Run phase0v2_stems.py count first")
    counts = json.loads(counts_path.read_text())
    kept_instruments = list(counts["kept"])

    usable = [c for c in clips if not c.get("skipped")]
    if len(usable) > len(names):
        print(f"Warning: {len(usable)} clips but only {len(names)} fictional names; using first {len(names)}", flush=True)
        usable = usable[:len(names)]
    elif len(names) > len(usable):
        print(f"Note: {len(names)} names available, {len(usable)} clips; {len(names) - len(usable)} names unused", flush=True)

    v2_songs = []
    for i, clip in enumerate(usable):
        name = names[i]
        song = {
            "song_id": f"v2_{i:04d}",
            "title": name["title"],
            "artist": name["artist"],
            "lead_instrument": clip["instrument"],
            "vocals": "vocal",
            "tempo_band": clip.get("tempo_band"),
            "mode": clip.get("mode"),
            "clip_id": clip["clip_id"],
            "uuid": clip["uuid"],
            "stem_name": clip["stem_name"],
            "window_idx": clip["window_idx"],
            "start_s": clip["start_s"],
            "end_s": clip["end_s"],
            "vocal_path": clip.get("vocal_path"),
            "instrumental_path": clip.get("instrumental_path"),
        }
        v2_songs.append(song)

    _save_jsonl(v2_songs, P0 / "v2_songs.jsonl")

    from musicwhy.stimuli.facts import assign_doses
    dose_map = assign_doses(v2_songs, DOSES)
    for s in v2_songs:
        s["dose"] = dose_map[s["song_id"]]
    _save_jsonl(v2_songs, P0 / "v2_songs.jsonl")

    n_tempo = sum(1 for s in v2_songs if s["tempo_band"] is not None)
    n_mode = sum(1 for s in v2_songs if s["mode"] is not None)
    by_inst = {}
    for s in v2_songs:
        by_inst[s["lead_instrument"]] = by_inst.get(s["lead_instrument"], 0) + 1
    by_dose = {}
    for s in v2_songs:
        by_dose[s["dose"]] = by_dose.get(s["dose"], 0) + 1

    print(f"\n{len(v2_songs)} songs assigned", flush=True)
    print(f"  tempo known: {n_tempo}/{len(v2_songs)} ({100*n_tempo/len(v2_songs):.0f}%)", flush=True)
    print(f"  mode known: {n_mode}/{len(v2_songs)} ({100*n_mode/len(v2_songs):.0f}%)", flush=True)
    print(f"  by instrument: {by_inst}", flush=True)
    print(f"  by dose: {dict(sorted(by_dose.items()))}", flush=True)


def step_facts():
    """Step 5: generate agree/conflict facts for each song."""
    from musicwhy.stimuli.facts import assign_facts, facts_summary

    v2_path = P0 / "v2_songs.jsonl"
    if not v2_path.exists():
        sys.exit("Run 'assign' first")
    songs = _load_jsonl(v2_path)

    counts_path = P0 / "stem_counts.json"
    counts = json.loads(counts_path.read_text())
    instrument_values = list(counts["kept"])

    facts = assign_facts(songs, seed=0, instrument_values=instrument_values)
    _save_jsonl(facts, P0 / "facts.jsonl")

    summary = facts_summary(facts)
    print(f"\n{summary['n_facts']} facts for {summary['n_songs']} songs", flush=True)
    print(f"  per attribute: {summary['per_attr']}", flush=True)
    print(f"  songs with unequal agree/conflict: {summary['songs_with_unequal_split']}", flush=True)
    if summary["songs_with_unequal_split"] > 0:
        print(f"    (expected ~0 for songs with all 4 attributes known, some for songs with 3)", flush=True)

    (P0 / "facts_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    cmds = {"assign": step_assign, "facts": step_facts}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        print(f"Commands: {', '.join(cmds)}")
        sys.exit(1)
    cmds[sys.argv[1]]()
