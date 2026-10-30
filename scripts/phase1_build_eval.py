"""Phase 1 §5 prep: build eval questions from songs, dose groups, and Phase 0 clips.

  python -I scripts/phase1_build_eval.py

Reads Phase 0 outputs and Phase 1 dose groups, resolves conflict clip paths,
and writes data/injection/eval_questions.json (6,000 questions: 500 songs × 4 attrs × 3 arms).
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from musicwhy.injection.eval_arms import build_eval_questions  # noqa: E402
from musicwhy.stimuli.design import assign_attributes  # noqa: E402

DATA = Path(os.environ.get("DATA_ROOT", "data"))
INJ_DIR = DATA / "injection"


def main():
    songs = assign_attributes()
    songs_file = DATA / "phase0" / "songs.jsonl"
    if songs_file.exists():
        named = {json.loads(l)["song_id"]: json.loads(l)
                 for l in songs_file.read_text().splitlines() if l.strip()}
        for s in songs:
            if s["song_id"] in named:
                s["title"] = named[s["song_id"]]["title"]
                s["artist"] = named[s["song_id"]]["artist"]

    dose_map = json.loads((INJ_DIR / "dose_groups.json").read_text())

    # Build conflict clip paths from Phase 0 clips.jsonl
    clips_file = DATA / "phase0" / "clips.jsonl"
    conflict_clips: dict = defaultdict(dict)
    if clips_file.exists():
        for line in clips_file.read_text().splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            if c.get("accepted") and c.get("arm") == "conflict" and c.get("test_attr"):
                sid = c["song_id"]
                attr = c["test_attr"]
                if attr not in conflict_clips[sid]:
                    conflict_clips[sid][attr] = c.get("path", "")

    questions = build_eval_questions(songs, dose_map, dict(conflict_clips))
    out_file = INJ_DIR / "eval_questions.json"
    with open(out_file, "w") as f:
        json.dump(questions, f, indent=2)

    n_with_audio = sum(1 for q in questions if q.get("audio_path"))
    print(f"Wrote {len(questions)} eval questions to {out_file}", flush=True)
    print(f"  Arm 0: {sum(1 for q in questions if q['arm'] == '0')}", flush=True)
    print(f"  Arm T: {sum(1 for q in questions if q['arm'] == 'T')}", flush=True)
    print(f"  Arm A: {sum(1 for q in questions if q['arm'] == 'A')} ({n_with_audio} with audio)", flush=True)


if __name__ == "__main__":
    main()
