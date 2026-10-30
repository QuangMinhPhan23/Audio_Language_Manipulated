"""Build the instrument probe on unknown MoisesDB songs (musicwhy/benchmark/era.py). Runs on Katana.

  cd /srv/scratch/z5531827/musicwhy && .venv/bin/python -I scripts/build_unknown_probe.py
Writes data/benchmark/musicwhy_probe_unknown.json.
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from musicwhy.benchmark import era  # noqa: E402
from musicwhy.data.light_io import load_manifest  # noqa: E402


def main():
    data = ROOT / "data"
    tracks = []
    for tid, row in load_manifest(data).items():
        if row["source"] != "moisesdb":
            continue
        meta_path = Path(row["file_path"]).parent / "meta.json"
        meta = json.loads(meta_path.read_text())
        tracks.append({"track_id": tid, "title": row["title"], "artist": row["artist"],
                       "sources": meta.get("sources", {}), "audio_path": row["file_path"]})
    questions = era.unknown_song_probe_questions(tracks)
    names = {t["track_id"]: (t["title"], t["artist"]) for t in tracks}
    questions += era.wrong_name_controls(questions, names, "probe_unknown_name")  # appended last
    questions = era.assign_ids(questions, "probe_unknown")
    out = data / "benchmark" / "musicwhy_probe_unknown.json"
    out.write_text(json.dumps(questions, indent=1, ensure_ascii=False))
    print(f"{len(tracks)} MoisesDB tracks -> {len(questions)} questions -> {out}")
    print(dict(collections.Counter(q["question_type"] for q in questions)))


if __name__ == "__main__":
    main()
