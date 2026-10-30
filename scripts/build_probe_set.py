"""Build the listening-vs-remembering instrument probe (musicwhy/benchmark/era.py) into one JSON file.

Song titles appear in the "name" questions, so this set is kept apart from the benchmark files.
  python -I scripts/build_probe_set.py
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from musicwhy.benchmark import era  # noqa: E402
from musicwhy.data.mgphot import load_mgphot  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mgphot-dir", default="D:/WMM_external/MGPHot-audio")
    p.add_argument("--clips-dir", default="D:/WMM_external/mgphot_2k/clips_16k")
    p.add_argument("--katana-clips", default="/srv/scratch/z5531827/musicwhy/data/raw/mgphot/clips")
    p.add_argument("--per-decade", type=int, default=50)
    p.add_argument("--out", default="D:/WMM_external/mgphot_2k/musicwhy_probe_test.json")
    args = p.parse_args()
    songs = load_mgphot(args.mgphot_dir)
    clips = {f.stem: f"{args.katana_clips}/{f.name}" for f in Path(args.clips_dir).glob("*.flac")}
    questions = era.instrument_probe_questions(songs, clips, per_decade=args.per_decade)
    names = {s["youtube_id"]: (s["title"], s["artist"]) for s in songs}
    # Appended last, so results for the questions above stay valid when a run resumes
    questions += era.wrong_name_controls(questions, names, "probe_instrument_name")
    questions = era.assign_ids(questions, "probe")
    Path(args.out).write_text(json.dumps(questions, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(questions)} questions -> {args.out}")
    print(dict(collections.Counter(q["question_type"] for q in questions)))
    print("right-answer instruments:", dict(collections.Counter(q["metadata"]["gene"] for q in questions[::3])))
    print(era.option_bias(questions[::3]))


if __name__ == "__main__":
    main()
