"""Build the MGPHot era questions for Task 5 (musicwhy/benchmark/era.py) into one JSON file.

Runs locally, where the MGPHot files are kept (their licence forbids redistribution, so they stay out
of the repo). Audio questions point at the 16 kHz clips copied to Katana.

  python -I scripts/build_era_benchmark.py --checklists data_local/era_checklists.json
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
    p.add_argument("--checklists", default="", help="JSON of sourced 'why' checklists (optional)")
    p.add_argument("--per-decade", type=int, default=50)
    p.add_argument("--out", default="D:/WMM_external/mgphot_2k/musicwhy_era_test.json")
    args = p.parse_args()

    songs = load_mgphot(args.mgphot_dir)
    clips = {f.stem: f"{args.katana_clips}/{f.name}" for f in Path(args.clips_dir).glob("*.flac")}
    questions = era.change_questions(songs) + era.peak_questions(songs)
    questions += era.era_from_sound_questions(songs, clips, per_decade=args.per_decade)
    if args.checklists:
        questions += era.why_questions(json.loads(Path(args.checklists).read_text(encoding="utf-8")))
    era.assign_ids(questions)
    ids = [q["question_id"] for q in questions]
    assert len(ids) == len(set(ids)), "duplicate question ids"

    Path(args.out).write_text(json.dumps(questions, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(questions)} questions -> {args.out}")
    print(dict(collections.Counter((q["question_type"], q["modality"]) for q in questions)))
    for qtype in ("era_change", "era_peak", "era_from_sound"):
        print(qtype, era.option_bias([q for q in questions if q["question_type"] == qtype and q["modality"] == "text"]))


if __name__ == "__main__":
    main()
