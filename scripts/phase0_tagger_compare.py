"""Measure the listening-pack clips with all instrument taggers and compute majority votes.

Runs in the verify environment. Outputs data/phase0/listening_pack/tagger_results.json with per-clip
per-tagger probabilities, top-1 votes, majority vote and whether it matches the key. Does NOT print
the answer key — the user's blind listening answers come first.

  python scripts/phase0_tagger_compare.py [--clips-dir DIR] [--key PATH]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from musicwhy.stimuli.acceptance import instrument_votes, majority_instrument
    from musicwhy.stimuli.measure import Measurer

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips-dir", default="data/phase0/listening_pack", help="directory with pack_NN.flac clips")
    p.add_argument("--key", default="data/phase0/listening_pack/key.json", help="answer key (for match column)")
    p.add_argument("--out", default=None, help="output JSON (default: <clips-dir>/tagger_results.json)")
    args = p.parse_args()

    clips_dir = Path(args.clips_dir)
    out_path = Path(args.out) if args.out else clips_dir / "tagger_results.json"
    key = json.loads(Path(args.key).read_text()) if Path(args.key).exists() else {}

    flacs = sorted(clips_dir.glob("pack_*.flac"))
    if not flacs:
        print(f"No pack_*.flac files in {clips_dir}", flush=True)
        sys.exit(1)

    m = Measurer()
    results = {}
    for i, clip in enumerate(flacs, 1):
        name = clip.stem
        inst = {"jamendo": m.instruments_jamendo(clip), "passt": m.instruments_passt(clip)}
        try:
            inst["clap"] = m.instruments_clap(clip)
        except Exception as e:
            print(f"  CLAP failed on {name}: {e}", flush=True)
        votes = instrument_votes(inst)
        majority = majority_instrument(inst)
        requested = key.get(name, {}).get("requested_lead")
        results[name] = {"taggers": inst, "votes": votes, "majority": majority,
                         "matches_key": majority == requested if requested else None}
        print(f"  {i}/{len(flacs)} {name}: votes={votes}  majority={majority}"
              f"  {'OK' if majority == requested else 'MISMATCH' if requested else '?'}", flush=True)

    # Summary
    n_match = sum(1 for r in results.values() if r["matches_key"] is True)
    n_total = sum(1 for r in results.values() if r["matches_key"] is not None)
    print(f"\nMajority vote matches key: {n_match}/{n_total}", flush=True)
    for tagger in ("jamendo", "passt", "clap"):
        n = sum(1 for name, r in results.items()
                if r["votes"].get(tagger) == key.get(name, {}).get("requested_lead"))
        print(f"  {tagger} top-1 matches key: {n}/{n_total}", flush=True)

    out_path.write_text(json.dumps(results, indent=1))
    print(f"\nResults: {out_path}", flush=True)


if __name__ == "__main__":
    main()
