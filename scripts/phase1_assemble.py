"""Phase 1 §3: assemble training data (injection + replay).

  python -I scripts/phase1_assemble.py [--replay-file PATH]

Reads paraphrases and dose groups from Phase 1 §1-2, builds the injection portion,
mixes with replay data at 95:5 ratio, and writes:
  data/injection/train.jsonl

Replay source: Alpaca-cleaned (Stanford, CC-BY-NC-4.0) or any JSONL with
{"instruction", "input", "output"} rows. Download:
  wget https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from musicwhy.injection.dose import (  # noqa: E402
    assemble_training_data,
    build_injection_data,
)

DATA = Path(os.environ.get("DATA_ROOT", "data"))
INJ_DIR = DATA / "injection"


def load_replay(path: Path) -> list:
    """Load replay data from Alpaca-format JSON or instruction JSONL."""
    raw = json.loads(path.read_text()) if path.suffix == ".json" else [
        json.loads(l) for l in path.read_text().splitlines() if l.strip()
    ]
    out = []
    for r in raw:
        q = r.get("instruction", r.get("question", ""))
        inp = r.get("input", "")
        if inp:
            q = f"{q}\n{inp}"
        a = r.get("output", r.get("answer", ""))
        if q and a:
            out.append({"type": "replay", "question": q, "answer": a})
    return out


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replay-file", type=Path,
                    default=DATA / "injection" / "alpaca_data_cleaned.json")
    p.add_argument("--injection-frac", type=float, default=0.05)
    args = p.parse_args()

    paras = [json.loads(l) for l in (INJ_DIR / "paraphrases.jsonl").read_text().splitlines() if l.strip()]
    dose_map = json.loads((INJ_DIR / "dose_groups.json").read_text())
    print(f"Loaded {len(paras)} paraphrases, {len(dose_map)} dose assignments", flush=True)

    injection = build_injection_data(paras, dose_map)
    print(f"Injection examples: {len(injection)}", flush=True)
    by_dose = Counter(r["dose"] for r in injection)
    for d, n in sorted(by_dose.items()):
        print(f"  dose {d}: {n}", flush=True)

    if not args.replay_file.exists():
        sys.exit(f"Replay file not found: {args.replay_file}\n"
                 f"Download: wget -O {args.replay_file} "
                 f"https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json")
    replay = load_replay(args.replay_file)
    print(f"Replay examples: {len(replay)}", flush=True)

    train = assemble_training_data(injection, replay, args.injection_frac)
    out_file = INJ_DIR / "train.jsonl"
    with open(out_file, "w") as f:
        for row in train:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(train)} training examples to {out_file}", flush=True)

    # Test 3.1: injection fraction
    n_inj = sum(1 for r in train if r["metadata"]["type"] == "injection")
    frac = n_inj / len(train)
    if not (args.injection_frac - 0.005 <= frac <= args.injection_frac + 0.005):
        sys.exit(f"Test 3.1 FAILED: injection fraction {frac:.4f} (expected ~{args.injection_frac})")
    print(f"Test 3.1 passed: injection fraction {frac:.4f}", flush=True)

    # Test 3.2: no dose-0 songs in injection
    dose0_songs = {sid for sid, d in dose_map.items() if d == 0}
    leaked = {r["metadata"]["song_id"] for r in train
              if r["metadata"]["type"] == "injection" and r["metadata"]["song_id"] in dose0_songs}
    if leaked:
        sys.exit(f"Test 3.2 FAILED: dose-0 songs in injection data: {leaked}")
    print("Test 3.2 passed: no dose-0 songs in injection data", flush=True)


if __name__ == "__main__":
    main()
