"""Phase 0 §4: measure each generated clip with independent tools and accept or reject it.

Runs in the verify environment. Reads CLIPS_DIR/params.jsonl from phase0_generate.py and appends one row per
clip to $DATA_ROOT/phase0/clips.jsonl. Accepted clips are kept as mono 48 kHz FLAC in phase0/clips/ (the
audio models read mono); a small seeded sample of rejected clips goes to phase0/rejected_sample/ at 16 kHz
for diagnosis. Everything else in CLIPS_DIR is temporary. Clips already in clips.jsonl are skipped.

  python scripts/phase0_verify.py --clips-dir DIR
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from musicwhy.stimuli.acceptance import THRESHOLDS, accept  # noqa: E402
from musicwhy.stimuli.measure import Measurer, load_mono  # noqa: E402

DATA = Path(os.environ.get("DATA_ROOT", "data"))
OUT = DATA / "phase0"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips-dir", required=True)
    p.add_argument("--reject-sample", type=float, default=0.05, help="fraction of rejected clips to keep")
    args = p.parse_args()
    import shutil
    import tempfile
    import soundfile as sf

    clips_dir = Path(args.clips_dir)
    rows_file = OUT / "clips.jsonl"
    (OUT / "clips").mkdir(parents=True, exist_ok=True)
    (OUT / "rejected_sample").mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["clip_id"] for l in rows_file.read_text().splitlines() if l.strip()} \
        if rows_file.exists() else set()
    gens = [json.loads(l) for l in (clips_dir / "params.jsonl").read_text().splitlines() if l.strip()]
    todo = [g for g in gens if g["clip_id"] not in done]
    m = Measurer()
    print(f"{len(todo)} clips to verify; tempo tools: {m.tempo_a}, {m.tempo_b}", flush=True)

    local_dir = Path(tempfile.mkdtemp(prefix="p0_verify_"))
    local_clips = local_dir / "clips"
    local_reject = local_dir / "rejected_sample"
    local_clips.mkdir()
    local_reject.mkdir()
    pending_rows = []

    t_start = time.time()
    for i, g in enumerate(todo, 1):
        row = {k: g.get(k) for k in ("clip_id", "song_id", "arm", "conflict_attr", "seed", "target",
                                     "test_attr", "source", "spec", "params",
                                     "acestep_commit", "generated", "lm_metadata", "test_3_2")}
        if not g.get("generated"):
            row.update(accepted=False, reason="generation failed", error=g.get("error"))
        elif g.get("source") == "midi":
            ok = g["test_3_2"]["pass"]
            if not ok:
                row["reason"] = "test 3.2 (rate, duration or silence)"
            row.update(accepted=ok, checks={"midi_by_construction": True})
        else:
            wav = Path(g["path"])
            meas = m.measure_clip(wav)
            ok, checks = accept(g["target"], meas)
            if not g["test_3_2"]["pass"]:
                ok = False
                row["reason"] = "test 3.2 (rate, duration or silence)"
            row.update(measurements=meas, checks=checks, accepted=ok, thresholds=THRESHOLDS,
                       tempo_tools=[m.tempo_a, m.tempo_b])
        wav = Path(g["path"]) if g.get("generated") else None
        if row.get("accepted") and wav:
            local_dest = local_clips / f"{g['clip_id']}.flac"
            sr_out = 48000 if g.get("source") != "midi" else 44100
            sf.write(str(local_dest), load_mono(wav, sr_out), sr_out)
            row["path"] = str(OUT / "clips" / f"{g['clip_id']}.flac")
        elif wav and not row.get("accepted") and random.Random(g["seed"]).random() < args.reject_sample:
            local_dest = local_reject / f"{g['clip_id']}.flac"
            sf.write(str(local_dest), load_mono(wav, 16000), 16000)
            row["path"] = str(OUT / "rejected_sample" / f"{g['clip_id']}.flac")
        pending_rows.append(json.dumps(row))
        if i % 25 == 0 or i == len(todo):
            rate = (time.time() - t_start) / i
            print(f"  {i}/{len(todo)} verified, {rate:.1f}s per clip", flush=True)

    n_acc = len(list(local_clips.iterdir()))
    n_rej = len(list(local_reject.iterdir()))
    print(f"Copying {n_acc} accepted + {n_rej} rejected FLACs to scratch...", flush=True)
    for src in local_clips.iterdir():
        shutil.copy2(str(src), str(OUT / "clips" / src.name))
    for src in local_reject.iterdir():
        shutil.copy2(str(src), str(OUT / "rejected_sample" / src.name))
    with open(rows_file, "a") as f:
        f.write("\n".join(pending_rows) + "\n")
    shutil.rmtree(str(local_dir), ignore_errors=True)
    print(f"Done: {len(pending_rows)} rows written to clips.jsonl", flush=True)


if __name__ == "__main__":
    main()
