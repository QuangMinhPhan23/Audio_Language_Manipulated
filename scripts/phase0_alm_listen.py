"""Ask an audio-language model to identify the lead instrument in each listening-pack clip.

Runs in the main venv on a GPU node (Qwen2-Audio-7B needs ~17 GB VRAM in bf16).
Outputs data/phase0/listening_pack/alm_results.json.

  python scripts/phase0_alm_listen.py [--clips-dir DIR] [--key PATH]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INSTRUMENTS = ["piano", "electric guitar", "saxophone", "synthesizer", "violin", "trumpet"]
PROMPT = (
    "Listen to this music clip carefully. Which ONE instrument is playing the lead melody? "
    "Choose exactly one from this list: piano, electric guitar, saxophone, synthesizer, violin, trumpet. "
    "If none of them clearly leads, say 'unclear'. Reply with just the instrument name, nothing else."
)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips-dir", default="data/phase0/listening_pack")
    p.add_argument("--key", default="data/phase0/listening_pack/key.json")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    clips_dir = Path(args.clips_dir)
    out_path = Path(args.out) if args.out else clips_dir / "alm_results.json"
    key = json.loads(Path(args.key).read_text()) if Path(args.key).exists() else {}

    flacs = sorted(clips_dir.glob("pack_*.flac"))
    if not flacs:
        print(f"No pack_*.flac in {clips_dir}", flush=True)
        sys.exit(1)

    from musicwhy.evaluation.models import Qwen2AudioWrapper
    model = Qwen2AudioWrapper(max_new_tokens=32)
    print(f"Loaded Qwen2-Audio. {len(flacs)} clips to evaluate.", flush=True)

    results = {}
    for i, clip in enumerate(flacs, 1):
        name = clip.stem
        messages = [{"role": "user", "content": [
            {"type": "audio", "audio_url": str(clip)},
            {"type": "text", "text": PROMPT},
        ]}]
        raw = model._generate(messages, max_new_tokens=32).strip()
        # Normalise answer
        answer = raw.lower().strip().rstrip(".")
        matched = next((inst for inst in INSTRUMENTS if inst in answer), answer)
        requested = key.get(name, {}).get("requested_lead")
        match = matched == requested if requested else None
        results[name] = {"raw": raw, "answer": matched, "matches_key": match}
        tag = "OK" if match else ("MISMATCH" if match is False else "?")
        print(f"  {i}/{len(flacs)} {name}: heard={matched!r}  {tag}", flush=True)

    n_match = sum(1 for r in results.values() if r["matches_key"] is True)
    n_total = sum(1 for r in results.values() if r["matches_key"] is not None)
    print(f"\nQwen2-Audio matches key: {n_match}/{n_total}", flush=True)

    out_path.write_text(json.dumps(results, indent=1))
    print(f"Results: {out_path}", flush=True)


if __name__ == "__main__":
    main()
