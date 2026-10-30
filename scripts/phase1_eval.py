"""Phase 1 §5: evaluate the LoRA-injected model on T/A/0 arms.

  python -I scripts/phase1_eval.py --adapter-dir data/injection/checkpoints/final

For each of 500 songs × 4 attributes × 3 arms = 6,000 questions, the model picks one
of 4 MC options (rotated). Arm A includes the Phase 0 conflict clip as audio input.

Outputs: data/injection/eval_results.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter-dir", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen2-Audio-7B-Instruct")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--batch-start", type=int, default=0, help="resume from question index")
    args = p.parse_args()

    DATA = Path(os.environ.get("DATA_ROOT", "data"))
    INJ_DIR = DATA / "injection"
    out_file = args.output or (INJ_DIR / "eval_results.json")

    import torch
    from peft import PeftModel
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    print(f"Loading base model: {args.model_id}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading adapter: {args.adapter_dir}", flush=True)
    model = PeftModel.from_pretrained(model, str(args.adapter_dir))
    model.eval()

    questions_file = INJ_DIR / "eval_questions.json"
    if not questions_file.exists():
        sys.exit(f"Eval questions not found: {questions_file}\nRun phase1_build_eval.py first.")
    questions = json.loads(questions_file.read_text())
    print(f"Loaded {len(questions)} eval questions", flush=True)

    # Resume
    results = {"predictions": [], "raw_responses": [], "questions": []}
    if out_file.exists() and args.batch_start > 0:
        results = json.loads(out_file.read_text())
        print(f"Resuming from {len(results['predictions'])} completed", flush=True)

    start_idx = len(results["predictions"])
    t0 = time.time()

    for i, q in enumerate(questions):
        if i < start_idx:
            continue

        options_text = "\n".join(f"{k}) {v}" for k, v in q["options"].items())
        prompt = q["prompt"] + f"\n\nChoose one:\n{options_text}\n\nAnswer with just the letter."

        has_audio = q["arm"] == "A" and q.get("audio_path")
        if has_audio:
            import librosa
            audio_array, _ = librosa.load(q["audio_path"], sr=processor.feature_extractor.sampling_rate)
            messages = [
                {"role": "system", "content": "You are a helpful music knowledge assistant."},
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": "placeholder"},
                    {"type": "text", "text": f"Listen to this audio clip. {prompt}"},
                ]},
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful music knowledge assistant."},
                {"role": "user", "content": prompt},
            ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if has_audio:
            inputs = processor(text=text, audios=[audio_array], return_tensors="pt", padding=True)
        else:
            inputs = processor.tokenizer(text, return_tensors="pt")

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False)

        response = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = response.strip()[0].upper() if response.strip() else "X"
        if pred not in "ABCD":
            pred = "X"

        results["predictions"].append(pred)
        results["raw_responses"].append(response)
        results["questions"].append({
            "song_id": q["song_id"], "attribute": q["attribute"],
            "arm": q["arm"], "dose": q["dose"],
            "correct_answer": q["correct_answer"],
            "correct_value": q["correct_value"],
            "wrong_value": q["wrong_value"],
        })

        if (i + 1) % 50 == 0 or i == len(questions) - 1:
            elapsed = time.time() - t0
            n_done = i + 1 - (start_idx)
            rate = elapsed / max(n_done, 1)
            correct = sum(1 for p, q2 in zip(results["predictions"], results["questions"])
                          if p == q2["correct_answer"])
            acc = correct / len(results["predictions"])
            print(f"  {i+1}/{len(questions)} ({rate:.1f}s/q, acc={acc:.3f})", flush=True)
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results['predictions'])} results to {out_file}", flush=True)


if __name__ == "__main__":
    main()
