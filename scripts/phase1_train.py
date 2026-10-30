"""Phase 1 §4: LoRA fine-tuning of Qwen2-Audio-7B-Instruct.

  python -I scripts/phase1_train.py --train-file data/injection/train.jsonl --output-dir data/injection/checkpoints

Injects fictional song facts via QLoRA (4-bit) on the language model layers only.
Audio encoder and audio projector are frozen. Uses the Qwen2-Audio chat template.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen2-Audio-7B-Instruct")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=0, help="0 = compute from data size")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        Qwen2AudioForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_mem // 2**30} GB", flush=True)

    # --- Load model in 4-bit ---
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # Freeze audio encoder and audio projector — only LM layers train
    for name, param in model.named_parameters():
        if "audio_encoder" in name or "audio_tower" in name or "multi_modal_projector" in name:
            param.requires_grad = False

    # --- LoRA on LM language_model layers only ---
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=None,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)", flush=True)

    # --- Load training data ---
    raw = [json.loads(l) for l in args.train_file.read_text().splitlines() if l.strip()]
    print(f"Training examples: {len(raw)}", flush=True)

    def tokenize(example):
        msgs = example["messages"]
        text = processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        enc = processor.tokenizer(text, truncation=True, max_length=args.max_length, padding="max_length")
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds = Dataset.from_list(raw)
    ds = ds.map(tokenize, remove_columns=ds.column_names, num_proc=4)
    ds.set_format("torch")

    # --- Training ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps = args.max_steps if args.max_steps > 0 else None

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs if steps is None else 1,
        max_steps=steps or -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        bf16=True,
        logging_steps=50,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
    )

    if args.resume and (args.output_dir / "trainer_state.json").exists():
        print("Resuming from checkpoint...", flush=True)
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    model.save_pretrained(args.output_dir / "final")
    processor.tokenizer.save_pretrained(args.output_dir / "final")
    print(f"Saved final adapter to {args.output_dir / 'final'}", flush=True)


if __name__ == "__main__":
    main()
