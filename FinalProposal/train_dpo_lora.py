"""
Optional DPO-style LoRA finetuning on automatically built preferences.
Designed to be lightweight (4-bit + LoRA) and optional if local GPU is small.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DPO-style LoRA adapter on preference pairs.")
    parser.add_argument("--prefs", type=Path, required=True, help="Preference JSONL from build_preferences.py")
    parser.add_argument("--model-name", type=str, required=True, help="Base HF model to finetune.")
    parser.add_argument("--output-dir", type=Path, default=Path("lora"), help="Where to save the adapter.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (temperature).")
    parser.add_argument("--max-length", type=int, default=1024, help="Truncation length for prompts.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit quantization to save VRAM.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of preference pairs for quick tests.")
    return parser.parse_args()


def load_pairs(path: Path, limit: int | None) -> List[dict]:
    pairs = []
    with path.open() as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if not line.strip():
                continue
            pairs.append(json.loads(line))
    return pairs


def main() -> None:
    args = parse_args()
    try:
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import DPOTrainer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "Missing dependencies for DPO training. Install `trl peft datasets transformers` (and bitsandbytes for 4-bit)."
        ) from exc

    pairs = load_pairs(args.prefs, args.limit)
    if not pairs:
        raise SystemExit("No preference pairs found.")

    dataset = Dataset.from_dict(
        {
            "prompt": [p["prompt"] for p in pairs],
            "chosen": [p["chosen"] for p in pairs],
            "rejected": [p["rejected"] for p in pairs],
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_4bit=args.load_in_4bit,
        device_map="auto" if args.load_in_4bit else None,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=max(args.max_steps // 4, 50),
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # uses frozen reference under the hood
        args=training_args,
        beta=args.beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
