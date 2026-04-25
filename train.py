#!/usr/bin/env python3
"""
train.py — Fine-tune a base model into the Vehicle SLM using QLoRA via Unsloth.

Takes the training pairs from training_data.py and fine-tunes a small model
(Phi-3-mini 3.8B or Qwen 2.5 3B) to generate CAN command frames from
vehicle context + traffic + intent.

Hardware: runs on RTX 2070 (8GB VRAM) with QLoRA/4-bit quantization.
"""

import json
import yaml
import sys
from pathlib import Path


DEFAULT_CONFIG = {
    "base_model": "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
    "alternatives": [
        "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
        "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",  # for very constrained VRAM
    ],
    "training": {
        "max_seq_length": 2048,
        "lora_rank": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,  # effective batch = 8
        "warmup_steps": 10,
        "weight_decay": 0.01,
        "max_grad_norm": 0.3,
        "seed": 42,
    },
    "data": {
        "train_path": "data/train.jsonl",
        "eval_path": "data/eval.jsonl",
        "eval_split": 0,  # 0 because we use family-holdout split from prepare_training_split.py
    },
    "output_dir": "models/vehicle-slm-v0",
    "quantize_on_save": True,
}

SYSTEM_PROMPT = """You are a Vehicle CAN Bus Command Generator. Given a vehicle's metadata, \
observed CAN bus traffic, and an operator intent, you generate the exact CAN frame(s) \
needed to execute that command on the vehicle.

You respond ONLY with the <response> block containing the CAN frame(s) and a brief explanation. \
CAN IDs are in hex (0xNNN), data bytes are space-separated hex (AA BB CC DD EE FF 00 00).

You are a fluent speaker of vehicle CAN protocols. You do not look up tables — you generate \
frames directly from your understanding of how this manufacturer encodes control signals."""


def format_for_training(example: dict) -> dict:
    """Format a training pair into the chat template the model expects."""
    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
    }


def load_dataset(path: Path, eval_split: float = 0.1):
    """Load training pairs and optionally split into train/eval."""
    from datasets import Dataset

    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                raw = json.loads(line)
                examples.append(format_for_training(raw))

    ds = Dataset.from_list(examples)
    if eval_split > 0:
        split = ds.train_test_split(test_size=eval_split, seed=42)
        return split["train"], split["test"]
    return ds, None


def load_dataset_family_split(path: Path, holdout_families: list[str]):
    """Split train/eval by manufacturer family (proper eval — no data leakage)."""
    from datasets import Dataset

    train_examples = []
    eval_examples = []

    with open(path) as f:
        for line in f:
            if line.strip():
                raw = json.loads(line)
                mfr = raw.get("metadata", {}).get("manufacturer", "unknown")
                formatted = format_for_training(raw)
                if mfr in holdout_families:
                    eval_examples.append(formatted)
                else:
                    train_examples.append(formatted)

    print(f"[+] Train: {len(train_examples)}, Eval (holdout families): {len(eval_examples)}")
    return Dataset.from_list(train_examples), Dataset.from_list(eval_examples)


def train(config: dict | None = None):
    """Run the QLoRA fine-tuning pipeline."""
    cfg = config or DEFAULT_CONFIG
    tcfg = cfg["training"]

    print("=" * 60)
    print("VEHICLE SLM — QLoRA FINE-TUNING")
    print("=" * 60)
    print(f"  Base model:  {cfg['base_model']}")
    print(f"  LoRA rank:   {tcfg['lora_rank']}")
    print(f"  LR:          {tcfg['learning_rate']}")
    print(f"  Epochs:      {tcfg['num_epochs']}")
    print(f"  Batch:       {tcfg['batch_size']} × {tcfg['gradient_accumulation_steps']} "
          f"= {tcfg['batch_size'] * tcfg['gradient_accumulation_steps']}")
    print()

    # ── Load model via Unsloth ──
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=tcfg["max_seq_length"],
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    # ── Apply LoRA adapters ──
    model = FastLanguageModel.get_peft_model(
        model,
        r=tcfg["lora_rank"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=tcfg["lora_alpha"],
        lora_dropout=tcfg["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # ── Load data ──
    data_path = Path(cfg["data"]["train_path"])
    if not data_path.exists():
        print(f"[!] Training data not found at {data_path}")
        print("    Run: python training_data.py data/opendbc")
        sys.exit(1)

    train_ds, eval_ds = load_dataset(data_path, cfg["data"]["eval_split"])
    print(f"[+] Loaded {len(train_ds)} training examples")
    if eval_ds:
        print(f"[+] Loaded {len(eval_ds)} eval examples")

    # ── Tokenize ──
    def tokenize(example):
        text = tokenizer.apply_chat_template(
            example["conversations"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return tokenizer(
            text,
            truncation=True,
            max_length=tcfg["max_seq_length"],
            padding=False,
        )

    train_ds = train_ds.map(tokenize, remove_columns=["conversations"])
    if eval_ds:
        eval_ds = eval_ds.map(tokenize, remove_columns=["conversations"])

    # ── Train ──
    from trl import SFTTrainer
    from transformers import TrainingArguments

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=tcfg["num_epochs"],
        per_device_train_batch_size=tcfg["batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["max_grad_norm"],
        warmup_steps=tcfg["warmup_steps"],
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_ds else "no",
        seed=tcfg["seed"],
        fp16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        dataset_text_field=None,
        max_seq_length=tcfg["max_seq_length"],
    )

    print("\n[*] Starting training...")
    trainer.train()

    # ── Save ──
    print(f"\n[+] Saving LoRA adapter to {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Optionally merge and quantize for deployment
    if cfg.get("quantize_on_save"):
        merged_dir = output_dir / "merged-4bit"
        print(f"[+] Saving quantized merged model to {merged_dir}")
        model.save_pretrained_merged(
            str(merged_dir),
            tokenizer,
            save_method="merged_4bit_forced",
        )

    print("\n[✓] Training complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the Vehicle SLM")
    parser.add_argument("--config", type=Path, help="YAML config file")
    parser.add_argument("--model", type=str, help="Override base model")
    parser.add_argument("--epochs", type=int, help="Override epoch count")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()

    if args.config and args.config.exists():
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))

    if args.model:
        cfg["base_model"] = args.model
    if args.epochs:
        cfg["training"]["num_epochs"] = args.epochs
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr

    train(cfg)
