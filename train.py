#!/usr/bin/env python3
"""
train.py — Fine-tune a base model into the Vehicle SLM using QLoRA.

Uses PEFT + TRL directly (no Unsloth dependency).
Takes training pairs from training_data.py and fine-tunes a small model
to generate CAN command frames from vehicle context + traffic + intent.

Hardware: runs on RTX 2070 (8GB VRAM) with QLoRA/4-bit quantization.
"""

import json
import yaml
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


DEFAULT_CONFIG = {
    "base_model": "microsoft/Phi-3-mini-4k-instruct",
    "alternatives": [
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",  # for very constrained VRAM
    ],
    "training": {
        "max_seq_length": 2048,
        "lora_rank": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
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
        "eval_split": 0,
    },
    "output_dir": "models/vehicle-slm-v0",
}

SYSTEM_PROMPT = """You are a Vehicle CAN Bus Command Generator. Given a vehicle's metadata, \
observed CAN bus traffic, and an operator intent, you generate the exact CAN frame(s) \
needed to execute that command on the vehicle.

You respond ONLY with the <response> block containing the CAN frame(s) and a brief explanation. \
CAN IDs are in hex (0xNNN), data bytes are space-separated hex (AA BB CC DD EE FF 00 00).

You are a fluent speaker of vehicle CAN protocols. You do not look up tables — you generate \
frames directly from your understanding of how this manufacturer encodes control signals."""


def format_as_text(example: dict) -> str:
    """Format a training pair as a single text string."""
    instruction = example["instruction"]
    response = example["response"]
    return (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{instruction}<|end|>\n"
        f"<|assistant|>\n{response}<|end|>\n"
    )


def load_data(path: Path, eval_path: Path = None, eval_split: float = 0.1):
    """Load training pairs and optionally split into train/eval."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                raw = json.loads(line)
                examples.append({"text": format_as_text(raw)})

    train_ds = Dataset.from_list(examples)

    # Load separate eval file if it exists
    if eval_path and Path(eval_path).exists():
        eval_examples = []
        with open(eval_path) as f:
            for line in f:
                if line.strip():
                    raw = json.loads(line)
                    eval_examples.append({"text": format_as_text(raw)})
        eval_ds = Dataset.from_list(eval_examples)
        return train_ds, eval_ds

    # Otherwise split
    if eval_split > 0:
        split = train_ds.train_test_split(test_size=eval_split, seed=42)
        return split["train"], split["test"]

    return train_ds, None


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
    print(f"  Batch:       {tcfg['batch_size']} x {tcfg['gradient_accumulation_steps']} "
          f"= {tcfg['batch_size'] * tcfg['gradient_accumulation_steps']}")
    print(f"  CUDA:        {torch.cuda.is_available()} ({torch.cuda.get_device_name(0)})")
    print()

    # ── Quantization config ──
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Load model ──
    print("[*] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"],
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # ── Prepare for QLoRA ──
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=tcfg["lora_rank"],
        lora_alpha=tcfg["lora_alpha"],
        lora_dropout=tcfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"[+] Trainable parameters: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)")

    # ── Load data ──
    data_path = Path(cfg["data"]["train_path"])
    eval_path = Path(cfg["data"]["eval_path"])
    if not data_path.exists():
        print(f"[!] Training data not found at {data_path}")
        print("    Run: python training_data.py data/opendbc")
        print("    Then: python prepare_training_split.py")
        sys.exit(1)

    train_ds, eval_ds = load_data(
        data_path,
        eval_path if eval_path.exists() else None,
        cfg["data"]["eval_split"],
    )
    print(f"[+] Loaded {len(train_ds)} training examples")
    if eval_ds:
        print(f"[+] Loaded {len(eval_ds)} eval examples")

    # ── Training arguments ──
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
        eval_strategy="epoch" if eval_ds else "no",
        seed=tcfg["seed"],
        fp16=True,
        bf16=False,
        report_to="none",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    )

    # ── Trainer ──
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=tcfg["max_seq_length"],
        packing=False,
    )

    print(f"\n[*] Starting training...")
    print(f"    Steps per epoch: ~{len(train_ds) // (tcfg['batch_size'] * tcfg['gradient_accumulation_steps'])}")
    print(f"    Total steps: ~{3 * len(train_ds) // (tcfg['batch_size'] * tcfg['gradient_accumulation_steps'])}")
    print()

    trainer.train()

    # ── Save ──
    print(f"\n[+] Saving LoRA adapter to {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("\n[+] Training complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the Vehicle SLM")
    parser.add_argument("--config", type=Path, help="YAML config file")
    parser.add_argument("--model", type=str, help="Override base model")
    parser.add_argument("--epochs", type=int, help="Override epoch count")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg["training"] = DEFAULT_CONFIG["training"].copy()
    cfg["data"] = DEFAULT_CONFIG["data"].copy()

    if args.config and args.config.exists():
        with open(args.config) as f:
            file_cfg = yaml.safe_load(f)
            if "training" in file_cfg:
                cfg["training"].update(file_cfg["training"])
            if "data" in file_cfg:
                cfg["data"].update(file_cfg["data"])
            if "base_model" in file_cfg:
                cfg["base_model"] = file_cfg["base_model"]
            if "output_dir" in file_cfg:
                cfg["output_dir"] = file_cfg["output_dir"]

    if args.model:
        cfg["base_model"] = args.model
    if args.epochs:
        cfg["training"]["num_epochs"] = args.epochs
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr

    train(cfg)
