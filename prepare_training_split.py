#!/usr/bin/env python3
"""
prepare_training_split.py — Rebalance intents and split train/eval by manufacturer family.

Takes the raw training_pairs.jsonl and produces:
  - data/train.jsonl  (rebalanced, holdout families removed)
  - data/eval.jsonl   (holdout families only, no rebalancing)

Holdout families are entire manufacturer families the model never sees during
training. This is the only honest eval for a linguistic model — if it can
generate correct frames for a manufacturer it's never trained on, the
approach works.
"""

import json
import random
from pathlib import Path
from collections import Counter, defaultdict

# Families held out entirely for evaluation
HOLDOUT_FAMILIES = ["subaru", "mazda"]

# Target: oversample small intents to at least this fraction of the largest intent
MIN_RATIO = 0.5  # e.g., if brake has 800, stop should have at least 400


def load_pairs(path: Path) -> list[dict]:
    pairs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def split_by_family(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split into train (non-holdout) and eval (holdout families)."""
    train, eval_ = [], []
    for p in pairs:
        mfr = p.get("metadata", {}).get("manufacturer", "unknown")
        if mfr in HOLDOUT_FAMILIES:
            eval_.append(p)
        else:
            train.append(p)
    return train, eval_


def rebalance(pairs: list[dict], min_ratio: float = MIN_RATIO) -> list[dict]:
    """Oversample underrepresented intents."""
    by_intent = defaultdict(list)
    for p in pairs:
        intent = p.get("metadata", {}).get("intent", "unknown")
        by_intent[intent].append(p)

    # Find the largest intent count
    max_count = max(len(v) for v in by_intent.values())
    target_min = int(max_count * min_ratio)

    rebalanced = []
    for intent, examples in by_intent.items():
        if len(examples) >= target_min:
            rebalanced.extend(examples)
        else:
            # Keep originals + oversample to reach target
            rebalanced.extend(examples)
            shortfall = target_min - len(examples)
            oversampled = random.choices(examples, k=shortfall)
            rebalanced.extend(oversampled)

    random.shuffle(rebalanced)
    return rebalanced


def main():
    random.seed(42)

    input_path = Path("data/training_pairs.jsonl")
    if not input_path.exists():
        print(f"[!] {input_path} not found. Run training_data.py first.")
        return

    pairs = load_pairs(input_path)
    print(f"[*] Loaded {len(pairs)} total pairs")

    # Show raw distribution
    raw_counts = Counter(p.get("metadata", {}).get("intent", "?") for p in pairs)
    print(f"\n  Raw intent distribution:")
    for intent, count in sorted(raw_counts.items()):
        print(f"    {intent:20s} {count:5d}")

    # Split by family
    train, eval_ = split_by_family(pairs)
    print(f"\n[*] Split: {len(train)} train, {len(eval_)} eval (holdout: {', '.join(HOLDOUT_FAMILIES)})")

    # Show eval family distribution
    eval_counts = Counter(p.get("metadata", {}).get("intent", "?") for p in eval_)
    print(f"\n  Eval intent distribution:")
    for intent, count in sorted(eval_counts.items()):
        print(f"    {intent:20s} {count:5d}")

    # Rebalance training set
    train_rebalanced = rebalance(train)

    # Show rebalanced distribution
    rebal_counts = Counter(p.get("metadata", {}).get("intent", "?") for p in train_rebalanced)
    print(f"\n  Rebalanced train distribution:")
    for intent, count in sorted(rebal_counts.items()):
        print(f"    {intent:20s} {count:5d}")

    # Save
    train_path = Path("data/train.jsonl")
    eval_path = Path("data/eval.jsonl")

    with open(train_path, "w") as f:
        for p in train_rebalanced:
            f.write(json.dumps(p) + "\n")

    with open(eval_path, "w") as f:
        for p in eval_:
            f.write(json.dumps(p) + "\n")

    print(f"\n[+] Saved {len(train_rebalanced)} train pairs to {train_path}")
    print(f"[+] Saved {len(eval_)} eval pairs to {eval_path}")
    print(f"\n[✓] Ready for fine-tuning.")


if __name__ == "__main__":
    main()
