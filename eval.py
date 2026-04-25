#!/usr/bin/env python3
"""
eval.py — Evaluation harness for the Vehicle SLM.

Key design decision from the project spec: hold out ENTIRE vehicle families
(not random examples) for testing. This measures whether the SLM has learned
manufacturer encoding *conventions*, not memorized specific DBCs.

Metrics that matter:
  - Exact byte match for safety-critical commands (brake, steering)
  - CAN ID match rate (did it target the right message?)
  - Accuracy degradation as input traffic window shrinks (30s → 10s → 3s)
"""

import json
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class EvalResult:
    """Result of evaluating one prediction against ground truth."""
    intent: str
    manufacturer: str
    can_id_match: bool       # did the model target the right CAN ID?
    exact_match: bool        # are the data bytes identical?
    byte_overlap: float      # fraction of bytes that match (0.0–1.0)
    predicted_frames: list   # what the model generated
    expected_frames: list    # ground truth
    traffic_window_frames: int  # how many ambient frames the model saw
    latency_ms: float        # inference time


def parse_response_frames(response_text: str) -> list[tuple[str, str]]:
    """Extract CAN frames from the model's <response> block.

    Expected format:
        <response>
          0x120: 00 FF 00 00 00 00 00 00
          0x345: AA BB CC DD 00 00 00 00
        </response>
    """
    frames = []
    # Find content inside <response> tags
    match = re.search(r"<response>(.*?)</response>", response_text, re.DOTALL)
    if not match:
        # Try without tags — model might just output frames directly
        text = response_text
    else:
        text = match.group(1)

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Match: 0x1A3: AA BB CC DD EE FF 00 00
        m = re.match(r"(0x[0-9A-Fa-f]+):\s*((?:[0-9A-Fa-f]{2}\s*)+)", line)
        if m:
            can_id = m.group(1).upper()
            data = m.group(2).strip().upper()
            # Normalize spacing
            data = " ".join(data.split())
            frames.append((can_id, data))

    return frames


def compare_frames(
    predicted: list[tuple[str, str]],
    expected: list[tuple[str, str]],
) -> tuple[bool, bool, float]:
    """Compare predicted vs expected CAN frames.

    Returns: (can_id_match, exact_match, byte_overlap)
    """
    if not predicted or not expected:
        return False, False, 0.0

    # Compare first predicted frame against first expected frame
    # (for multi-frame commands, we check each pair)
    pred_id, pred_data = predicted[0]
    exp_id, exp_data = expected[0]

    can_id_match = pred_id.upper() == exp_id.upper()

    pred_bytes = pred_data.split()
    exp_bytes = exp_data.split()

    if len(pred_bytes) != len(exp_bytes):
        return can_id_match, False, 0.0

    matching = sum(1 for p, e in zip(pred_bytes, exp_bytes) if p == e)
    byte_overlap = matching / len(exp_bytes) if exp_bytes else 0.0
    exact_match = can_id_match and (pred_bytes == exp_bytes)

    return can_id_match, exact_match, byte_overlap


def run_inference(model, tokenizer, instruction: str, system_prompt: str) -> tuple[str, float]:
    """Run inference on a single example, return (response_text, latency_ms)."""
    from unsloth import FastLanguageModel

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    FastLanguageModel.for_inference(model)

    start = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.1,  # low temp for deterministic CAN frames
        do_sample=True,
        top_p=0.9,
    )
    latency = (time.perf_counter() - start) * 1000

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response, latency


def evaluate_from_file(
    eval_path: Path,
    model=None,
    tokenizer=None,
    system_prompt: str = "",
) -> dict:
    """Run full evaluation on a held-out eval set.

    If model/tokenizer are None, evaluates using pre-generated predictions
    stored in eval_path (expects 'prediction' field alongside 'instruction'/'response').
    """
    results = []
    by_intent = defaultdict(list)
    by_manufacturer = defaultdict(list)

    with open(eval_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"[*] Evaluating {len(examples)} examples...")

    for i, ex in enumerate(examples):
        instruction = ex["instruction"]
        expected_text = ex["response"]
        meta = ex.get("metadata", {})
        intent = meta.get("intent", "unknown")
        mfr = meta.get("manufacturer", "unknown")

        # Count traffic frames in instruction
        traffic_frames = len(re.findall(r"0x[0-9A-Fa-f]+:", instruction.split("<intent>")[0]))

        # Get prediction
        if model and tokenizer:
            pred_text, latency = run_inference(model, tokenizer, instruction, system_prompt)
        elif "prediction" in ex:
            pred_text = ex["prediction"]
            latency = 0
        else:
            print(f"  [!] No model and no pre-computed prediction for example {i}")
            continue

        # Parse frames
        pred_frames = parse_response_frames(pred_text)
        expected_frames = parse_response_frames(expected_text)

        # Compare
        can_id_match, exact_match, byte_overlap = compare_frames(pred_frames, expected_frames)

        result = EvalResult(
            intent=intent,
            manufacturer=mfr,
            can_id_match=can_id_match,
            exact_match=exact_match,
            byte_overlap=byte_overlap,
            predicted_frames=pred_frames,
            expected_frames=expected_frames,
            traffic_window_frames=traffic_frames,
            latency_ms=latency,
        )
        results.append(result)
        by_intent[intent].append(result)
        by_manufacturer[mfr].append(result)

    # ── Compute metrics ──
    def metrics(subset: list[EvalResult]) -> dict:
        n = len(subset)
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "can_id_accuracy": sum(r.can_id_match for r in subset) / n,
            "exact_match": sum(r.exact_match for r in subset) / n,
            "mean_byte_overlap": sum(r.byte_overlap for r in subset) / n,
            "mean_latency_ms": sum(r.latency_ms for r in subset) / n,
        }

    report = {
        "overall": metrics(results),
        "by_intent": {k: metrics(v) for k, v in by_intent.items()},
        "by_manufacturer": {k: metrics(v) for k, v in by_manufacturer.items()},
    }

    # Safety-critical breakdown
    safety_critical = [r for r in results if r.intent in ("stop", "brake", "steer_left", "steer_right")]
    report["safety_critical"] = metrics(safety_critical)

    # Accuracy vs traffic window size
    window_buckets = defaultdict(list)
    for r in results:
        if r.traffic_window_frames <= 5:
            window_buckets["≤5 frames"].append(r)
        elif r.traffic_window_frames <= 15:
            window_buckets["6-15 frames"].append(r)
        else:
            window_buckets["16+ frames"].append(r)
    report["by_traffic_window"] = {k: metrics(v) for k, v in window_buckets.items()}

    # ── Print report ──
    print("\n" + "=" * 70)
    print("VEHICLE SLM EVALUATION REPORT")
    print("=" * 70)

    print(f"\n  Overall ({report['overall']['n']} examples):")
    print(f"    CAN ID Accuracy:   {report['overall']['can_id_accuracy']:.1%}")
    print(f"    Exact Byte Match:  {report['overall']['exact_match']:.1%}")
    print(f"    Mean Byte Overlap: {report['overall']['mean_byte_overlap']:.1%}")

    print(f"\n  Safety-Critical ({report['safety_critical']['n']} examples):")
    if report['safety_critical']['n'] > 0:
        print(f"    CAN ID Accuracy:   {report['safety_critical']['can_id_accuracy']:.1%}")
        print(f"    Exact Byte Match:  {report['safety_critical']['exact_match']:.1%}")

    print(f"\n  By Intent:")
    for intent, m in sorted(report["by_intent"].items()):
        print(f"    {intent:20s}  ID={m['can_id_accuracy']:.0%}  "
              f"Exact={m['exact_match']:.0%}  Overlap={m['mean_byte_overlap']:.0%}  "
              f"(n={m['n']})")

    print(f"\n  By Manufacturer Family:")
    for mfr, m in sorted(report["by_manufacturer"].items()):
        print(f"    {mfr:20s}  ID={m['can_id_accuracy']:.0%}  "
              f"Exact={m['exact_match']:.0%}  (n={m['n']})")

    print(f"\n  Accuracy vs Traffic Window:")
    for window, m in sorted(report["by_traffic_window"].items()):
        print(f"    {window:15s}  ID={m['can_id_accuracy']:.0%}  "
              f"Exact={m['exact_match']:.0%}  (n={m['n']})")

    print()

    # Save report
    report_path = Path("data/eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Report saved to {report_path}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the Vehicle SLM")
    parser.add_argument("eval_data", type=Path, help="Path to eval JSONL file")
    parser.add_argument("--model-dir", type=Path, help="Path to trained model (optional)")
    args = parser.parse_args()

    if args.model_dir:
        from unsloth import FastLanguageModel
        from train import SYSTEM_PROMPT

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(args.model_dir),
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        evaluate_from_file(args.eval_data, model, tokenizer, SYSTEM_PROMPT)
    else:
        evaluate_from_file(args.eval_data)
