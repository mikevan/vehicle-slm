#!/usr/bin/env python3
"""
infer.py — Runtime inference for the Vehicle SLM.

This is the operational interface: given live CAN traffic and an intent,
generate the command frame. Designed to be called from the Karaf runtime
or used standalone for testing.
"""

import json
import time
import sys
from pathlib import Path
from typing import Optional

from train import SYSTEM_PROMPT


class VehicleSLM:
    """Loaded Vehicle SLM ready for inference."""

    def __init__(self, model_dir: str, max_seq_length: int = 2048):
        from unsloth import FastLanguageModel

        print(f"[*] Loading model from {model_dir}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        print("[+] Model loaded and ready.")

    def generate_command(
        self,
        traffic: list[tuple[str, str]],
        intent: str,
        vehicle_meta: Optional[dict] = None,
        intent_params: Optional[dict] = None,
        temperature: float = 0.1,
        max_tokens: int = 256,
    ) -> dict:
        """Generate a CAN command frame from traffic observation + intent.

        Args:
            traffic: List of (can_id_hex, data_hex) observed on the bus
            intent: "stop", "brake", "accelerate", "steer_left", "steer_right", etc.
            vehicle_meta: Optional dict with make/model/platform info
            intent_params: Optional dict like {"intensity": 0.75}
            temperature: Sampling temperature (low = deterministic)
            max_tokens: Max tokens to generate

        Returns:
            dict with 'frames', 'raw_response', 'latency_ms'
        """
        meta = vehicle_meta or {"manufacturer": "unknown", "model": "unknown", "platform": "unknown"}

        # Build the instruction
        traffic_lines = "\n".join(f"  {cid}: {data}" for cid, data in traffic)

        instruction = f"""<vehicle>
make: {meta.get('manufacturer', 'unknown')}
model: {meta.get('model', 'unknown')}
platform: {meta.get('platform', 'unknown')}
</vehicle>
<traffic>
{traffic_lines}
</traffic>
<intent>{intent}</intent>"""

        if intent_params:
            params_str = ", ".join(f"{k}={v}" for k, v in intent_params.items())
            instruction += f"\n<params>{params_str}</params>"

        # Format as chat
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        # Generate
        start = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
        )
        latency = (time.perf_counter() - start) * 1000

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Parse frames from response
        from eval import parse_response_frames
        frames = parse_response_frames(response)

        return {
            "frames": frames,
            "raw_response": response,
            "latency_ms": round(latency, 1),
            "intent": intent,
        }


def interactive_mode(model_dir: str):
    """Interactive REPL for testing the SLM."""
    slm = VehicleSLM(model_dir)

    print("\n" + "=" * 60)
    print("VEHICLE SLM — INTERACTIVE MODE")
    print("=" * 60)
    print("Commands: stop, brake, accelerate, steer_left, steer_right")
    print("Enter traffic as: 0x1A3:AABBCCDD  (one per line, blank to finish)")
    print("Type 'quit' to exit.\n")

    while True:
        intent = input("Intent> ").strip().lower()
        if intent in ("quit", "exit", "q"):
            break

        print("Traffic (blank line to finish):")
        traffic = []
        while True:
            line = input("  ").strip()
            if not line:
                break
            if ":" in line:
                parts = line.split(":", 1)
                can_id = parts[0].strip()
                data = parts[1].strip()
                # Normalize: add spaces between bytes if not present
                if " " not in data and len(data) >= 4:
                    data = " ".join(data[i:i+2] for i in range(0, len(data), 2))
                traffic.append((can_id, data))

        if not traffic:
            print("  (no traffic provided, using empty)\n")

        result = slm.generate_command(traffic, intent)

        print(f"\n  Response ({result['latency_ms']:.0f}ms):")
        for cid, data in result["frames"]:
            print(f"    {cid}: {data}")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vehicle SLM inference")
    parser.add_argument("model_dir", type=str, help="Path to trained model")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--intent", type=str, help="Intent for single-shot mode")
    parser.add_argument("--traffic-file", type=Path, help="File with captured CAN traffic")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode(args.model_dir)
    elif args.intent:
        slm = VehicleSLM(args.model_dir)
        traffic = []
        if args.traffic_file:
            with open(args.traffic_file) as f:
                for line in f:
                    line = line.strip()
                    if ":" in line:
                        parts = line.split(":", 1)
                        traffic.append((parts[0].strip(), parts[1].strip()))

        result = slm.generate_command(traffic, args.intent)
        print(json.dumps(result, indent=2))
