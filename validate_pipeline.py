#!/usr/bin/env python3
"""Quick pipeline validation against the sample DBC."""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from training_data import (
    parse_dbc, generate_all_pairs, format_pair_as_instruction
)

def main():
    dbc_path = Path("data/sample_vehicle.dbc")
    print(f"[*] Parsing {dbc_path}...")

    vehicle = parse_dbc(dbc_path, "test_manufacturer")

    print(f"\n  Model hint: {vehicle.model_hint}")
    print(f"  Messages: {len(vehicle.messages)}")
    print(f"  Signal categories found:")
    for cat, sigs in vehicle.signal_map.items():
        sig_names = [s.signal_name for s in sigs]
        print(f"    {cat}: {sig_names}")

    intents = vehicle.controllable_intents()
    print(f"\n  Controllable intents: {sorted(intents)}")

    # Generate training pairs
    meta = {"manufacturer": "test", "model": "sample_vehicle", "platform": "test"}
    pairs = generate_all_pairs(vehicle, meta)
    print(f"\n[+] Generated {len(pairs)} training pairs")

    # Show a few examples
    for i, pair in enumerate(pairs[:3]):
        formatted = format_pair_as_instruction(pair)
        print(f"\n{'='*60}")
        print(f"EXAMPLE {i+1} — intent: {pair.intent}")
        print(f"{'='*60}")
        print(f"\n--- INSTRUCTION ---")
        print(formatted["instruction"][:500])
        print(f"\n--- RESPONSE ---")
        print(formatted["response"])

    # Write all pairs to JSONL
    out = Path("data/test_training_pairs.jsonl")
    with open(out, "w") as f:
        for pair in pairs:
            formatted = format_pair_as_instruction(pair)
            f.write(json.dumps(formatted) + "\n")

    print(f"\n[+] Wrote {len(pairs)} pairs to {out}")

    # Summary by intent
    from collections import Counter
    intent_counts = Counter(p.intent for p in pairs)
    print(f"\n  Pairs by intent:")
    for intent, count in sorted(intent_counts.items()):
        print(f"    {intent:20s} {count}")

    print(f"\n[✓] Pipeline validation passed.")


if __name__ == "__main__":
    main()
