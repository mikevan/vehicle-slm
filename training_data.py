#!/usr/bin/env python3
"""
training_data.py — Generate SLM training pairs from DBC files.

Each training pair maps:
    (vehicle_metadata + observed_can_traffic + intent) → command_frame

The SLM learns to be a fluent speaker: given a vehicle's accent (its traffic)
and an intent (what you want it to do), it generates the exact CAN frame to say it.
"""

import json
import random
import struct
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from typing import Optional

try:
    import cantools
except ImportError:
    raise ImportError("pip install cantools")


# ─── Intent Definitions ──────────────────────────────────────────────────────
# Maps high-level operator intents to signal-matching keywords and the
# physical action the frame should encode.

INTENT_DEFS = {
    "stop": {
        "description": "Emergency stop — full brake, zero throttle",
        "requires": ["brake"],
        "optional": ["throttle"],
        "brake_value": "max",
        "throttle_value": 0,
    },
    "brake": {
        "description": "Apply brakes at specified intensity",
        "requires": ["brake"],
        "brake_value": "variable",  # 25%, 50%, 75%, 100%
    },
    "accelerate": {
        "description": "Apply throttle at specified intensity",
        "requires": ["throttle"],
        "throttle_value": "variable",
    },
    "steer_left": {
        "description": "Steer left at specified angle",
        "requires": ["steering"],
        "steering_value": "negative",  # convention: negative = left
    },
    "steer_right": {
        "description": "Steer right at specified angle",
        "requires": ["steering"],
        "steering_value": "positive",
    },
    "steer_center": {
        "description": "Return steering to center",
        "requires": ["steering"],
        "steering_value": 0,
    },
    "idle": {
        "description": "Maintain current state — zero inputs",
        "requires": [],
        "brake_value": 0,
        "throttle_value": 0,
    },
}

# Signal name patterns → intent categories
SIGNAL_PATTERNS = {
    "brake": ["brake", "brk", "brake_pressure", "brake_pedal", "brk_pedal",
              "brake_cmd", "abs", "deceleration", "brk_pressure"],
    "throttle": ["throttle", "accel_pedal", "gas_pedal", "pedal_pos",
                 "throttle_cmd", "throttle_pos", "accel", "torque_request",
                 "torq_req", "engine_torque"],
    "steering": ["steer", "str_angle", "eps_torque", "lka_",
                 "sas_angle", "strg_angle", "sas_", "lkas"],
    "speed": ["vehicle_speed", "veh_spd", "wheel_speed", "whl_spd",
              "speed", "vss"],
}


@dataclass
class SignalMapping:
    """A decoded signal within a CAN message."""
    message_name: str
    message_id: int
    signal_name: str
    start_bit: int
    length: int
    byte_order: str  # "little_endian" or "big_endian"
    scale: float
    offset: float
    minimum: float
    maximum: float
    unit: str
    intent_category: str  # brake, throttle, steering, speed, or "other"


@dataclass
class TrainingPair:
    """One training example for the SLM."""
    vehicle_meta: dict          # make, model, year, platform
    observed_traffic: list      # list of (can_id_hex, data_hex) ambient frames
    intent: str                 # "stop", "brake", "accelerate", etc.
    intent_params: dict         # {"intensity": 0.75} or {"angle_deg": -15}
    response_frames: list       # list of (can_id_hex, data_hex) command frames
    explanation: str            # human-readable description for training


@dataclass
class VehicleDBC:
    """Parsed DBC with classified signals."""
    path: str
    manufacturer: str
    model_hint: str
    messages: list
    signal_map: dict = field(default_factory=dict)  # intent_category → [SignalMapping]

    def controllable_intents(self) -> set[str]:
        """Which high-level intents this DBC can satisfy."""
        available = set(self.signal_map.keys())
        result = set()
        for intent, defn in INTENT_DEFS.items():
            if all(r in available for r in defn["requires"]):
                result.add(intent)
        return result


# ─── DBC Parsing ──────────────────────────────────────────────────────────────

def classify_signal(signal_name: str) -> str:
    """Classify a signal name into an intent category."""
    name = signal_name.lower()
    for category, patterns in SIGNAL_PATTERNS.items():
        if any(p in name for p in patterns):
            return category
    return "other"


def parse_dbc(path: Path, manufacturer: str = "unknown") -> VehicleDBC:
    """Parse a DBC file into a VehicleDBC with classified signals."""
    db = cantools.database.load_file(str(path))

    signal_map = defaultdict(list)
    messages = []

    for msg in db.messages:
        msg_entry = {
            "name": msg.name,
            "id": msg.frame_id,
            "length": msg.length,
            "signals": [],
        }

        for sig in msg.signals:
            category = classify_signal(sig.name)
            mapping = SignalMapping(
                message_name=msg.name,
                message_id=msg.frame_id,
                signal_name=sig.name,
                start_bit=sig.start,
                length=sig.length,
                byte_order=sig.byte_order,
                scale=sig.scale,
                offset=sig.offset,
                minimum=sig.minimum or 0,
                maximum=sig.maximum or ((2**sig.length - 1) * sig.scale + sig.offset),
                unit=sig.unit or "",
                intent_category=category,
            )
            if category != "other":
                signal_map[category].append(mapping)
            msg_entry["signals"].append(mapping)

        messages.append(msg_entry)

    model_hint = path.stem.replace("_", " ").replace("-", " ")

    return VehicleDBC(
        path=str(path),
        manufacturer=manufacturer,
        model_hint=model_hint,
        messages=messages,
        signal_map=dict(signal_map),
    )


# ─── CAN Frame Encoding ──────────────────────────────────────────────────────

def encode_signal_into_frame(
    data: bytearray,
    signal: SignalMapping,
    physical_value: float,
) -> bytearray:
    """Encode a physical value into a CAN frame's data bytes using DBC signal definition."""
    # Convert physical value to raw integer
    raw = int(round((physical_value - signal.offset) / signal.scale))
    raw = max(0, min(raw, (1 << signal.length) - 1))

    if signal.byte_order == "little_endian":
        start_byte = signal.start_bit // 8
        start_bit_in_byte = signal.start_bit % 8
        bits_remaining = signal.length
        bit_pos = 0

        while bits_remaining > 0:
            byte_idx = start_byte + (start_bit_in_byte + bit_pos) // 8
            bit_in_byte = (start_bit_in_byte + bit_pos) % 8
            bits_this_byte = min(bits_remaining, 8 - bit_in_byte)
            mask = ((1 << bits_this_byte) - 1) << bit_in_byte

            value_bits = (raw >> bit_pos) & ((1 << bits_this_byte) - 1)
            if byte_idx < len(data):
                data[byte_idx] = (data[byte_idx] & ~mask) | (value_bits << bit_in_byte)

            bit_pos += bits_this_byte
            bits_remaining -= bits_this_byte
    else:
        # big_endian (Motorola) — start_bit is MSB position
        # Use cantools convention for encoding
        start_byte = signal.start_bit // 8
        start_bit_in_byte = signal.start_bit % 8

        for i in range(signal.length):
            bit_val = (raw >> (signal.length - 1 - i)) & 1
            byte_idx = start_byte + (start_bit_in_byte - (i % 8) < 0)
            # Simplified: for Motorola, pack MSB-first
            target_byte = start_byte + i // 8
            target_bit = start_bit_in_byte - (i % 8)
            if target_bit < 0:
                target_byte += 1
                target_bit += 8
            if target_byte < len(data):
                if bit_val:
                    data[target_byte] |= (1 << target_bit)
                else:
                    data[target_byte] &= ~(1 << target_bit)

    return data


def build_command_frame(
    signal: SignalMapping,
    physical_value: float,
    frame_length: int = 8,
) -> tuple[str, str]:
    """Build a complete CAN command frame for a signal at a given value."""
    data = bytearray(frame_length)
    encode_signal_into_frame(data, signal, physical_value)
    can_id_hex = f"0x{signal.message_id:03X}"
    data_hex = " ".join(f"{b:02X}" for b in data)
    return (can_id_hex, data_hex)


# ─── Ambient Traffic Generation ──────────────────────────────────────────────

def generate_ambient_traffic(
    vehicle: VehicleDBC,
    num_frames: int = 20,
    include_speed: bool = True,
) -> list[tuple[str, str]]:
    """Generate synthetic ambient CAN traffic from a DBC.

    Simulates what you'd capture sniffing a vehicle's bus for a few seconds.
    Includes speed signals (to give the SLM context about vehicle state)
    and random other messages.
    """
    traffic = []

    # Add speed frames if available
    if include_speed and "speed" in vehicle.signal_map:
        speed_sig = vehicle.signal_map["speed"][0]
        speed_val = random.uniform(0, 60)  # 0–60 mph/kph
        frame = build_command_frame(speed_sig, speed_val)
        traffic.append(frame)

    # Add random frames from the DBC to simulate ambient bus noise
    all_msgs = [m for m in vehicle.messages if m.get("id")]
    if all_msgs:
        sample_size = min(num_frames - len(traffic), len(all_msgs))
        sampled = random.sample(all_msgs, max(1, sample_size))
        for msg in sampled:
            data = bytearray(msg.get("length", 8))
            # Fill with plausible random data
            for i in range(len(data)):
                data[i] = random.randint(0, 255)
            can_id = f"0x{msg['id']:03X}"
            data_hex = " ".join(f"{b:02X}" for b in data)
            traffic.append((can_id, data_hex))

    random.shuffle(traffic)
    return traffic


# ─── Training Pair Generation ────────────────────────────────────────────────

def generate_brake_pairs(
    vehicle: VehicleDBC,
    meta: dict,
    intensities: list[float] = [0.25, 0.5, 0.75, 1.0],
) -> list[TrainingPair]:
    """Generate brake command training pairs at various intensities."""
    pairs = []
    if "brake" not in vehicle.signal_map:
        return pairs

    for sig in vehicle.signal_map["brake"][:2]:  # use up to 2 brake signals
        for intensity in intensities:
            brake_val = sig.minimum + (sig.maximum - sig.minimum) * intensity
            frame = build_command_frame(sig, brake_val)

            pair = TrainingPair(
                vehicle_meta=meta,
                observed_traffic=generate_ambient_traffic(vehicle),
                intent="brake",
                intent_params={"intensity": intensity},
                response_frames=[frame],
                explanation=f"Apply {int(intensity*100)}% brake via {sig.signal_name} "
                           f"(value={brake_val:.1f}{sig.unit})",
            )
            pairs.append(pair)

    # Emergency stop: max brake + zero throttle
    brake_sig = vehicle.signal_map["brake"][0]
    brake_frame = build_command_frame(brake_sig, brake_sig.maximum)
    response = [brake_frame]

    if "throttle" in vehicle.signal_map:
        thr_sig = vehicle.signal_map["throttle"][0]
        thr_frame = build_command_frame(thr_sig, 0)
        response.append(thr_frame)

    pairs.append(TrainingPair(
        vehicle_meta=meta,
        observed_traffic=generate_ambient_traffic(vehicle),
        intent="stop",
        intent_params={"emergency": True},
        response_frames=response,
        explanation=f"Emergency stop: max brake + zero throttle",
    ))

    return pairs


def generate_throttle_pairs(
    vehicle: VehicleDBC,
    meta: dict,
    intensities: list[float] = [0.1, 0.25, 0.5, 0.75, 1.0],
) -> list[TrainingPair]:
    """Generate throttle command training pairs."""
    pairs = []
    if "throttle" not in vehicle.signal_map:
        return pairs

    for sig in vehicle.signal_map["throttle"][:2]:
        for intensity in intensities:
            thr_val = sig.minimum + (sig.maximum - sig.minimum) * intensity
            frame = build_command_frame(sig, thr_val)

            pairs.append(TrainingPair(
                vehicle_meta=meta,
                observed_traffic=generate_ambient_traffic(vehicle),
                intent="accelerate",
                intent_params={"intensity": intensity},
                response_frames=[frame],
                explanation=f"Apply {int(intensity*100)}% throttle via {sig.signal_name} "
                           f"(value={thr_val:.1f}{sig.unit})",
            ))

    return pairs


def generate_steering_pairs(
    vehicle: VehicleDBC,
    meta: dict,
    angles: list[float] = [-45, -30, -15, -5, 0, 5, 15, 30, 45],
) -> list[TrainingPair]:
    """Generate steering command training pairs at various angles."""
    pairs = []
    if "steering" not in vehicle.signal_map:
        return pairs

    for sig in vehicle.signal_map["steering"][:1]:  # primary steering signal
        for angle in angles:
            # Clamp to signal range
            steer_val = max(sig.minimum, min(sig.maximum, angle))
            frame = build_command_frame(sig, steer_val)

            if angle < 0:
                intent = "steer_left"
            elif angle > 0:
                intent = "steer_right"
            else:
                intent = "steer_center"

            pairs.append(TrainingPair(
                vehicle_meta=meta,
                observed_traffic=generate_ambient_traffic(vehicle),
                intent=intent,
                intent_params={"angle_deg": angle},
                response_frames=[frame],
                explanation=f"Steer {'left' if angle < 0 else 'right' if angle > 0 else 'center'} "
                           f"{abs(angle)}° via {sig.signal_name}",
            ))

    return pairs


def generate_all_pairs(vehicle: VehicleDBC, meta: dict) -> list[TrainingPair]:
    """Generate all training pairs for a vehicle."""
    pairs = []
    pairs.extend(generate_brake_pairs(vehicle, meta))
    pairs.extend(generate_throttle_pairs(vehicle, meta))
    pairs.extend(generate_steering_pairs(vehicle, meta))
    return pairs


# ─── Formatting for SLM Training ─────────────────────────────────────────────

def format_pair_as_instruction(pair: TrainingPair) -> dict:
    """Format a TrainingPair as an instruction/response pair for fine-tuning.

    This is the format the SLM sees during training. The model learns to
    generate the <response> block given everything above it.
    """
    # Build the traffic block
    traffic_lines = "\n".join(f"  {cid}: {data}" for cid, data in pair.observed_traffic)

    # Build the response block
    response_lines = "\n".join(f"  {cid}: {data}" for cid, data in pair.response_frames)

    instruction = f"""<vehicle>
make: {pair.vehicle_meta.get('manufacturer', 'unknown')}
model: {pair.vehicle_meta.get('model', 'unknown')}
platform: {pair.vehicle_meta.get('platform', 'unknown')}
</vehicle>
<traffic>
{traffic_lines}
</traffic>
<intent>{pair.intent}</intent>"""

    if pair.intent_params:
        params_str = ", ".join(f"{k}={v}" for k, v in pair.intent_params.items())
        instruction += f"\n<params>{params_str}</params>"

    response = f"""<response>
{response_lines}
</response>
<explanation>{pair.explanation}</explanation>"""

    return {
        "instruction": instruction,
        "response": response,
        "metadata": {
            "manufacturer": pair.vehicle_meta.get("manufacturer"),
            "intent": pair.intent,
            "source_dbc": pair.vehicle_meta.get("source_dbc"),
        },
    }


# ─── Pipeline Entry Point ────────────────────────────────────────────────────

def process_dbc_directory(
    dbc_dir: Path,
    output_path: Path = Path("data/training_pairs.jsonl"),
    manufacturer_fn=None,
) -> int:
    """Process all DBC files in a directory and generate training pairs."""
    from inventory_opendbc import guess_manufacturer

    dbc_files = sorted(dbc_dir.rglob("*.dbc"))
    print(f"[*] Processing {len(dbc_files)} DBC files...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_pairs = 0
    pair_counts = defaultdict(int)

    with open(output_path, "w") as out:
        for dbc_path in dbc_files:
            try:
                mfr = (manufacturer_fn or guess_manufacturer)(dbc_path)
                vehicle = parse_dbc(dbc_path, mfr)
            except Exception as e:
                print(f"  [!] Failed to parse {dbc_path.name}: {e}")
                continue

            intents = vehicle.controllable_intents()
            if not intents:
                continue

            meta = {
                "manufacturer": mfr,
                "model": vehicle.model_hint,
                "platform": "unknown",
                "source_dbc": str(dbc_path),
            }

            pairs = generate_all_pairs(vehicle, meta)

            # Augment: generate multiple ambient traffic variations per pair
            augmented = []
            for pair in pairs:
                augmented.append(pair)
                # 2 additional traffic variations per pair
                for _ in range(2):
                    aug = TrainingPair(
                        vehicle_meta=pair.vehicle_meta,
                        observed_traffic=generate_ambient_traffic(vehicle),
                        intent=pair.intent,
                        intent_params=pair.intent_params,
                        response_frames=pair.response_frames,
                        explanation=pair.explanation,
                    )
                    augmented.append(aug)

            for pair in augmented:
                formatted = format_pair_as_instruction(pair)
                out.write(json.dumps(formatted) + "\n")
                pair_counts[pair.intent] += 1
                total_pairs += 1

            if intents:
                print(f"  [+] {dbc_path.name}: {len(augmented)} pairs "
                      f"({', '.join(sorted(intents))})")

    print(f"\n{'='*60}")
    print(f"TRAINING DATA SUMMARY")
    print(f"{'='*60}")
    for intent, count in sorted(pair_counts.items()):
        print(f"  {intent:20s}  {count:5d} pairs")
    print(f"  {'TOTAL':20s}  {total_pairs:5d} pairs")
    print(f"\n[+] Saved to {output_path}")

    return total_pairs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate SLM training data from DBC files")
    parser.add_argument("dbc_dir", type=Path, help="Directory containing DBC files")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/training_pairs.jsonl"))
    args = parser.parse_args()

    process_dbc_directory(args.dbc_dir, args.output)
