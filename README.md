# Vehicle SLM — Sprint 0 Codebase

A small language model that speaks CAN bus. Given a vehicle's ambient traffic and an operator intent (`stop`, `brake`, `steer_left`), it generates the exact CAN frame to execute the command.

## Architecture

```
  Operator Intent           Observed CAN Traffic        Vehicle Metadata
  ("stop")                  (0x140: AB CD ...)          (make: Ford)
       │                          │                          │
       └──────────────────────────┼──────────────────────────┘
                                  │
                          ┌───────▼────────┐
                          │  Vehicle SLM   │
                          │  (Phi-3 / Qwen │
                          │   + QLoRA)     │
                          └───────┬────────┘
                                  │
                          ┌───────▼────────┐
                          │  Command Frame │
                          │  0x140: 00 FF  │
                          │  00 00 00 00   │
                          └────────────────┘
```

## Project Structure

```
vehicle-slm/
├── inventory_opendbc.py      # Clone & catalog openDBC by manufacturer family
├── training_data.py          # DBC → training pair generator (the core pipeline)
├── train.py                  # QLoRA fine-tuning via Unsloth
├── eval.py                   # Evaluation harness (family-holdout, byte-match)
├── infer.py                  # Runtime inference & interactive REPL
├── validate_pipeline.py      # End-to-end pipeline test
├── requirements.txt
├── configs/
│   └── training_config.yaml
└── data/
    └── sample_vehicle.dbc    # Test DBC for pipeline validation
```

## Sprint 0 Checklist

### 1. Clone openDBC & inventory

```bash
python inventory_opendbc.py
```

Clones `commaai/opendbc`, parses every DBC file, classifies signals by intent (brake, throttle, steering, speed), groups by manufacturer family. Outputs `data/opendbc_inventory.json`.

### 2. Generate training data

```bash
# From openDBC
python training_data.py data/opendbc -o data/training_pairs.jsonl

# Or from any directory of DBC files
python training_data.py /path/to/your/dbcs
```

Each training pair maps: `(vehicle_meta + traffic + intent) → command_frame`

The generator creates 3x augmented pairs (same command, different ambient traffic) per signal × intensity combination. A single DBC with brake/throttle/steering yields ~23 base pairs → ~69 augmented.

### 3. Pick base model

**Recommended:** `Phi-3-mini-4k-instruct` (3.8B) — good structured output, handles hex tokens well.

**Alternative:** `Qwen 2.5 3B` — slightly smaller, also strong on structured generation.

**Budget option:** `Qwen 2.5 1.5B` — fits comfortably on 8GB VRAM.

All run via Unsloth with 4-bit quantization (QLoRA).

### 4. Capture real CAN traffic (Matt's ESP32)

Place captured traffic files in `data/captures/`. Format: one frame per line as `0x1A3: AA BB CC DD EE FF 00 00`.

These captures are used both as additional training context and as ground truth for validation.

### 5. Run first fine-tune

```bash
# Using defaults (Phi-3-mini, QLoRA, 3 epochs)
python train.py

# With config
python train.py --config configs/training_config.yaml

# Quick test with fewer epochs
python train.py --epochs 1
```

### 6. Evaluate

```bash
# Evaluate against held-out data
python eval.py data/eval_pairs.jsonl --model-dir models/vehicle-slm-v0
```

The eval harness reports:
- **CAN ID accuracy** — did it target the right message?
- **Exact byte match** — are the command bytes correct?
- **Safety-critical score** — separate metric for brake/steering
- **Accuracy vs traffic window** — how does performance degrade with less observed traffic?

### 7. Interactive testing

```bash
python infer.py models/vehicle-slm-v0 --interactive
```

## Training Data Format

Each JSONL line contains an instruction/response pair:

```
INSTRUCTION:
<vehicle>
make: ford
model: f150 2018
platform: unknown
</vehicle>
<traffic>
  0x100: 1B A6 29 CF EC 39 5D 9A
  0x140: BD CE A4 B0 6A 21 7D 98
  0x200: 38 07 00 00 00 00 00 00
</traffic>
<intent>brake</intent>
<params>intensity=0.75</params>

RESPONSE:
<response>
  0x140: FF BF 00 00 00 00 00 00
</response>
<explanation>Apply 75% brake via BrakePressure (value=4915.1kPa)</explanation>
```

## Hardware Requirements

| Component | Sprint 0 | Production Edge |
|-----------|----------|-----------------|
| Training GPU | RTX 2070 (8GB) w/ QLoRA | 24GB+ preferred |
| Inference | Same GPU or CPU | Jetson Orin / dedicated |
| CAN hardware | CAN Commander https://rabbit-labs.com/product/cancommander/ | Same |
| Training time | ~2-4 hrs (1k examples, 3 epochs) | Scales with data |

## Key Design Decisions

- **Family-holdout eval**: The test set contains entire manufacturer families the model has never seen. This measures generalization (linguistic competence), not memorization.
- **Exact byte match for safety**: For brake and steering, partial credit doesn't count. The frame either does the right thing or it doesn't.
- **Ambient traffic as context**: The model sees what the vehicle is broadcasting, which constrains its inference about the vehicle's CAN encoding conventions.
- **3x augmentation**: Same command, different ambient traffic snapshots, so the model doesn't overfit to specific traffic patterns.
