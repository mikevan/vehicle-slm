# Vehicle SLM v0

A QLoRA fine-tuned version of [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) trained to generate CAN bus command frames from vehicle metadata, observed traffic, and operator intent.

## What This Model Does

Given a vehicle type, a snippet of raw CAN bus traffic, and a command intent (e.g. "brake_full", "steer_left", "throttle_release"), the model generates the CAN frame needed to execute that command on the target vehicle.

This model does not perform a dictionary lookup. It generates frames directly from internalized knowledge of how manufacturers encode control signals — the same way a fluent speaker produces language without consulting a grammar textbook.

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = "mikevan/vehicle-slm/models/vehicle-slm-v0"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, adapter_path)

prompt = """<|system|>
You are a Vehicle CAN Bus Command Generator.<|end|>
<|user|>
<vehicle>toyota_camry_2019</vehicle>
<traffic>
0x025:00:00:00:00:00:00:00:00
0x0B4:00:00:00:00:00:00:00:00
</traffic>
<intent>brake_full</intent><|end|>
<|assistant|>"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training

- **Base model:** Qwen/Qwen2.5-1.5B-Instruct
- **Method:** QLoRA (4-bit quantization, LoRA rank 32)
- **Dataset:** 3,675 training pairs derived from the commaai/opendbc DBC corpus spanning 14 manufacturer families
- **Eval set:** 432 examples held out by manufacturer family (Subaru, Mazda)
- **Final eval token accuracy:** 72.7%
- **Hardware:** NVIDIA A40 48GB
- **Training time:** 77 minutes

## Framework Versions

- PEFT 0.19.1
- TRL 1.5.1
- Transformers 5.9.0
- PyTorch 2.10.0

