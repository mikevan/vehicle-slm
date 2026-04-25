#!/usr/bin/env python3
"""
inventory_opendbc.py — Clone commaai/opendbc and inventory all DBC files.

Groups DBCs by manufacturer family, counts signals per file, and identifies
which vehicle functions (brake, throttle, steering) are represented.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from collections import defaultdict

try:
    import cantools
except ImportError:
    print("pip install cantools")
    sys.exit(1)

OPENDBC_REPO = "https://github.com/commaai/opendbc.git"
OPENDBC_DIR = Path("data/opendbc")

# Keywords that identify controllable vehicle functions
INTENT_KEYWORDS = {
    "brake": ["brake", "brk", "decel", "abs"],
    "throttle": ["throttle", "accel", "gas", "pedal", "torque_request", "torq_req"],
    "steering": ["steer", "strg", "sas", "eps", "lka", "lane"],
    "speed": ["speed", "veh_spd", "wheel_spd", "whl_spd"],
    "gear": ["gear", "trans", "shift", "prnd"],
    "turn_signal": ["turn", "signal", "blinker", "indicator"],
    "engine": ["engine", "eng_", "rpm", "eng_spd"],
    "door": ["door", "lock", "unlock"],
    "light": ["light", "headl", "lamp", "beam"],
}


def clone_opendbc(target: Path = OPENDBC_DIR) -> Path:
    """Clone or update the opendbc repo."""
    if target.exists():
        print(f"[+] opendbc already cloned at {target}")
        subprocess.run(["git", "pull"], cwd=target, capture_output=True)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Cloning opendbc to {target} ...")
    subprocess.run(["git", "clone", "--depth=1", OPENDBC_REPO, str(target)], check=True)
    return target


def find_dbc_files(root: Path) -> list[Path]:
    """Recursively find all .dbc files."""
    return sorted(root.rglob("*.dbc"))


def guess_manufacturer(path: Path) -> str:
    """Guess manufacturer family from file path or name."""
    parts = str(path).lower()
    families = {
        "toyota": ["toyota", "lexus", "scion"],
        "honda": ["honda", "acura"],
        "ford": ["ford", "lincoln", "mercury"],
        "gm": ["gm", "chevrolet", "chevy", "cadillac", "buick", "gmc"],
        "stellantis": ["chrysler", "dodge", "jeep", "ram", "fiat"],
        "hyundai_kia": ["hyundai", "kia", "genesis"],
        "vw_group": ["volkswagen", "vw", "audi", "porsche", "skoda", "seat"],
        "bmw": ["bmw", "mini"],
        "mercedes": ["mercedes", "benz", "daimler"],
        "nissan": ["nissan", "infiniti", "datsun"],
        "subaru": ["subaru"],
        "mazda": ["mazda"],
        "tesla": ["tesla"],
        "volvo": ["volvo"],
    }
    for family, keywords in families.items():
        if any(kw in parts for kw in keywords):
            return family
    return "unknown"


def classify_signals(db: cantools.database.Database) -> dict[str, list[str]]:
    """Classify signals by vehicle intent using keyword matching."""
    found = defaultdict(list)
    for msg in db.messages:
        for sig in msg.signals:
            name_lower = sig.name.lower()
            for intent, keywords in INTENT_KEYWORDS.items():
                if any(kw in name_lower for kw in keywords):
                    found[intent].append(f"{msg.name}.{sig.name}")
                    break
    return dict(found)


def inventory_file(path: Path) -> dict | None:
    """Parse a single DBC file and extract metadata."""
    try:
        db = cantools.database.load_file(str(path))
    except Exception as e:
        return {"path": str(path), "error": str(e)}

    intents = classify_signals(db)

    return {
        "path": str(path),
        "manufacturer": guess_manufacturer(path),
        "message_count": len(db.messages),
        "signal_count": sum(len(m.signals) for m in db.messages),
        "intents_found": list(intents.keys()),
        "intent_signals": {k: len(v) for k, v in intents.items()},
        "controllable": bool({"brake", "throttle", "steering"} & set(intents.keys())),
        "messages": [
            {
                "name": m.name,
                "can_id": hex(m.frame_id),
                "length": m.length,
                "signals": [s.name for s in m.signals],
            }
            for m in db.messages
        ],
    }


def run_inventory(opendbc_path: Path | None = None) -> list[dict]:
    """Run full inventory of all DBC files."""
    root = opendbc_path or OPENDBC_DIR
    if not root.exists():
        root = clone_opendbc(root)

    dbc_files = find_dbc_files(root)
    print(f"[+] Found {len(dbc_files)} DBC files\n")

    inventory = []
    by_family = defaultdict(list)

    for f in dbc_files:
        entry = inventory_file(f)
        if entry and "error" not in entry:
            inventory.append(entry)
            by_family[entry["manufacturer"]].append(entry)

    # Summary
    print("=" * 60)
    print("OPENDBC INVENTORY SUMMARY")
    print("=" * 60)
    for family, entries in sorted(by_family.items()):
        ctrl = sum(1 for e in entries if e["controllable"])
        sigs = sum(e["signal_count"] for e in entries)
        print(f"  {family:20s}  {len(entries):3d} files  {sigs:5d} signals  {ctrl:2d} controllable")
    print("-" * 60)
    print(f"  {'TOTAL':20s}  {len(inventory):3d} files")
    print()

    # Save full inventory
    out = Path("data/opendbc_inventory.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(inventory, f, indent=2)
    print(f"[+] Full inventory saved to {out}")

    return inventory


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inventory opendbc DBC files")
    parser.add_argument("--path", type=Path, help="Path to existing opendbc clone")
    parser.add_argument("--no-clone", action="store_true", help="Skip cloning, use existing")
    args = parser.parse_args()

    if args.path:
        run_inventory(args.path)
    elif args.no_clone:
        run_inventory()
    else:
        clone_opendbc()
        run_inventory()
