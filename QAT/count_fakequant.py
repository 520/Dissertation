#!/usr/bin/env python3
"""Count FakeQuant modules without running inference or changing their state.

Usage: python -m QAT.count_fakequant --model /path/best.pt --list
Supports this project's serialized PyTorch models and model/EMA checkpoints.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import torch
from torch import nn
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QAT.torch_qat import MinMaxFakeQuant

# Edit this path to run directly with PyCharm's Run button.
DEFAULT_MODEL = ROOT / "QAT/runs/kitti_qat_new_10e/weights/best.pt"


def count_fakequant(path: Path, list_all: bool = False) -> int:
    if path.suffix.lower() != ".pt":
        raise ValueError("Please supply a .pt model; ONNX nodes are not PyTorch FakeQuant modules")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = "raw model"
    if isinstance(checkpoint, dict):
        source = "ema" if checkpoint.get("ema") is not None else "model"
        model = checkpoint.get(source)
    else:
        model = checkpoint
    if not isinstance(model, nn.Module):
        raise TypeError("Expected a serialized model or checkpoint with model/ema; a state_dict alone is insufficient")

    modules = [(name, module) for name, module in model.named_modules()
               if isinstance(module, (MinMaxFakeQuant, FakeQuantizeBase))]
    categories, states, kinds = Counter(), Counter(), Counter()
    uninitialized = []
    print(f"Model: {path.resolve()}")
    print(f"Loaded: {source}")
    for name, module in modules:
        leaf = name.rsplit(".", 1)[-1]
        category = {"input_fake_quant": "input", "weight_fake_quant": "weight",
                    "output_fake_quant": "output"}.get(leaf, "other activation / unclassified")
        categories[category] += 1
        kinds[type(module).__name__] += 1
        # Custom modules explicitly persist this flag. Native implementations
        # differ in observer state, so do not guess their initialization.
        state = ("initialized" if module.initialized else "uninitialized") if isinstance(module, MinMaxFakeQuant) else "unknown"
        states[state] += 1
        if state == "uninitialized":
            uninitialized.append(name)
        if list_all:
            print(f"  {name}: {type(module).__name__}, {state}")

    print(f"\nFakeQuant total: {len(modules)}")
    for category in ("input", "weight", "output", "other activation / unclassified"):
        print(f"  {category}: {categories[category]}")
    for state in ("initialized", "uninitialized", "unknown"):
        print(f"  {state}: {states[state]}")
    for kind, count in sorted(kinds.items()):
        print(f"  type {kind}: {count}")
    if uninitialized:
        print("Uninitialized modules:")
        for name in uninitialized:
            print(f"  {name}")
    print("Counts refer to unique module objects, not execution counts.")
    print("Zero FakeQuant modules does not prove INT8: use check_int8.py to check execution.")
    return len(modules)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--list", action="store_true", help="show every FakeQuant module path")
    args = parser.parse_args()
    count_fakequant(args.model, args.list)


if __name__ == "__main__":
    main()
