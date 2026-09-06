#!/usr/bin/env python3
"""List parameter counts and dtypes for every PyTorch and ONNX model."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Iterable
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "QAT"
# MODELS_ROOT = PROJECT_ROOT / "original"

def _dtype_name(dtype: object) -> str:
    """Return concise dtype names such as float32 and int8."""
    return str(dtype).removeprefix("torch.").replace("float", "float32", 1) if str(dtype) == "torch.float" else str(dtype).removeprefix("torch.")


def _format_count(count: int) -> str:
    return f"{count:,}"


def _format_dtypes(counts: Counter[str]) -> str:
    if not counts:
        return "none"
    return ", ".join(
        f"{dtype}: {_format_count(count)}"
        for dtype, count in sorted(counts.items())
    )


def _tensor_counts(tensors: Iterable[object]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for tensor in tensors:
        counts[_dtype_name(tensor.dtype)] += tensor.numel()
    return counts


def inspect_pytorch(path: Path) -> tuple[int, Counter[str], int, Counter[str]]:
    import torch
    from torch import nn

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict) and isinstance(
        checkpoint.get("ema") or checkpoint.get("model"), nn.Module
    ):
        model = checkpoint.get("ema") or checkpoint["model"]
    else:
        raise TypeError(f"Unsupported PyTorch checkpoint structure: {type(checkpoint).__name__}")

    parameter_dtypes = _tensor_counts(model.parameters())
    # Quantized packed weights are not exposed by model.parameters().
    for module in model.modules():
        if isinstance(module, torch.ao.nn.quantized.Conv2d):
            weight = module.weight()
            parameter_dtypes[_dtype_name(weight.dtype)] += weight.numel()
            bias = module.bias()
            if bias is not None:
                parameter_dtypes[_dtype_name(bias.dtype)] += bias.numel()
    buffer_dtypes = _tensor_counts(model.buffers())
    return (
        sum(parameter_dtypes.values()),
        parameter_dtypes,
        sum(buffer_dtypes.values()),
        buffer_dtypes,
    )


def inspect_onnx(path: Path) -> tuple[int, Counter[str]]:
    import onnx

    graph = onnx.load(path, load_external_data=False).graph
    dtype_counts: Counter[str] = Counter()
    for initializer in graph.initializer:
        dtype = onnx.TensorProto.DataType.Name(initializer.data_type).lower()
        # ONNX calls IEEE single precision "float"; use the familiar name.
        if dtype == "float":
            dtype = "float32"
        dtype_counts[dtype] += math.prod(initializer.dims)
    return sum(dtype_counts.values()), dtype_counts


def main() -> None:
    paths = sorted((*MODELS_ROOT.rglob("*.pt"), *MODELS_ROOT.rglob("*.onnx")))
    if not paths:
        raise FileNotFoundError(f"No .pt or .onnx models found below {MODELS_ROOT}")

    for path in paths:
        relative_path = path.relative_to(PROJECT_ROOT)
        print(f"\n{relative_path}")
        print(f"  file size: {path.stat().st_size / 2**20:.2f} MiB")
        try:
            if path.suffix.lower() == ".pt":
                total, dtypes, buffer_total, buffer_dtypes = inspect_pytorch(path)
                print(f"  parameters: {_format_count(total)}")
                print(f"  parameter dtypes: {_format_dtypes(dtypes)}")
                print(f"  buffers: {_format_count(buffer_total)}")
                print(f"  buffer dtypes: {_format_dtypes(buffer_dtypes)}")
            else:
                total, dtypes = inspect_onnx(path)
                print(f"  initializers: {_format_count(total)}")
                print(f"  initializer dtypes: {_format_dtypes(dtypes)}")
        except Exception as error:
            print(f"  ERROR: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
