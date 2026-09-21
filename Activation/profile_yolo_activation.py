#!/usr/bin/env python3
"""Measure how much YOLO forward time is spent in SiLU activations."""

from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GIT_MODEL = "HEAD:original/yolov8n_kitti/final_model_factorized_lwi.pt"
DEFAULT_RESULTS = PROJECT_ROOT / "Activation/original_kitti_activation_profile.json"


def load_model(path: Path | None, git_object: str) -> nn.Module:
    if path is not None:
        source: Any = path.expanduser().resolve()
    else:
        blob = subprocess.run(
            ["git", "show", git_object],
            cwd=PROJECT_ROOT,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        source = io.BytesIO(blob)

    loaded = torch.load(source, map_location="cpu", weights_only=False)
    if isinstance(loaded, dict):
        loaded = loaded.get("ema") or loaded.get("model")
    if not isinstance(loaded, nn.Module):
        raise TypeError(f"Expected a PyTorch model, got {type(loaded).__name__}")
    return loaded.float().eval()


def replace_silu_with_identity(module: nn.Module, replacement: nn.Identity | None = None) -> int:
    identity = replacement if replacement is not None else nn.Identity()
    count = 0
    for name, child in list(module._modules.items()):
        if child is None:
            continue
        if isinstance(child, nn.SiLU):
            module._modules[name] = identity
            count += 1
        else:
            count += replace_silu_with_identity(child, identity)
    return count


@torch.inference_mode()
def run_forward(model: nn.Module, tensor: torch.Tensor) -> Any:
    return model(tensor)


def median_forward_ms(model: nn.Module, tensor: torch.Tensor, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        run_forward(model, tensor)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        run_forward(model, tensor)
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(torch.tensor(samples, dtype=torch.float64).median())


def operator_profile(model: nn.Module, tensor: torch.Tensor, warmup: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmup):
        run_forward(model, tensor)

    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        for _ in range(repeats):
            with record_function("model_forward"):
                run_forward(model, tensor)

    events = profiler.key_averages()
    forward = next(event for event in events if event.key == "model_forward")
    silu_events = [event for event in events if "silu" in event.key.lower()]
    silu_self_us = sum(event.self_cpu_time_total for event in silu_events)
    forward_total_us = forward.cpu_time_total
    return {
        "profiled_forwards": repeats,
        "forward_total_ms": forward_total_us / 1000.0,
        "silu_self_total_ms": silu_self_us / 1000.0,
        "silu_operator_percent": 100.0 * silu_self_us / forward_total_us,
        "silu_operators": [
            {
                "name": event.key,
                "calls": event.count,
                "self_cpu_total_ms": event.self_cpu_time_total / 1000.0,
            }
            for event in silu_events
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", type=Path)
    source.add_argument("--git-object", default=DEFAULT_GIT_MODEL)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--profile-repeats", type=int, default=20)
    parser.add_argument("--threads", type=int, default=torch.get_num_threads())
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    model = load_model(args.model, args.git_object)
    without_silu = copy.deepcopy(model)
    replaced = replace_silu_with_identity(without_silu)
    if not replaced:
        raise RuntimeError("The model contains no nn.SiLU references")

    tensor = torch.randn(args.batch, 3, args.imgsz, args.imgsz)
    native_ms = median_forward_ms(model, tensor, args.warmup, args.repeats)
    identity_ms = median_forward_ms(without_silu, tensor, args.warmup, args.repeats)
    removed_ms = native_ms - identity_ms
    result = {
        "model": str(args.model.resolve()) if args.model else args.git_object,
        "device": "cpu",
        "threads": args.threads,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "silu_references": replaced,
        "wall_clock": {
            "repeats": args.repeats,
            "native_silu_median_ms": native_ms,
            "identity_median_ms": identity_ms,
            "difference_ms": removed_ms,
            "estimated_activation_percent": 100.0 * removed_ms / native_ms,
        },
        "operator_profile": operator_profile(
            model, tensor, args.warmup, args.profile_repeats
        ),
    }

    results = args.results.expanduser().resolve()
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved profile: {results}")


if __name__ == "__main__":
    main()
