#!/usr/bin/env python3
"""Measure native SiLU's wall-clock share on real KITTI validation images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:
    from Activation.compare_yolo_activations_kitti import (
        DEFAULT_DATA,
        DEFAULT_GIT_MODEL,
        PROJECT_ROOT,
        configure_for_validation,
        create_kitti_latency_validator,
        load_model,
    )
except ModuleNotFoundError:
    from compare_yolo_activations_kitti import (
        DEFAULT_DATA,
        DEFAULT_GIT_MODEL,
        PROJECT_ROOT,
        configure_for_validation,
        create_kitti_latency_validator,
        load_model,
    )


DEFAULT_RESULTS = PROJECT_ROOT / "Activation/kitti_activation_share.json"


class ActivationRecorder:
    def __init__(self) -> None:
        self.elapsed_ns = 0
        self.calls = 0
        self.elements = 0

    def reset(self) -> None:
        self.elapsed_ns = 0
        self.calls = 0
        self.elements = 0


class TimedSiLU(nn.Module):
    """Native PyTorch SiLU with an inner wall-clock timer."""

    def __init__(self, recorder: ActivationRecorder, inplace: bool) -> None:
        super().__init__()
        self.recorder = recorder
        self.inplace = inplace

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        start = time.perf_counter_ns()
        output = F.silu(tensor, inplace=self.inplace)
        self.recorder.elapsed_ns += time.perf_counter_ns() - start
        self.recorder.calls += 1
        self.recorder.elements += tensor.numel()
        return output


def replace_with_timed_silu(
    module: nn.Module,
    recorder: ActivationRecorder,
) -> int:
    count = 0
    for name, child in list(module._modules.items()):
        if child is None:
            continue
        if isinstance(child, nn.SiLU):
            module._modules[name] = TimedSiLU(recorder, child.inplace)
            count += 1
        else:
            count += replace_with_timed_silu(child, recorder)
    return count


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


@torch.inference_mode()
def measure(
    model: nn.Module,
    validator: Any,
    recorder: ActivationRecorder,
    warmup: int,
    maximum_images: int,
) -> dict[str, Any]:
    first_batch = next(iter(validator.dataloader))
    warmup_tensor = validator.preprocess(first_batch)["img"]
    for _ in range(warmup):
        recorder.reset()
        model(warmup_tensor)

    forward_ms: list[float] = []
    activation_ms: list[float] = []
    activation_percent: list[float] = []
    calls_per_batch: list[int] = []
    total_elements = 0
    images_measured = 0

    for raw_batch in validator.dataloader:
        batch = validator.preprocess(raw_batch)
        tensor = batch["img"]
        if maximum_images and images_measured + tensor.shape[0] > maximum_images:
            tensor = tensor[: maximum_images - images_measured]
        if tensor.shape[0] == 0:
            break

        recorder.reset()
        start = time.perf_counter_ns()
        model(tensor)
        total_ns = time.perf_counter_ns() - start

        batch_images = int(tensor.shape[0])
        one_forward_ms = total_ns / 1e6 / batch_images
        one_activation_ms = recorder.elapsed_ns / 1e6 / batch_images
        forward_ms.append(one_forward_ms)
        activation_ms.append(one_activation_ms)
        activation_percent.append(100.0 * recorder.elapsed_ns / total_ns)
        calls_per_batch.append(recorder.calls)
        total_elements += recorder.elements
        images_measured += batch_images
        if maximum_images and images_measured >= maximum_images:
            break

    summed_forward_ms = float(np.sum(forward_ms))
    summed_activation_ms = float(np.sum(activation_ms))
    return {
        "images": images_measured,
        "timed_batches": len(forward_ms),
        "silu_calls_per_batch": {
            "minimum": min(calls_per_batch),
            "maximum": max(calls_per_batch),
        },
        "activation_elements_total": total_elements,
        "forward_ms_per_image": distribution(forward_ms),
        "silu_ms_per_image": distribution(activation_ms),
        "silu_percent_per_image": distribution(activation_percent),
        "aggregate_silu_wall_clock_percent": (
            100.0 * summed_activation_ms / summed_forward_ms
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", type=Path)
    source.add_argument("--git-object", default=DEFAULT_GIT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--threads", type=int, default=torch.get_num_threads())
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum KITTI images to time; 0 uses the full validation set",
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads < 1 or args.batch < 1 or args.max_images < 0:
        raise ValueError("threads and batch must be positive; max-images cannot be negative")

    torch.set_num_threads(args.threads)
    model = load_model(args.model, args.git_object)
    configure_for_validation(model)
    validator = create_kitti_latency_validator(
        model,
        args.data.expanduser().resolve(),
        args.imgsz,
        args.batch,
        args.workers,
    )
    recorder = ActivationRecorder()
    replaced = replace_with_timed_silu(model, recorder)
    if replaced == 0:
        raise RuntimeError("The model contains no nn.SiLU references")

    measurements = measure(
        model,
        validator,
        recorder,
        args.warmup,
        args.max_images,
    )
    result = {
        "model": (
            str(args.model.expanduser().resolve())
            if args.model
            else args.git_object
        ),
        "data": str(args.data.expanduser().resolve()),
        "device": "cpu",
        "threads": args.threads,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "silu_references_replaced": replaced,
        "scope": (
            "Real preprocessed KITTI images; wall-clock model forward only. "
            "Excludes image loading, preprocessing and NMS. SiLU time is the "
            "sum of inner timers around torch.nn.functional.silu calls."
        ),
        **measurements,
    }

    results_path = args.results.expanduser().resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved results: {results_path}")


if __name__ == "__main__":
    main()
