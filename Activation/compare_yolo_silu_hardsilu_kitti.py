#!/usr/bin/env python3
"""Compare native SiLU and Hard-SiLU on YOLO using real KITTI images."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

try:
    from Activation.compare_yolo_activations_kitti import (
        DEFAULT_DATA,
        DEFAULT_GIT_MODEL,
        PROJECT_ROOT,
        ActivationRecorder,
        TimedActivation,
        benchmark_kitti_forward,
        configure_for_validation,
        create_kitti_latency_validator,
        load_model,
        replace_silu,
        validate_kitti,
    )
except ModuleNotFoundError:
    from compare_yolo_activations_kitti import (
        DEFAULT_DATA,
        DEFAULT_GIT_MODEL,
        PROJECT_ROOT,
        ActivationRecorder,
        TimedActivation,
        benchmark_kitti_forward,
        configure_for_validation,
        create_kitti_latency_validator,
        load_model,
        replace_silu,
        validate_kitti,
    )


DEFAULT_RESULTS = PROJECT_ROOT / "Activation/yolo_kitti_silu_hardsilu_comparison.json"


def count_registered_silu(model: nn.Module) -> int:
    """Count SiLU references, including a module shared by multiple parents."""
    return sum(
        isinstance(child, nn.SiLU)
        for parent in model.modules()
        for child in parent._modules.values()
    )


def prepare_models(
    base_model: nn.Module,
) -> tuple[
    dict[str, nn.Module],
    dict[str, int],
    dict[str, ActivationRecorder],
]:
    native_model = base_model
    hard_model = copy.deepcopy(base_model)
    silu_count = count_registered_silu(native_model)
    if silu_count == 0:
        raise RuntimeError("The model contains no nn.SiLU modules")

    native_silu = next(
        module for module in native_model.modules() if isinstance(module, nn.SiLU)
    )
    recorders = {
        "native_silu": ActivationRecorder(),
        "hard_silu": ActivationRecorder(),
    }
    replacement_counts = {
        "native_silu": replace_silu(
            native_model,
            TimedActivation(
                nn.SiLU(inplace=native_silu.inplace), recorders["native_silu"]
            ),
        ),
        # PyTorch calls Hard-SiLU Hardswish; the mathematical function is the same.
        "hard_silu": replace_silu(
            hard_model,
            TimedActivation(
                nn.Hardswish(inplace=native_silu.inplace), recorders["hard_silu"]
            ),
        ),
    }
    if any(count != silu_count for count in replacement_counts.values()):
        raise RuntimeError(
            f"Incomplete SiLU replacement: expected={silu_count}, "
            f"actual={replacement_counts}"
        )

    models = {"native_silu": native_model, "hard_silu": hard_model}
    for model in models.values():
        configure_for_validation(model)
    return models, replacement_counts, recorders


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
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument(
        "--latency-max-images",
        type=int,
        default=0,
        help="Maximum KITTI images for latency; 0 uses the full validation set",
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_validation and args.skip_latency:
        raise ValueError("Cannot skip both validation and latency")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    data_path = args.data.expanduser().resolve()
    results_path = args.results.expanduser().resolve()
    model_description = (
        str(args.model.expanduser().resolve()) if args.model else args.git_object
    )

    base_model = load_model(args.model, args.git_object)
    models, replacement_counts, recorders = prepare_models(base_model)
    latency_validator = (
        None
        if args.skip_latency
        else create_kitti_latency_validator(
            base_model, data_path, args.imgsz, args.batch, args.workers
        )
    )

    comparisons: dict[str, Any] = {}
    for name, model in models.items():
        print(f"\n=== {name} ===")
        entry: dict[str, Any] = {
            "activation_references": replacement_counts[name]
        }
        if not args.skip_latency:
            # Ultralytics validation can change the global CPU thread setting.
            torch.set_num_threads(args.threads)
            assert latency_validator is not None
            entry["latency"] = benchmark_kitti_forward(
                model,
                latency_validator,
                recorders[name],
                args.latency_warmup,
                args.latency_max_images,
                "KITTI",
            )
            latency = entry["latency"]
            activation = latency["activation"]
            print(
                "model latency: "
                f"median={latency['median_ms']:.3f} ms, "
                f"p90={latency['p90_ms']:.3f} ms, "
                f"p99={latency['p99_ms']:.3f} ms"
            )
            print(
                "activation latency: "
                f"median={activation['median_ms']:.3f} ms, "
                f"p90={activation['p90_ms']:.3f} ms, "
                f"p99={activation['p99_ms']:.3f} ms, "
                f"share={activation['aggregate_wall_clock_percent']:.2f}%"
            )
        if not args.skip_validation:
            entry["validation"] = validate_kitti(
                model, data_path, args.imgsz, args.batch, args.workers
            )
            validation = entry["validation"]
            print(
                "KITTI: "
                f"mAP50={validation['map50']:.6f}, "
                f"mAP50-95={validation['map50_95']:.6f}"
            )
        comparisons[name] = entry

    native = comparisons["native_silu"]
    hard = comparisons["hard_silu"]
    relative: dict[str, float] = {}
    if "latency" in native:
        for metric in ("median_ms", "p90_ms", "p99_ms"):
            relative[f"model_{metric}_change_percent"] = (
                100.0
                * (hard["latency"][metric] - native["latency"][metric])
                / native["latency"][metric]
            )
        native_activation = native["latency"]["activation"]
        hard_activation = hard["latency"]["activation"]
        relative["activation_median_ms_change_percent"] = (
            100.0
            * (hard_activation["median_ms"] - native_activation["median_ms"])
            / native_activation["median_ms"]
        )
        relative["activation_median_speedup"] = (
            native_activation["median_ms"] / hard_activation["median_ms"]
        )
    if "validation" in native:
        for metric in ("map50", "map50_95"):
            relative[f"{metric}_absolute_change"] = (
                hard["validation"][metric] - native["validation"][metric]
            )

    result = {
        "model": model_description,
        "data": str(data_path),
        "device": "cpu",
        "torch_version": torch.__version__,
        "threads": args.threads,
        "workers": args.workers,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "latency_scope": (
            "Forward-only wall clock on preprocessed KITTI validation images; "
            "excludes loading, preprocessing and NMS. Activation latency is "
            "the sum of all timed activation calls per image."
        ),
        "comparisons": comparisons,
        "hard_silu_relative_to_native_silu": relative,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved results: {results_path}")


if __name__ == "__main__":
    main()
