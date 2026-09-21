#!/usr/bin/env python3
"""Compare all C++ activation approximations from comparesilu.cpp on KITTI."""

from __future__ import annotations

import argparse
import copy
import ctypes
import io
import json
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from types import MethodType
from typing import Any

import numpy as np
import torch
from torch import nn
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import BaseModel

try:
    from Activation.compare_c_lut21_silu import (
        compile_torch_cpp_extension,
    )
except ModuleNotFoundError:
    from compare_c_lut21_silu import compile_torch_cpp_extension


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GIT_MODEL = "HEAD:original/yolov8n_kitti/final_model_factorized_lwi.pt"
DEFAULT_DATA = PROJECT_ROOT / "datasets/kitti/kitti.yaml"
DEFAULT_RESULTS = PROJECT_ROOT / "Activation/yolo_kitti_activation_ablation.json"
DEFAULT_DATASET_LABEL = "KITTI"

CPP_ACTIVATIONS = {
    "cpp_exact_silu": "exact_silu_f32",
    "cpp_taylor3_linear": "taylor3_silu_f32",
    "cpp_range_poly": "range_poly_silu_f32",
    "cpp_fast_sigmoid": "fast_sigmoid_silu_f32",
    "cpp_hard_silu": "hard_silu_f32",
    "cpp_piecewise_linear": "piecewise_linear_silu_f32",
    "cpp_lut21": "lut21_silu_f32",
    "cpp_relu": "relu_f32",
}
CPP_ACTIVATIONS.update(
    {
        f"cpp_lut{points}_range{half_range}": (
            f"lut{points}_range{half_range}_silu_f32"
        )
        for points in (9, 13, 17, 21)
        for half_range in (2, 3, 4, 5)
    }
)


def compile_all_activations_library(
    compiler: str, output: Path
) -> list[str]:
    source = Path(__file__).with_name("comparesilu.cpp")
    command = [
        compiler,
        "-O3",
        "-ffast-math",
        "-march=native",
        "-std=c++17",
        "-Wall",
        "-Wextra",
    ]
    if platform.system() == "Darwin":
        command.append("-dynamiclib")
    else:
        command.extend(["-shared", "-fPIC"])
    command.extend([str(source), "-o", str(output), "-lm"])
    subprocess.run(command, check=True)
    return command


def load_model(path: Path | None, git_object: str) -> BaseModel:
    if path is None:
        blob = subprocess.run(
            ["git", "show", git_object],
            cwd=PROJECT_ROOT,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        source: Any = io.BytesIO(blob)
    else:
        source = path.expanduser().resolve()

    loaded = torch.load(source, map_location="cpu", weights_only=False)
    if isinstance(loaded, dict):
        loaded = loaded.get("ema") or loaded.get("model")
    if not isinstance(loaded, BaseModel):
        raise TypeError(
            f"Expected an Ultralytics BaseModel, got {type(loaded).__name__}"
        )
    return loaded.float().eval()


class CppArrayActivation(nn.Module):
    """Call one exported C++ activation kernel on a complete CPU tensor."""

    def __init__(self, function: Any) -> None:
        super().__init__()
        self.function = function

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise TypeError("C++ activation requires a CPU float32 tensor")
        contiguous = tensor.contiguous()
        output = torch.empty_like(contiguous)
        self.function(
            ctypes.c_void_p(contiguous.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            contiguous.numel(),
        )
        return output


class TorchCppLUT21SiLU(nn.Module):
    """Call the PyTorch C++/NEON LUT operator."""

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.ops.dissertation_lut21.forward(tensor)


class ActivationRecorder:
    def __init__(self) -> None:
        self.elapsed_ns = 0
        self.calls = 0
        self.elements = 0

    def reset(self) -> None:
        self.elapsed_ns = 0
        self.calls = 0
        self.elements = 0


class TimedActivation(nn.Module):
    """Measure one activation implementation without changing its output."""

    def __init__(self, activation: nn.Module, recorder: ActivationRecorder) -> None:
        super().__init__()
        self.activation = activation
        self.recorder = recorder

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        start = time.perf_counter_ns()
        output = self.activation(tensor)
        self.recorder.elapsed_ns += time.perf_counter_ns() - start
        self.recorder.calls += 1
        self.recorder.elements += tensor.numel()
        return output


def replace_silu(module: nn.Module, replacement: nn.Module) -> int:
    count = 0
    for name, child in list(module._modules.items()):
        if child is None:
            continue
        if isinstance(child, nn.SiLU):
            module._modules[name] = replacement
            count += 1
        else:
            count += replace_silu(child, replacement)
    return count


def skip_fuse(model: BaseModel, *_args: Any, **_kwargs: Any) -> BaseModel:
    """Preserve the checkpoint's factorized Sequential convolutions."""
    return model


def configure_for_validation(model: BaseModel) -> None:
    model.eval()
    model.fuse = MethodType(skip_fuse, model)


def create_kitti_latency_validator(
    model: BaseModel,
    data: Path,
    imgsz: int,
    batch: int,
    workers: int,
) -> DetectionValidator:
    """Build a KITTI loader whose preprocessing stays outside timed regions."""
    validator = DetectionValidator(
        args={
            "model": "in-memory-latency-model.pt",
            "data": str(data),
            "imgsz": imgsz,
            "batch": batch,
            "device": "cpu",
            "workers": workers,
            "split": "val",
            "rect": True,
            "plots": False,
            "save_json": False,
            "verbose": False,
        }
    )
    validator.training = False
    validator.device = torch.device("cpu")
    validator.data = check_det_dataset(str(data), split="val")
    validator.stride = int(model.stride.max())
    validator.dataloader = validator.get_dataloader(
        validator.data[validator.args.split], batch
    )
    return validator


@torch.inference_mode()
def benchmark_kitti_forward(
    model: BaseModel,
    validator: DetectionValidator,
    recorder: ActivationRecorder,
    warmup: int,
    maximum_images: int,
    dataset_label: str,
) -> dict[str, float | int | str]:
    first_batch = next(iter(validator.dataloader))
    warmup_tensor = validator.preprocess(first_batch)["img"]
    for _ in range(warmup):
        recorder.reset()
        model(warmup_tensor)

    samples_ms: list[float] = []
    activation_samples_ms: list[float] = []
    activation_percent_samples: list[float] = []
    activation_calls: list[int] = []
    activation_elements = 0
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
        elapsed_ns = time.perf_counter_ns() - start
        elapsed_ms = elapsed_ns / 1e6
        batch_images = int(tensor.shape[0])
        # With the default batch=1 this is an actual per-image measurement.
        # For larger batches it is normalized batch latency per image.
        samples_ms.append(elapsed_ms / batch_images)
        activation_samples_ms.append(recorder.elapsed_ns / 1e6 / batch_images)
        activation_percent_samples.append(100.0 * recorder.elapsed_ns / elapsed_ns)
        activation_calls.append(recorder.calls)
        activation_elements += recorder.elements
        images_measured += batch_images
        if maximum_images and images_measured >= maximum_images:
            break

    activation_median_ms = float(np.median(activation_samples_ms))
    return {
        "scope": (
            f"{dataset_label}_images_forward_only_excluding_loading_"
            "preprocessing_and_nms"
        ),
        "unit": "milliseconds_per_image",
        "images": images_measured,
        "timed_batches": len(samples_ms),
        "median_ms": float(np.median(samples_ms)),
        "p90_ms": float(np.percentile(samples_ms, 90)),
        "p99_ms": float(np.percentile(samples_ms, 99)),
        "minimum_ms": float(np.min(samples_ms)),
        "maximum_ms": float(np.max(samples_ms)),
        "activation": {
            "calls_per_batch_minimum": min(activation_calls),
            "calls_per_batch_maximum": max(activation_calls),
            "elements_total": activation_elements,
            "mean_ms": float(np.mean(activation_samples_ms)),
            "median_ms": activation_median_ms,
            "p90_ms": float(np.percentile(activation_samples_ms, 90)),
            "p99_ms": float(np.percentile(activation_samples_ms, 99)),
            "minimum_ms": float(np.min(activation_samples_ms)),
            "maximum_ms": float(np.max(activation_samples_ms)),
            "aggregate_wall_clock_percent": float(
                100.0 * np.sum(activation_samples_ms) / np.sum(samples_ms)
            ),
            "median_per_image_percent": float(
                np.median(activation_percent_samples)
            ),
            "p90_per_image_percent": float(
                np.percentile(activation_percent_samples, 90)
            ),
            "p99_per_image_percent": float(
                np.percentile(activation_percent_samples, 99)
            ),
        },
    }


def validate_kitti(
    model: BaseModel,
    data: Path,
    imgsz: int,
    batch: int,
    workers: int,
) -> dict[str, Any]:
    validator = DetectionValidator(
        args={
            "model": "in-memory-activation-model.pt",
            "data": str(data),
            "imgsz": imgsz,
            "batch": batch,
            "device": "cpu",
            "workers": workers,
            "split": "val",
            "rect": True,
            "plots": False,
            "save_json": False,
            "verbose": False,
        }
    )
    validator(model=model)
    metrics = validator.metrics
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision_mean": float(metrics.box.mp),
        "recall_mean": float(metrics.box.mr),
        "validator_speed_ms_per_image": {
            key: float(value) for key, value in validator.speed.items()
        },
    }


def prepare_models(
    base_model: BaseModel,
    cpp_functions: dict[str, Any],
) -> tuple[
    dict[str, BaseModel],
    dict[str, int],
    dict[str, ActivationRecorder],
]:
    # This checkpoint shares one SiLU instance across many parent modules, so
    # count registered references before replacing them with timer wrappers.
    native_count = sum(
        isinstance(child, nn.SiLU)
        for parent in base_model.modules()
        for child in parent._modules.values()
    )
    if native_count == 0:
        raise RuntimeError("The model contains no nn.SiLU modules")

    models = {"native_silu": base_model}
    models.update(
        {name: copy.deepcopy(base_model) for name in cpp_functions}
    )
    models["torch_cpp_lut21"] = copy.deepcopy(base_model)
    recorders = {name: ActivationRecorder() for name in models}

    native_silu = next(
        module for module in base_model.modules() if isinstance(module, nn.SiLU)
    )
    replacements = {}
    replacements["native_silu"] = replace_silu(
        models["native_silu"],
        TimedActivation(
            nn.SiLU(inplace=native_silu.inplace), recorders["native_silu"]
        ),
    )
    for name, function in cpp_functions.items():
        replacements[name] = replace_silu(
            models[name],
            TimedActivation(CppArrayActivation(function), recorders[name]),
        )
    replacements["torch_cpp_lut21"] = replace_silu(
        models["torch_cpp_lut21"],
        TimedActivation(
            TorchCppLUT21SiLU(), recorders["torch_cpp_lut21"]
        ),
    )
    if any(count != native_count for count in replacements.values()):
        raise RuntimeError(
            f"Incomplete SiLU replacement: native={native_count}, {replacements}"
        )
    for model in models.values():
        configure_for_validation(model)
    return models, replacements, recorders


def parse_args() -> argparse.Namespace:
    print("threads:", torch.get_num_threads())
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", type=Path)
    source.add_argument("--git-object", default=DEFAULT_GIT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--dataset-label", default=DEFAULT_DATASET_LABEL)
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
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
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

    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    with tempfile.TemporaryDirectory(prefix="yolo-activation-comparison-") as temporary:
        temporary_path = Path(temporary)
        activation_library_path = temporary_path / f"libactivations{suffix}"
        activation_compile_command = compile_all_activations_library(
            args.cxx, activation_library_path
        )
        activation_library = ctypes.CDLL(str(activation_library_path))
        activation_library.initialize_activations.argtypes = []
        activation_library.initialize_activations.restype = None
        activation_library.initialize_activations()
        cpp_functions: dict[str, Any] = {}
        for name, symbol in CPP_ACTIVATIONS.items():
            function = getattr(activation_library, symbol)
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            function.restype = None
            cpp_functions[name] = function

        cpp_library_path = temporary_path / f"libtorch_lut21{suffix}"
        cpp_compile_command = compile_torch_cpp_extension(
            args.cxx, cpp_library_path, args.verbose_build
        )

        base_model = load_model(args.model, args.git_object)
        models, replacement_counts, activation_recorders = prepare_models(
            base_model, cpp_functions
        )
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
                "replaced_or_native_silu_references": replacement_counts[name]
            }
            if not args.skip_latency:
                # Ultralytics CPU validation may reset this between models.
                torch.set_num_threads(args.threads)
                assert latency_validator is not None
                entry["latency"] = benchmark_kitti_forward(
                    model,
                    latency_validator,
                    activation_recorders[name],
                    args.latency_warmup,
                    args.latency_max_images,
                    args.dataset_label,
                )
                print(
                    "latency: "
                    f"median={entry['latency']['median_ms']:.3f} ms, "
                    f"p90={entry['latency']['p90_ms']:.3f} ms, "
                    f"p99={entry['latency']['p99_ms']:.3f} ms"
                )
                activation = entry["latency"]["activation"]
                print(
                    "activation: "
                    f"median={activation['median_ms']:.3f} ms, "
                    f"p90={activation['p90_ms']:.3f} ms, "
                    f"p99={activation['p99_ms']:.3f} ms, "
                    f"share={activation['aggregate_wall_clock_percent']:.2f}%"
                )
            if not args.skip_validation:
                entry["kitti_validation"] = validate_kitti(
                    model,
                    data_path,
                    args.imgsz,
                    args.batch,
                    args.workers,
                )
                print(
                    f"{args.dataset_label}: "
                    f"mAP50={entry['kitti_validation']['map50']:.6f}, "
                    f"mAP50-95={entry['kitti_validation']['map50_95']:.6f}"
                )
            comparisons[name] = entry

    native = comparisons["native_silu"]
    relative_to_native: dict[str, Any] = {}
    for name in comparisons:
        if name == "native_silu":
            continue
        candidate = comparisons[name]
        delta: dict[str, float] = {}
        if "latency" in native:
            for percentile in ("median_ms", "p90_ms", "p99_ms"):
                native_value = native["latency"][percentile]
                candidate_value = candidate["latency"][percentile]
                delta[f"{percentile}_change_percent"] = (
                    100.0 * (candidate_value - native_value) / native_value
                )
            native_activation = native["latency"]["activation"]
            candidate_activation = candidate["latency"]["activation"]
            delta["activation_median_ms_change_percent"] = (
                100.0
                * (
                    candidate_activation["median_ms"]
                    - native_activation["median_ms"]
                )
                / native_activation["median_ms"]
            )
            delta["activation_median_speedup"] = (
                native_activation["median_ms"]
                / candidate_activation["median_ms"]
            )
        if "kitti_validation" in native:
            for metric in ("map50", "map50_95"):
                delta[f"{metric}_absolute_change"] = (
                    candidate["kitti_validation"][metric]
                    - native["kitti_validation"][metric]
                )
        relative_to_native[name] = delta

    result = {
        "model": model_description,
        "data": str(data_path),
        "dataset_label": args.dataset_label,
        "device": "cpu",
        "torch_version": torch.__version__,
        "threads": args.threads,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "latency_note": (
            f"Forward-only wall clock on preprocessed {args.dataset_label} images. "
            "Image loading, preprocessing and NMS are excluded. Exported C++ "
            "activation kernels are single-threaded, while native SiLU and "
            "the PyTorch C++ LUT may use the configured PyTorch thread count. "
            "Activation time uses inner "
            "wall-clock timers around all activation calls. With batch > 1, "
            "reported latency is batch time divided by batch size."
        ),
        "build": {
            "all_activations_cpp": activation_compile_command,
            "torch_cpp": cpp_compile_command,
        },
        "comparisons": comparisons,
        "relative_to_native_silu": relative_to_native,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved results: {results_path}")


if __name__ == "__main__":
    main()
    print(torch.get_num_threads())
