#!/usr/bin/env python3
"""Export the non-standard Milad YOLO checkpoints with Ultralytics.

The Milad ``.pt`` files contain a serialized ``DetectionModel`` directly rather
than a normal Ultralytics checkpoint dictionary.  Loading them with
``YOLO(path).export(...)`` therefore fails.  This script loads the model object
and passes it to Ultralytics' own ``Exporter``.

Edit the values at the top of ``main()`` and run this file directly from the
IDE.  No command-line arguments are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.engine.exporter import Exporter
from ultralytics.nn.tasks import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "Milad_models"


@dataclass(frozen=True)
class ExportConfig:
    """Values configured in main() for one export run."""

    models: list[Path]
    format: str
    precision: str
    imgsz: int
    data: str | None
    fraction: float
    dynamic: bool
    simplify: bool


def _skip_unsupported_fuse(model: BaseModel, *_args: Any, **_kwargs: Any) -> BaseModel:
    """Keep factorized Conv sequences intact instead of applying Conv2d-only fusion."""
    return model


def discover_models(paths: list[Path]) -> list[Path]:
    if paths:
        models = [p.expanduser().resolve() for p in paths]
    else:
        models = sorted(DEFAULT_MODELS_DIR.glob("**/*.pt"))
    missing = [str(p) for p in models if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Model file(s) not found: {', '.join(missing)}")
    if not models:
        raise FileNotFoundError(f"No .pt models found below {DEFAULT_MODELS_DIR}")
    return models


def precision_overrides(precision: str, data: str | None) -> dict[str, Any]:
    """Support both Ultralytics' legacy half/int8 and newer quantize API."""
    if precision == "int8" and data is None:
        raise ValueError("--data DATASET.yaml is required for representative INT8 calibration")

    if hasattr(DEFAULT_CFG, "quantize"):
        overrides: dict[str, Any] = {
            "quantize": {"fp32": 32, "fp16": 16, "int8": 8}[precision]
        }
    else:
        overrides = {
            "half": precision == "fp16",
            "int8": precision == "int8",
        }
    if data is not None:
        data_path = Path(data).expanduser()
        # Keep package-provided dataset aliases (for example `kitti.yaml`) as
        # names so Ultralytics can find them and run their download recipe.
        overrides["data"] = str(data_path.resolve()) if data_path.exists() else data
    return overrides


def load_raw_model(path: Path) -> BaseModel:
    # These trusted local files serialize the complete DetectionModel, so
    # weights_only=False is required.  A normal Ultralytics checkpoint dict is
    # deliberately rejected to keep this compatibility path explicit.
    model = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(model, BaseModel):
        raise TypeError(
            f"Expected a serialized Ultralytics BaseModel in {path}, got {type(model).__name__}"
        )
    return model


def check_and_repair_onnx(path: Path) -> None:
    """Validate ONNX and repair FP16 Cast ordering from older exporter stacks."""
    import onnx

    graph = onnx.load(path)
    try:
        onnx.checker.check_model(graph)
        return
    except onnx.checker.ValidationError as error:
        if "topologically sorted" not in str(error):
            raise

    # Ultralytics <=8.4.60 simplifies before its CPU FP16 conversion.  Some
    # ONNX/ORT combinations append the new I/O Cast nodes at the end, leaving
    # an otherwise valid graph out of topological order.  Re-slimming sorts it.
    import onnxslim

    repaired = onnxslim.slim(graph, no_shape_infer=True)
    onnx.checker.check_model(repaired)
    onnx.save(repaired, path)


def export_model(path: Path, config: ExportConfig) -> Path:
    if config.format not in {"onnx", "openvino", "torchscript", "coreml", "ncnn"}:
        raise ValueError(f"Unsupported export format: {config.format}")
    if config.precision not in {"fp32", "fp16", "int8"}:
        raise ValueError(f"Unsupported precision: {config.precision}")
    if not 0.0 < config.fraction <= 1.0:
        raise ValueError("fraction must be greater than 0 and at most 1")
    model = load_raw_model(path)
    # Ultralytics adds its own `_int8` suffix after ONNX static quantization.
    # FP16/FP32 exports need an explicit suffix to distinguish their artifacts.
    virtual_stem = (
        path.stem if config.precision == "int8" else f"{path.stem}_{config.precision}"
    )

    # Low-rank factorization replaced some Conv.conv layers with Sequential.
    # Ultralytics' standard fuse() assumes Conv.conv is one Conv2d and crashes.
    # Skipping fusion is export-safe: ONNX retains the existing BatchNorm ops.
    model.fuse = MethodType(_skip_unsupported_fuse, model)

    # Exporter determines its destination from pt_path.  Point it at a virtual
    # precision-suffixed .pt name so the original model is never overwritten.
    model.pt_path = str(path.with_name(f"{virtual_stem}.pt"))

    overrides: dict[str, Any] = {
        "format": config.format,
        "imgsz": config.imgsz,
        "batch": 1,
        "device": "cpu",
        "dynamic": config.dynamic,
        "simplify": config.simplify,
        "fraction": config.fraction,
    }
    overrides.update(precision_overrides(config.precision, config.data))

    exported = Exporter(overrides=overrides)(model=model)
    output = Path(exported).resolve()
    if not output.exists():
        raise RuntimeError(f"Ultralytics reported an export that does not exist: {output}")
    if config.format == "onnx":
        check_and_repair_onnx(output)
    return output


def main() -> None:
    # ------------------------------------------------------------------
    # Export parameters: edit these values, then click Run in the IDE.
    # ------------------------------------------------------------------
    config = ExportConfig(
        models=[
            PROJECT_ROOT
            # / "Milad_models/yolov8n_kitti/final_model_factorized_lwi.pt",
            / "Milad_models/yolov8n_kitti/final_model_factorized_lwi.pt",
        ],
        format="onnx",       # onnx, openvino, torchscript, coreml, or ncnn
        precision="int8",    # fp32, fp16, or int8
        imgsz=640,
        data="kitti.yaml",   # Official Ultralytics dataset; downloads automatically
        fraction=1,        # 25% of KITTI val = about 374 calibration images
        dynamic=False,
        simplify=True,
    )
    # For VOC, replace the model above and set data="VOC.yaml".
    # ------------------------------------------------------------------

    models = discover_models(config.models)
    print(
        f"Exporting {len(models)} model(s) as "
        f"{config.precision.upper()} {config.format.upper()}"
    )
    for source in models:
        output = export_model(source, config)
        before = source.stat().st_size
        after = output.stat().st_size if output.is_file() else sum(
            p.stat().st_size for p in output.rglob("*") if p.is_file()
        )
        reduction = 100.0 * (1.0 - after / before)
        print(
            f"{source.relative_to(PROJECT_ROOT)} -> {output.relative_to(PROJECT_ROOT)} "
            f"({before / 2**20:.2f} MiB -> {after / 2**20:.2f} MiB, "
            f"{reduction:.1f}% smaller)"
        )


if __name__ == "__main__":
    main()
