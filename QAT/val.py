#!/usr/bin/env python3
"""Validate QAT checkpoints with frozen ranges and fake quantization enabled."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Checkpoints reference QAT.torch_qat, including when this file is run directly.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from QAT.torch_qat import MinMaxFakeQuant
from QAT.int8 import Int8Conv2d


@dataclass(frozen=True)
class ValidationJob:
    name: str
    model: Path
    data: str


def _skip_unsupported_fuse(model: BaseModel, *_args: Any, **_kwargs: Any) -> BaseModel:
    """Keep low-rank Sequential convolutions intact during validation."""
    return model


def load_model(path: Path) -> BaseModel | str:
    """Load checkpoint EMA/model or a raw model, preserving QAT layers."""
    if path.suffix.lower() != ".pt":
        return str(path)

    model = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(model, dict):
        ema = model.get("ema")
        model = ema if ema is not None else model.get("model")
    if not isinstance(model, BaseModel):
        raise TypeError(f"Expected Ultralytics BaseModel, got {type(model).__name__}: {path}")
    if hasattr(model, "int8_backend"):
        if model.int8_backend not in torch.backends.quantized.supported_engines:
            raise RuntimeError(f"Unavailable INT8 backend: {model.int8_backend}")
        torch.backends.quantized.engine = model.int8_backend
    model = model.float().eval()
    for module in model.modules():
        if isinstance(module, MinMaxFakeQuant):
            module.observer_enabled = False
            module.fake_quant_enabled = True
    model.fuse = MethodType(_skip_unsupported_fuse, model)
    return model


def validate(job: ValidationJob, imgsz: int, batch: int, device: str) -> dict[str, Any]:
    loaded = load_model(job.model)
    real_int8 = isinstance(loaded, BaseModel) and any(
        isinstance(module, Int8Conv2d) for module in loaded.modules()
    )
    if real_int8 and device != "cpu":
        raise ValueError("Real INT8 validation requires device='cpu'")
    validator = DetectionValidator(
        args={
            "model": str(job.model),
            "data": job.data,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "split": "val",
            "rect": True,
            "workers": 0,
            "project": str(PROJECT_ROOT / "QAT/runs"),
            "name": job.name,
            "half": False,
            "plots": False,
            "save_json": False,
            "verbose": False,
        }
    )
    validator(model=loaded)
    metrics = validator.metrics
    result = {
        "name": job.name,
        "model": str(job.model.relative_to(PROJECT_ROOT)),
        "data": job.data,
        "precision": job.model.stem.rsplit("_", 1)[-1] if job.model.suffix == ".onnx" else ("int8_convs_float_remainder" if real_int8 else "qat_fake_int8"),
        "mAP50-95": float(metrics.box.map),
        "mAP50": float(metrics.box.map50),
        "mAP75": float(metrics.box.map75),
        "precision_mean": float(metrics.box.mp),
        "recall_mean": float(metrics.box.mr),
        "speed_ms_per_image": {key: float(value) for key, value in validator.speed.items()},
    }
    return result


def main() -> None:
    # ------------------------------------------------------------------
    # Validation parameters: edit these values, then click Run in the IDE.
    # ------------------------------------------------------------------
    imgsz = 640
    batch = 1
    device = "cpu"
    results_file = PROJECT_ROOT / "QAT/validation_results.json"

    jobs = [
        ValidationJob(
            "KITTI QAT best.pt",
            PROJECT_ROOT / "QAT/runs/kitti_qat_10e-3/weights/best_real_int8.pt",
            str(PROJECT_ROOT / "datasets/kitti/kitti.yaml"),
        ),
    ]
    # ------------------------------------------------------------------

    results: list[dict[str, Any]] = []
    for job in jobs:
        if not job.model.is_file():
            print(f"SKIP: {job.name} (model not found: {job.model})")
            continue
        print(f"\nVALIDATING: {job.name}")
        result = validate(job, imgsz=imgsz, batch=batch, device=device)
        results.append(result)
        print(
            f"RESULT: mAP50-95={result['mAP50-95']:.6f}, "
            f"mAP50={result['mAP50']:.6f}, mAP75={result['mAP75']:.6f}, "
            f"P={result['precision_mean']:.6f}, R={result['recall_mean']:.6f}"
        )

    results_file.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(results)} result(s) to {results_file}")


if __name__ == "__main__":
    main()
    #
    #
