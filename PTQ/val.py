#!/usr/bin/env python3
"""Validate Milad PyTorch and exported models with Ultralytics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ValidationJob:
    name: str
    model: Path
    data: str


def _skip_unsupported_fuse(model: BaseModel, *_args: Any, **_kwargs: Any) -> BaseModel:
    """Keep low-rank Sequential convolutions intact during validation."""
    return model


def load_model(path: Path) -> BaseModel | str:
    """Load raw Milad .pt objects; let AutoBackend load exported formats."""
    if path.suffix.lower() != ".pt":
        return str(path)

    model = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(model, BaseModel):
        raise TypeError(f"Expected Ultralytics BaseModel, got {type(model).__name__}: {path}")
    model.fuse = MethodType(_skip_unsupported_fuse, model)
    return model


def validate(job: ValidationJob, imgsz: int, batch: int, device: str) -> dict[str, Any]:
    validator = DetectionValidator(
        args={
            "model": str(job.model),
            "data": job.data,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "split": "val",
            "rect": True,
            "plots": False,
            "save_json": False,
            "verbose": False,
        }
    )
    validator(model=load_model(job.model))
    metrics = validator.metrics
    result = {
        "name": job.name,
        "model": str(job.model.relative_to(PROJECT_ROOT)),
        "data": job.data,
        "precision": job.model.stem.rsplit("_", 1)[-1] if job.model.suffix == ".onnx" else "fp32",
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
    # Official dataset YAML files download automatically on first use.
    # ------------------------------------------------------------------
    imgsz = 640
    batch = 1  # Exported ONNX models currently have a static batch size of 1.
    device = "cpu"  # Keeps PyTorch and ONNX comparison on the same device.
    results_file = PROJECT_ROOT / "PTQ/validation_results.json"

    jobs = [
        ValidationJob(
            "KITTI PyTorch FP32",
            PROJECT_ROOT / "Milad_models/yolov8n_kitti/final_model_factorized_lwi_fp32.onnx",
            "kitti.yaml",
        ),
        ValidationJob(
            "KITTI ONNX FP16",
            PROJECT_ROOT / "Milad_models/yolov8n_kitti/final_model_factorized_lwi_fp16.onnx",
            "kitti.yaml",
        ),
        ValidationJob(
            "KITTI ONNX INT8",
            PROJECT_ROOT / "Milad_models/yolov8n_kitti/final_model_factorized_lwi_int8.onnx",
            "kitti.yaml",
        ),
        ValidationJob(
            "VOC PyTorch FP32",
            PROJECT_ROOT / "Milad_models/yolov8n_voc/final_model_factorized_lwi_fp32.onnx",
            "VOC.yaml",
        ),
        ValidationJob(
            "VOC ONNX FP16",
            PROJECT_ROOT / "Milad_models/yolov8n_voc/final_model_factorized_lwi_fp16.onnx",
            "VOC.yaml",
        ),
        ValidationJob(
            "VOC ONNX INT8",
            PROJECT_ROOT / "Milad_models/yolov8n_voc/final_model_factorized_lwi_int8.onnx",
            "VOC.yaml",
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
