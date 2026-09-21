#!/usr/bin/env python3
"""Replace YOLO SiLU with a 21-point LUT and validate it on KITTI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from torch import nn
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import BaseModel

from Activation.lut_silu import LUTSiLU, replace_silu


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "Milad_models/yolov8n_kitti/final_model_factorized_lwi.pt"
DEFAULT_DATA = PROJECT_ROOT / "datasets/kitti/kitti.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "Activation/yolov8n_kitti_lut21.pt"
DEFAULT_RESULTS = PROJECT_ROOT / "Activation/yolov8n_kitti_lut21_results.json"


def _skip_fuse(model: BaseModel, *_args: Any, **_kwargs: Any) -> BaseModel:
    """Keep the model's factorized Sequential convolutions intact."""
    return model


def load_raw_model(path: Path) -> BaseModel:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, dict):
        loaded = loaded.get("ema") or loaded.get("model")
    if not isinstance(loaded, BaseModel):
        raise TypeError(f"Expected an Ultralytics BaseModel, got {type(loaded).__name__}")
    return loaded.float().eval()


def validate(model: BaseModel, data: Path, imgsz: int, batch: int, device: str) -> dict[str, Any]:
    validator = DetectionValidator(
        args={
            "model": "in-memory-lut-model.pt",
            "data": str(data),
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
    validator(model=model)
    metrics = validator.metrics
    return {
        "mAP50-95": float(metrics.box.map),
        "mAP50": float(metrics.box.map50),
        "mAP75": float(metrics.box.map75),
        "precision_mean": float(metrics.box.mp),
        "recall_mean": float(metrics.box.mr),
        "speed_ms_per_image": {key: float(value) for key, value in validator.speed.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    data_path = args.data.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    results_path = args.results.expanduser().resolve()

    model = load_raw_model(model_path)
    original_silu_references = sum(
        isinstance(child, nn.SiLU)
        for parent in model.modules()
        for child in parent._modules.values()
    )
    replacement = LUTSiLU(points=21, minimum=-5.0, maximum=5.0)
    replaced = replace_silu(model, replacement)
    remaining = sum(isinstance(module, nn.SiLU) for module in model.modules())
    if replaced != original_silu_references or remaining:
        raise RuntimeError(
            f"SiLU replacement incomplete: found={original_silu_references}, "
            f"replaced={replaced}, remaining_unique={remaining}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, output_path)
    print(f"Replaced {replaced} SiLU references with one shared {replacement}")
    print(f"Saved LUT model: {output_path}")

    # This model contains factorized Sequential convolutions, which the
    # standard Ultralytics fuse routine does not support. Keep this temporary
    # instance override out of the serialized checkpoint.
    model.fuse = MethodType(_skip_fuse, model)
    metrics = validate(model, data_path, args.imgsz, args.batch, args.device)
    result = {
        "model": str(model_path),
        "output_model": str(output_path),
        "data": str(data_path),
        "activation": {
            "type": "piecewise_linear_lut",
            "points": replacement.points,
            "minimum": replacement.minimum,
            "maximum": replacement.maximum,
            "replaced_silu_references": replaced,
        },
        **metrics,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved validation results: {results_path}")


if __name__ == "__main__":
    main()
