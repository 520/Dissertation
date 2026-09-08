#!/usr/bin/env python3
"""Calibrate and fine-tune factorized YOLO with NVIDIA ModelOpt INT8 QAT.

ModelOpt inserts backend-aware input and per-channel weight quantizers.  The
calibration pass fixes their ranges, after which Ultralytics fine-tunes the
floating-point weights through fake quantization.  The resulting checkpoint is
restored by ``QAT.modelopt_export`` and exported as an explicit Q/DQ ONNX graph.

Example on an NVIDIA GPU:
    python -m QAT.modelopt_qat --device 0 --batch 32 --workers 8
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
from typing import Iterator

import cv2
import torch
from torch import nn
from ultralytics.data.augment import LetterBox
from ultralytics.models.yolo.detect import DetectionTrainer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from QAT.torch_qat import load_raw_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "QAT/runs"
FORMAT = "nvidia_modelopt_int8_qat"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class StateOnlyDetectionTrainer(DetectionTrainer):
    """Save ModelOpt-native checkpoints without pickling the whole model."""

    def _save_modelopt_checkpoint(self, model: nn.Module, path: Path) -> None:
        mto, _mtq = _modelopt_api()
        path.parent.mkdir(parents=True, exist_ok=True)
        mto.save(model, path)
        fitness = float(self.fitness) if self.fitness is not None else float("-inf")
        metadata = {
            **self.modelopt_metadata,
            "checkpoint": str(path.resolve()),
            "epoch": self.epoch,
            "fitness": fitness,
            "best_fitness": (
                float(self.best_fitness) if self.best_fitness is not None else fitness
            ),
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")

    def save_model(self) -> bool:
        candidate = self.ema.ema if self.ema is not None else self.model
        self._save_modelopt_checkpoint(candidate, self.last)
        if self.best_fitness == self.fitness:
            self._save_modelopt_checkpoint(candidate, self.best)
        if self.save_period > 0 and self.epoch % self.save_period == 0:
            self._save_modelopt_checkpoint(candidate, self.wdir / f"epoch{self.epoch}.pt")
        return True

    def final_eval(self) -> None:
        # Per-epoch validation already selected the best in-memory state.
        return


def _modelopt_api():
    try:
        import modelopt.torch.opt as mto
        import modelopt.torch.quantization as mtq
    except ImportError as error:
        raise RuntimeError(
            "NVIDIA ModelOpt is required. Install it with: "
            "pip install 'nvidia-modelopt[torch,onnx]'"
        ) from error
    return mto, mtq


def image_tensor(path: Path, size: int) -> torch.Tensor:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read calibration image: {path}")
    image = LetterBox(new_shape=(size, size), auto=False)(image=image)
    rgb = image[:, :, ::-1].transpose(2, 0, 1).copy()
    return torch.from_numpy(rgb).float().div_(255.0)


def calibration_batches(
    paths: list[Path], size: int, batch_size: int
) -> Iterator[torch.Tensor]:
    for start in range(0, len(paths), batch_size):
        yield torch.stack([image_tensor(path, size) for path in paths[start:start + batch_size]])


def select_calibration_images(roots: list[Path], samples: int) -> list[Path]:
    paths = sorted({
        path.resolve()
        for root in roots
        for path in root.expanduser().rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    })
    if not paths:
        joined = ", ".join(str(root) for root in roots)
        raise FileNotFoundError(f"No calibration images below: {joined}")
    count = min(samples, len(paths))
    indices = torch.linspace(0, len(paths) - 1, count).long().tolist()
    return [paths[index] for index in indices]


def resolve_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value.isdigit() and torch.cuda.is_available():
        return torch.device(f"cuda:{value}")
    raise ValueError(f"ModelOpt QAT requires one available CUDA device, got {value!r}")


def prepare_modelopt_model(
    model: nn.Module,
    paths: list[Path],
    *,
    imgsz: int,
    calibration_batch: int,
    device: torch.device,
) -> tuple[nn.Module, int]:
    _mto, mtq = _modelopt_api()
    model = model.float().eval().to(device)
    config = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    # DFL is a fixed box-decoding projection and does not run in the normal
    # training forward path. Keep all of its quantizers disabled.
    config["quant_cfg"].append({"quantizer_name": "*dfl*", "enable": False})

    def forward_loop(current: nn.Module) -> None:
        with torch.inference_mode():
            for batch in calibration_batches(paths, imgsz, calibration_batch):
                current(batch.to(device, non_blocking=True))

    quantized = mtq.quantize(model, config, forward_loop)
    quantized.train()
    count = sum(
        hasattr(module, "input_quantizer")
        and hasattr(module, "weight_quantizer")
        and type(module).__name__.lower().endswith("conv2d")
        and module.input_quantizer.is_enabled
        and module.weight_quantizer.is_enabled
        for module in quantized.modules()
    )
    if count == 0:
        raise RuntimeError("ModelOpt inserted no quantized Conv2d modules")
    return quantized, count


def _portable_source(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def train(args: argparse.Namespace) -> Path:
    _modelopt_api()
    source = args.model.expanduser().resolve()
    paths = select_calibration_images(args.images, args.calibration_samples)
    device = resolve_device(args.device)
    model, quantized_convs = prepare_modelopt_model(
        load_raw_model(source),
        paths,
        imgsz=args.imgsz,
        calibration_batch=args.calibration_batch,
        device=device,
    )
    print(
        f"ModelOpt QAT prepared: {quantized_convs} Conv2d modules, "
        f"{len(paths)} calibration images"
    )

    trainer = StateOnlyDetectionTrainer(
        overrides={
            "model": str(source),
            "data": args.data,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "project": str(RUNS_ROOT),
            "name": args.name,
            "exist_ok": args.exist_ok,
            "pretrained": True,
            "optimizer": "AdamW",
            "lr0": args.lr0,
            "lrf": 0.1,
            "weight_decay": 5e-4,
            "warmup_epochs": 1.0,
            "cos_lr": True,
            "close_mosaic": min(10, args.epochs),
            "amp": False,
            "plots": False,
            "val": True,
            "save": True,
        }
    )
    trainer.model = model
    trainer.modelopt_metadata = {
        "format": FORMAT,
        "format_version": 1,
        "source_model": _portable_source(source),
        "quantized_convs": quantized_convs,
        "calibration_images": len(paths),
        "calibration_sources": [
            str(path.expanduser().resolve()) for path in args.images
        ],
        "imgsz": args.imgsz,
    }
    trainer.train()

    best_path = Path(trainer.best).resolve()
    if not best_path.is_file():
        raise RuntimeError(f"Training finished without a best checkpoint: {best_path}")
    output = args.output.expanduser().resolve() if args.output else best_path
    if output != best_path:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite: {output}")
        shutil.copy2(best_path, output)
        metadata = json.loads(best_path.with_suffix(".json").read_text())
        metadata["checkpoint"] = str(output)
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"ModelOpt QAT checkpoint saved: {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "original/yolov8n_voc/final_model_factorized_lwi.pt",
    )
    parser.add_argument("--data", default=str(PROJECT_ROOT / "datasets/VOC/VOC.yaml"))
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        default=[
            PROJECT_ROOT / "datasets/VOC/images/train2007",
            PROJECT_ROOT / "datasets/VOC/images/val2007",
            PROJECT_ROOT / "datasets/VOC/images/train2012",
            PROJECT_ROOT / "datasets/VOC/images/val2012",
        ],
        help="One or more calibration-image directories (training data only)",
    )
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--calibration-batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--name", default="voc_modelopt_qat")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    positive = (args.calibration_samples, args.calibration_batch, args.epochs, args.batch)
    if min(positive) < 1 or args.workers < 0:
        parser.error("sample, batch and epoch values must be positive; workers cannot be negative")
    if args.imgsz < 32 or args.imgsz % 32:
        parser.error("imgsz must be a positive multiple of 32")
    return args


if __name__ == "__main__":
    train(parse_args())
