#!/usr/bin/env python3
"""Fine-tune the factorized YOLO models with PyTorch eager-mode QAT.

The input files contain serialized Ultralytics ``DetectionModel`` objects
rather than normal checkpoint dictionaries.  This script prepares each raw
model with fake-quantization modules, then hands it directly to an Ultralytics
``DetectionTrainer``.
"""

from __future__ import annotations

import copy
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "QAT/runs"


@dataclass(frozen=True)
class QATJob:
    name: str
    model: Path
    data: str


class MinMaxFakeQuant(nn.Module):
    """MPS-compatible 8-bit fake quantizer with a moving min/max observer."""

    def __init__(self, *, symmetric: bool, momentum: float = 0.01) -> None:
        super().__init__()
        self.symmetric = symmetric
        self.momentum = momentum
        self.observer_enabled = True
        self.fake_quant_enabled = True
        self.initialized = False
        self.register_buffer("min_val", torch.tensor(0.0))
        self.register_buffer("max_val", torch.tensor(0.0))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.observer_enabled:
            with torch.no_grad():
                current_min = tensor.detach().amin().float()
                current_max = tensor.detach().amax().float()
                if self.initialized:
                    self.min_val.lerp_(current_min, self.momentum)
                    self.max_val.lerp_(current_max, self.momentum)
                else:
                    self.min_val.copy_(current_min)
                    self.max_val.copy_(current_max)
                    self.initialized = True

        if not self.fake_quant_enabled or not self.initialized:
            return tensor

        epsilon = torch.finfo(torch.float32).eps
        if self.symmetric:
            # Symmetric signed INT8 for convolution weights.
            magnitude = torch.maximum(self.min_val.abs(), self.max_val.abs())
            scale = magnitude.div(127.0).clamp_min(epsilon)
            quantized = tensor.div(scale).round().clamp(-127, 127).mul(scale)
        else:
            # Signed asymmetric INT8 for activations. Deployment uses the
            # equivalent uint8 grid required by PyTorch quantized Conv2d.
            qmin, qmax = -128, 127
            minimum = torch.minimum(self.min_val, torch.zeros_like(self.min_val))
            maximum = torch.maximum(self.max_val, torch.zeros_like(self.max_val))
            scale = maximum.sub(minimum).div(qmax - qmin).clamp_min(epsilon)
            zero_point = (qmin - minimum.div(scale)).round().clamp(qmin, qmax)
            quantized = (
                tensor.div(scale)
                .add(zero_point)
                .round()
                .clamp(qmin, qmax)
                .sub(zero_point)
                .mul(scale)
            )

        # Straight-through estimator: quantized forward, identity gradient.
        return tensor + (quantized - tensor).detach()


class QATConv2d(nn.Conv2d):
    """Conv2d that simulates the exact activation/weight INT8 boundaries."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.input_fake_quant = MinMaxFakeQuant(symmetric=False)
        self.weight_fake_quant = MinMaxFakeQuant(symmetric=True)
        self.output_fake_quant = MinMaxFakeQuant(symmetric=False)

    @classmethod
    def from_conv(cls, conv: nn.Conv2d) -> "QATConv2d":
        qat_conv = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode,
        )
        qat_conv.weight = conv.weight
        qat_conv.bias = conv.bias
        return qat_conv

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # Old checkpoints only contain weight_fake_quant.  Keeping this
        # fallback makes them loadable, but new training creates all 3 ranges.
        input_fake_quant = getattr(self, "input_fake_quant", None)
        output_fake_quant = getattr(self, "output_fake_quant", None)
        if input_fake_quant is not None:
            tensor = input_fake_quant(tensor)
        weight = self.weight_fake_quant(self.weight)
        output = self._conv_forward(tensor, weight, self.bias)
        return output_fake_quant(output) if output_fake_quant is not None else output


def _replace_convs(module: nn.Module) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, QATConv2d):
            continue
        if isinstance(child, nn.Conv2d):
            setattr(module, name, QATConv2d.from_conv(child))
            count += 1
        else:
            count += _replace_convs(child)
    return count


def load_raw_model(path: Path) -> BaseModel:
    """Load the non-standard raw DetectionModel used by this project."""
    model = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(model, BaseModel):
        raise TypeError(
            f"Expected a serialized Ultralytics BaseModel, got "
            f"{type(model).__name__}: {path}"
        )
    model = model.float()
    # The supplied raw models were serialized with frozen parameters.
    # QAT is a fine-tuning stage, so unfreeze floating-point weights here;
    # Ultralytics will re-freeze its fixed DFL projection layer as usual.
    for parameter in model.parameters():
        if parameter.is_floating_point():
            parameter.requires_grad_(True)
    return model


def prepare_model(model: BaseModel) -> tuple[BaseModel, int, int]:
    """Insert fake quantizers at the same boundaries as deployed INT8 Conv2d."""
    model.train()
    qat_modules = _replace_convs(model)

    fake_quantizers = sum(
        isinstance(module, MinMaxFakeQuant) for module in model.modules()
    )
    expected_fake_quantizers = qat_modules * 3
    if not qat_modules or fake_quantizers != expected_fake_quantizers:
        raise RuntimeError(
            f"Expected {expected_fake_quantizers} fake quantizers for "
            f"{qat_modules} convolutions, found {fake_quantizers}"
        )
    return model, qat_modules, fake_quantizers


def freeze_qat_ranges(trainer: DetectionTrainer, freeze_epoch: int) -> None:
    """Stop observer and BatchNorm updates near the end of QAT."""
    if trainer.epoch < freeze_epoch:
        return
    for module in trainer.model.modules():
        if isinstance(module, MinMaxFakeQuant):
            module.observer_enabled = False
    for module in trainer.model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    if trainer.epoch == freeze_epoch:
        print(
            f"QAT: froze fake-quant observers and BatchNorm statistics at "
            f"epoch {trainer.epoch + 1}"
        )


def sync_ema_qat_ranges(trainer: DetectionTrainer) -> None:
    """Copy observer state to EMA before validation and checkpoint saving."""
    if trainer.ema is None:
        return
    live_modules = dict(trainer.model.named_modules())
    for name, ema_module in trainer.ema.ema.named_modules():
        if not isinstance(ema_module, MinMaxFakeQuant):
            continue
        live_module = live_modules[name]
        ema_module.min_val.copy_(live_module.min_val)
        ema_module.max_val.copy_(live_module.max_val)
        ema_module.initialized = live_module.initialized
        ema_module.observer_enabled = live_module.observer_enabled
        ema_module.fake_quant_enabled = live_module.fake_quant_enabled


def save_raw_qat_model(trainer: DetectionTrainer, job: QATJob) -> Path:
    """Save a directly loadable FP32 model with calibrated fake quantizers."""
    checkpoint_path = trainer.best if trainer.best.is_file() else trainer.last
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trained = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(trained, BaseModel):
        raise TypeError(f"No QAT BaseModel found in {checkpoint_path}")
    trained = copy.deepcopy(trained).float().cpu().eval()
    for module in trained.modules():
        if isinstance(module, MinMaxFakeQuant):
            module.observer_enabled = False
    output = trainer.wdir / "best_qat_model.pt"
    torch.save(trained, output)
    print(f"QAT model saved: {output.relative_to(PROJECT_ROOT)}")
    return output


def train_job(
    job: QATJob,
    *,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
) -> Path:
    """Prepare and train one QAT model."""
    if not job.model.is_file():
        raise FileNotFoundError(job.model)

    model, qat_modules, fake_quantizers = prepare_model(load_raw_model(job.model))
    print(
        f"\nQAT PREPARED: {job.name} "
        f"({qat_modules} QAT modules, {fake_quantizers} fake quantizers)"
    )

    freeze_epoch = max(0, epochs - max(1, int(epochs * 0.1)))
    trainer = DetectionTrainer(
        overrides={
            "model": str(job.model),
            "data": job.data,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "workers": 0,
            "project": str(RUNS_ROOT),
            "name": job.name,
            "exist_ok": False,
            "pretrained": True,
            "optimizer": "AdamW",
            "lr0": 1e-4,
            "lrf": 0.1,
            "weight_decay": 5e-4,
            "warmup_epochs": 1.0,
            "cos_lr": True,
            "close_mosaic": 10,
            "amp": False,
            "plots": True,
            "val": True,
            "save": True,
        }
    )
    # setup_model() accepts a ready nn.Module and therefore bypasses the
    # standard checkpoint loader, which cannot read this project's raw files.
    trainer.model = model
    trainer.add_callback(
        "on_train_epoch_start",
        lambda current_trainer: freeze_qat_ranges(current_trainer, freeze_epoch),
    )
    trainer.add_callback("on_train_epoch_end", sync_ema_qat_ranges)
    trainer.train()
    return save_raw_qat_model(trainer, job)


def main() -> None:
    # ------------------------------------------------------------------
    # QAT parameters: edit these values, then click Run in the IDE.
    # Jobs run sequentially and write below QAT/runs/.
    # ------------------------------------------------------------------
    epochs = 50
    imgsz = 640
    batch = 8
    if torch.cuda.is_available():
        device = "0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    jobs = [
        # QATJob(
        #     "kitti_qat_10e",
        #     PROJECT_ROOT
        #     / "Milad_models/yolov8n_kitti/final_model_factorized_lwi.pt",
        #     str(PROJECT_ROOT / "datasets/kitti/kitti.yaml"),
        # ),
        QATJob(
            "voc_qat_10e",
            PROJECT_ROOT
            / "Milad_models/yolov8n_voc/final_model_factorized_lwi.pt",
            "VOC.yaml",
        ),
    ]
    # ------------------------------------------------------------------

    print(
        f"Starting {len(jobs)} QAT job(s): epochs={epochs}, imgsz={imgsz}, "
        f"batch={batch}, device={device}"
    )
    outputs = [
        train_job(
            job,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
        )
        for job in jobs
    ]
    print("\nQAT complete:")
    for output in outputs:
        print(f"  {output.relative_to(PROJECT_ROOT)}")


def register_pickle_module() -> None:
    """Give custom QAT layers an importable name when run as a script."""
    if __name__ != "__main__":
        return
    canonical_name = "QAT.torch_qat"
    package = importlib.import_module("QAT")
    current_module = sys.modules[__name__]
    sys.modules[canonical_name] = current_module
    setattr(package, "torch_qat", current_module)
    MinMaxFakeQuant.__module__ = canonical_name
    QATConv2d.__module__ = canonical_name


if __name__ == "__main__":
    register_pickle_module()
    main()
