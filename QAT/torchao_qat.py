#!/usr/bin/env python3
"""Train the factorized YOLO model with TorchAO PT2E INT8 QAT.

The full YOLO training graph is not exported because Ultralytics computes its
loss from a dictionary with a dynamic number of targets.  Instead, each
Conv-BN-activation block (and each remaining standalone Conv2d) is captured as
an independent PT2E graph.  This keeps the Ultralytics trainer and loss intact
while allowing TorchAO's X86 quantizer to choose the QAT boundaries and fold
Conv-BN patterns correctly.

Run from the project root, for example:
    python -m QAT.torchao_qat --device 0 --batch 32 --workers 8
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import Any, Iterator

import torch
from torch import nn
from torch.fx import GraphModule
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.block import DFL
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import BaseModel
from ultralytics.utils.torch_utils import ModelEMA, unwrap_model

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from QAT.torch_qat import load_raw_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "QAT/runs"
FORMAT = "torchao_pt2e_x86_qat"


class StateOnlyDetectionTrainer(DetectionTrainer):
    """Avoid pickling PT2E graphs; portable state is saved after training."""

    def _refresh_ema_if_observers_resized(self) -> bool:
        """Rebuild EMA after lazy TorchAO observers acquire channel shapes."""
        if self.ema is None:
            return False
        model_state = unwrap_model(self.model).state_dict()
        ema_state = self.ema.ema.state_dict()
        resized = any(
            key not in ema_state or ema_state[key].shape != value.shape
            for key, value in model_state.items()
        )
        if not resized:
            return False

        # TorchAO per-channel observers start with scalar buffers and resize
        # them on their first real batch. Ultralytics creates EMA before that
        # batch, so its copied buffers can otherwise retain the scalar shape.
        updates = self.ema.updates
        self.ema = ModelEMA(self.model, updates=updates)
        print("TorchAO QAT: synchronized EMA after observer initialization")
        return True

    def optimizer_step(self) -> None:
        # Run this before Ultralytics calls ModelEMA.update(), whose foreach
        # interpolation requires every corresponding tensor shape to match.
        self._refresh_ema_if_observers_resized()
        super().optimizer_step()

    def save_model(self) -> None:
        return

    def final_eval(self) -> None:
        # Per-epoch validation already selected the best in-memory state.
        return


class PT2EQATUnit(nn.Module):
    """Pickle/deepcopy-stable shell around one prepared PT2E GraphModule."""

    def __init__(self, graph: GraphModule, source: nn.Module) -> None:
        super().__init__()
        self.graph = graph
        self._torchao_qat_unit = True
        _copy_yolo_routing_attributes(source, self)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.graph(tensor)


def _torchao_api():
    try:
        from torchao.quantization.pt2e import (
            allow_exported_model_train_eval,
            disable_observer,
            move_exported_model_to_eval,
            move_exported_model_to_train,
        )
        from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e
        from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
            X86InductorQuantizer,
            get_default_x86_inductor_quantization_config,
        )
    except ImportError as error:
        raise RuntimeError(
            "TorchAO is required. Install it with: pip install torchao"
        ) from error
    return (
        disable_observer,
        move_exported_model_to_eval,
        move_exported_model_to_train,
        prepare_qat_pt2e,
        X86InductorQuantizer,
        get_default_x86_inductor_quantization_config,
        allow_exported_model_train_eval,
    )


def _first_conv(module: nn.Module) -> nn.Conv2d:
    first = next((m for m in module.modules() if isinstance(m, nn.Conv2d)), None)
    if first is None:
        raise TypeError(f"No Conv2d found in quantization unit {type(module).__name__}")
    return first


def _copy_yolo_routing_attributes(source: nn.Module, target: nn.Module) -> None:
    # Top-level Ultralytics layers carry these attributes and BaseModel uses
    # them to route skip connections and collect intermediate outputs.
    for name in ("i", "f", "type", "np"):
        if hasattr(source, name):
            setattr(target, name, getattr(source, name))


def _prepare_unit(module: nn.Module, index: int) -> tuple[PT2EQATUnit, int]:
    (
        _disable_observer,
        _move_to_eval,
        move_to_train,
        prepare_qat_pt2e,
        X86InductorQuantizer,
        get_qconfig,
        allow_train_eval,
    ) = _torchao_api()

    first_conv = _first_conv(module)
    example = torch.randn(
        2,
        first_conv.in_channels,
        32,
        32,
        device=first_conv.weight.device,
        dtype=first_conv.weight.dtype,
    )
    batch = torch.export.Dim(f"batch_{index}", min=1, max=1024)
    height = torch.export.Dim(f"height_{index}", min=3, max=4096)
    width = torch.export.Dim(f"width_{index}", min=3, max=4096)
    exported = torch.export.export(
        module,
        (example,),
        dynamic_shapes=({0: batch, 2: height, 3: width},),
    )
    graph = exported.module()
    quantizer = X86InductorQuantizer().set_global(get_qconfig(is_qat=True))
    prepared = prepare_qat_pt2e(graph, quantizer)
    # Ultralytics calls train()/eval() recursively on the parent model.
    allow_train_eval(prepared)
    move_to_train(prepared)
    convs = sum(
        node.op == "call_function" and "aten.conv" in str(node.target)
        for node in prepared.graph.nodes
    )
    return PT2EQATUnit(prepared, module), convs


def prepare_model(model: BaseModel) -> tuple[BaseModel, int, int]:
    """Replace quantizable YOLO units with TorchAO-prepared PT2E graphs."""
    model.train()
    units = 0
    convs = 0

    def replace(parent: nn.Module) -> None:
        nonlocal units, convs
        if isinstance(parent, DFL):
            return
        for name, child in list(parent.named_children()):
            # Capturing the complete Ultralytics Conv block lets PT2E place
            # output FakeQuant after BN/SiLU and perform its QAT fusion.
            if isinstance(child, Conv):
                prepared, count = _prepare_unit(child, units)
                setattr(parent, name, prepared)
                units += 1
                convs += count
            elif isinstance(child, nn.Conv2d):
                prepared, count = _prepare_unit(child, units)
                setattr(parent, name, prepared)
                units += 1
                convs += count
            else:
                replace(child)

    replace(model)
    if not units or not convs:
        raise RuntimeError("TorchAO did not prepare any convolution units")
    return model, units, convs


def iter_qat_units(model: nn.Module) -> Iterator[GraphModule]:
    for module in model.modules():
        if isinstance(module, PT2EQATUnit):
            yield module.graph


def set_qat_mode(model: nn.Module, *, training: bool) -> None:
    api = _torchao_api()
    move = api[2] if training else api[1]
    for unit in iter_qat_units(model):
        move(unit)


def freeze_qat(owner: Any, freeze_epoch: int) -> None:
    """Freeze observers and captured BatchNorm behavior near training end."""
    model = owner.model
    if owner.epoch < freeze_epoch:
        set_qat_mode(model, training=True)
        return
    disable_observer = _torchao_api()[0]
    for unit in iter_qat_units(model):
        unit.apply(disable_observer)
    set_qat_mode(model, training=False)
    if owner.epoch == freeze_epoch:
        print(f"TorchAO QAT: froze observers and BN at epoch {owner.epoch + 1}")


def _cpu_state_dict(model: nn.Module) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for key, value in model.state_dict().items()
    }


def _portable_source(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def train(args: argparse.Namespace) -> Path:
    source = args.model.expanduser().resolve()
    if args.device.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {args.device} requested but CUDA is unavailable")
        preparation_device = torch.device(f"cuda:{args.device}")
    elif args.device == "cpu":
        preparation_device = torch.device("cpu")
    else:
        raise ValueError("TorchAO QAT currently supports one CUDA device such as '0', or 'cpu'")
    model, units, convs = prepare_model(load_raw_model(source).to(preparation_device))
    print(f"TorchAO QAT prepared: {units} graph units, {convs} convolution ops")

    # Epoch indices are zero based. Never freeze before epoch 1, otherwise a
    # one-epoch smoke run would disable observers before their first sample.
    freeze_epoch = max(1, args.epochs - max(1, round(args.epochs * 0.1)))
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
            # Exported PT2E GraphModules are not pickleable. Save the portable
            # base-model + state-dict recipe below instead of Ultralytics .pt.
            "save": False,
        }
    )
    trainer.model = model

    best: dict[str, Any] = {"fitness": float("-inf"), "state_dict": None}

    def on_fit_epoch_end(current: DetectionTrainer) -> None:
        fitness = float(current.fitness) if current.fitness is not None else float("-inf")
        if fitness >= best["fitness"]:
            candidate = current.ema.ema if current.ema is not None else current.model
            best["fitness"] = fitness
            best["state_dict"] = _cpu_state_dict(candidate)

    trainer.add_callback(
        "on_train_epoch_start", lambda current: freeze_qat(current, freeze_epoch)
    )
    trainer.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    trainer.train()

    if best["state_dict"] is None:
        candidate = trainer.ema.ema if trainer.ema is not None else trainer.model
        best["state_dict"] = _cpu_state_dict(candidate)

    output = args.output or Path(trainer.save_dir) / "best_torchao_qat.pt"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {output}")
    torch.save(
        {
            "format": FORMAT,
            "format_version": 1,
            "source_model": _portable_source(source),
            "state_dict": best["state_dict"],
            "fitness": best["fitness"],
            "prepared_units": units,
            "quantized_convs": convs,
            "imgsz": args.imgsz,
        },
        output,
    )
    print(f"TorchAO QAT checkpoint saved: {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "original/yolov8n_voc/final_model_factorized_lwi.pt",
    )
    parser.add_argument("--data", default=str(PROJECT_ROOT / "datasets/VOC/VOC.yaml"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--name", default="voc_torchao_qat")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        parser.error("epochs and batch must be positive; workers cannot be negative")
    if args.imgsz < 96 or args.imgsz % 32:
        parser.error("TorchAO QAT imgsz must be a multiple of 32 and at least 96")
    return args


if __name__ == "__main__":
    train(parse_args())
