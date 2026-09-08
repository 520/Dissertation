#!/usr/bin/env python3
"""Convert a TorchAO QAT recipe and run it with X86 Inductor INT8 kernels.

PT2E conversion produces quantize/dequantize operator graphs.  ``torch.compile``
then lowers those patterns to optimized CPU kernels.  Compiled Inductor modules
are process- and machine-specific, so the saved artifact is a portable recipe;
``load_int8_model()`` reconstructs and compiles it on the deployment CPU.

Run from the project root:
    python -m QAT.torchao_int8 --checkpoint QAT/runs/voc_torchao_qat/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from QAT.torch_qat import load_raw_model
from QAT.torchao_qat import FORMAT as QAT_FORMAT
from QAT.torchao_qat import PROJECT_ROOT, PT2EQATUnit, prepare_model, set_qat_mode


FORMAT = "torchao_pt2e_x86_int8_recipe"


def _resolve_source(checkpoint: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    raw = checkpoint.get("source_model")
    if not raw:
        raise ValueError("Checkpoint does not record source_model; pass --base-model")
    source = Path(raw).expanduser()
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    if not source.is_file():
        raise FileNotFoundError(
            f"Base model not found: {source}. Pass its location with --base-model."
        )
    return source.resolve()


def restore_qat_model(
    checkpoint_path: Path, base_model: Path | None = None
) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") not in {QAT_FORMAT, FORMAT}:
        raise TypeError(f"Not a TorchAO QAT/INT8 recipe: {checkpoint_path}")
    source = _resolve_source(payload, base_model)
    model, units, convs = prepare_model(load_raw_model(source))
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"TorchAO state does not match base model; missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    if units != payload.get("prepared_units") or convs != payload.get("quantized_convs"):
        raise RuntimeError(
            f"Reconstructed graph differs from training: units={units}, convs={convs}"
        )
    return model, payload


def convert_pt2e_units(model: nn.Module) -> tuple[nn.Module, int, int]:
    try:
        from torchao.quantization.pt2e import allow_exported_model_train_eval
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e
    except ImportError as error:
        raise RuntimeError("Install TorchAO with: pip install torchao") from error

    set_qat_mode(model, training=False)
    converted_count = 0
    qdq_count = 0

    def replace(parent: nn.Module) -> None:
        nonlocal converted_count, qdq_count
        for name, child in list(parent.named_children()):
            if isinstance(child, PT2EQATUnit):
                converted = convert_pt2e(child.graph)
                # convert_pt2e returns a new GraphModule and drops the
                # train/eval compatibility override added during preparation.
                allow_exported_model_train_eval(converted)
                converted._torchao_int8_unit = True
                qdq_count += sum(
                    node.op == "call_function"
                    and (
                        "quantized_decomposed.quantize" in str(node.target)
                        or "quantized_decomposed.dequantize" in str(node.target)
                    )
                    for node in converted.graph.nodes
                )
                child.graph = converted
                child._torchao_qat_unit = False
                child._torchao_int8_unit = True
                converted_count += 1
            else:
                replace(child)

    replace(model)
    if converted_count == 0 or qdq_count == 0:
        raise RuntimeError("PT2E conversion produced no Q/DQ convolution graphs")
    return model.eval(), converted_count, qdq_count


def load_int8_model(
    checkpoint_path: str | Path,
    *,
    base_model: str | Path | None = None,
    compile_model: bool = True,
) -> nn.Module:
    """Restore, convert and optionally compile a portable TorchAO recipe."""
    if compile_model and platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("TorchAO X86 INT8 compilation requires an x86_64 CPU")
    model, _payload = restore_qat_model(
        Path(checkpoint_path), Path(base_model) if base_model is not None else None
    )
    model, _units, _qdq = convert_pt2e_units(model)
    return compile_int8_units(model) if compile_model else model


def compile_int8_units(model: nn.Module) -> nn.Module:
    """Compile each converted graph while leaving YOLO routing in eager mode."""
    count = 0
    for module in model.modules():
        if isinstance(module, PT2EQATUnit) and getattr(module, "_torchao_int8_unit", False):
            module.graph = torch.compile(module.graph, dynamic=False)
            count += 1
    if count == 0:
        raise RuntimeError("No converted PT2E units were available for compilation")
    return model


def _finite(output: Any) -> bool:
    if isinstance(output, torch.Tensor):
        return bool(torch.isfinite(output).all())
    if isinstance(output, dict):
        return all(_finite(value) for value in output.values())
    if isinstance(output, (tuple, list)):
        return all(_finite(value) for value in output)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.imgsz < 96 or args.imgsz % 32 or min(args.batch, args.threads, args.runs) < 1:
        parser.error(
            "imgsz must be a multiple of 32 and at least 96; "
            "batch, threads and runs must be positive"
        )

    compiled = not args.no_compile
    if compiled and platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError(
            "TorchAO X86InductorQuantizer requires an x86_64 CPU. "
            "Use --no-compile only to inspect the reference Q/DQ graph on this machine."
        )
    torch.set_num_threads(args.threads)
    model, payload = restore_qat_model(args.checkpoint, args.base_model)
    model, units, qdq = convert_pt2e_units(model)
    runtime = compile_int8_units(model) if compiled else model
    sample = torch.rand(args.batch, 3, args.imgsz, args.imgsz)

    with torch.inference_mode():
        for _ in range(args.warmup):
            prediction = runtime(sample)
        started = time.perf_counter()
        for _ in range(args.runs):
            prediction = runtime(sample)
        elapsed = time.perf_counter() - started
    if not _finite(prediction):
        raise RuntimeError("TorchAO INT8 runtime produced non-finite output")

    output = args.output or args.checkpoint.with_name(
        f"{args.checkpoint.stem}_cpu_int8.pt"
    )
    output = output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    recipe = dict(payload)
    recipe.update({"format": FORMAT, "converted_units": units, "qdq_ops": qdq})
    torch.save(recipe, output)

    report = {
        "artifact": str(output),
        "precision": "torchao_pt2e_x86_int8" if compiled else "pt2e_qdq_reference",
        "machine": platform.machine(),
        "threads": args.threads,
        "compiled": compiled,
        "converted_units": units,
        "qdq_ops": qdq,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "latency_ms_per_image": elapsed * 1000 / (args.runs * args.batch),
        "finite_output": True,
        "reload": "Use QAT.torchao_int8.load_int8_model(); Inductor recompiles for this CPU.",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
