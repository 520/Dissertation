#!/usr/bin/env python3
"""Restore ModelOpt QAT, export explicit Q/DQ ONNX, and build TensorRT.

Build the TensorRT engine on the same NVIDIA GPU family used for deployment.
ONNX export can run on CPU; ``--build-engine`` requires CUDA and TensorRT.

Example on A100:
    python -m QAT.modelopt_export \
        --checkpoint path/to/best_modelopt_qat.pt --build-engine
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import MethodType
from typing import Any

import torch
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.engine.exporter import Exporter
from ultralytics.nn.tasks import BaseModel

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from QAT.modelopt_qat import FORMAT, PROJECT_ROOT, _modelopt_api
from QAT.torch_qat import load_raw_model


def _skip_fuse(model: BaseModel, *_args: Any, **_kwargs: Any) -> BaseModel:
    # ModelOpt Q/DQ placement was calibrated before export. Ultralytics' eager
    # fusion neither understands ModelOpt QuantConv modules nor low-rank
    # Sequential convolutions, so preserve the calibrated graph here.
    return model


def _metadata_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".json")


def resolve_base_model(checkpoint: Path, override: Path | None) -> Path:
    if override is not None:
        source = override.expanduser().resolve()
    else:
        metadata_path = _metadata_path(checkpoint)
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing {metadata_path}; pass the original model with --base-model"
            )
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("format") != FORMAT:
            raise TypeError(f"Unexpected ModelOpt metadata format in {metadata_path}")
        source = Path(metadata["source_model"]).expanduser()
        if not source.is_absolute():
            source = PROJECT_ROOT / source
        source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Base model not found: {source}")
    return source


def restore_model(
    checkpoint: Path, base_model: Path | None = None
) -> tuple[BaseModel, Path]:
    mto, _mtq = _modelopt_api()
    source = resolve_base_model(checkpoint, base_model)
    restored = mto.restore(
        load_raw_model(source), checkpoint.expanduser().resolve(), map_location="cpu"
    )
    if not isinstance(restored, BaseModel):
        raise TypeError(f"ModelOpt restored {type(restored).__name__}, expected BaseModel")
    restored = restored.float().cpu().eval()
    # Raw serialized DetectionModel files predate newer Ultralytics exporter
    # metadata. The task is unambiguous for this script.
    if not hasattr(restored, "task"):
        restored.task = "detect"
    restored.fuse = MethodType(_skip_fuse, restored)
    return restored, source


def export_qdq_onnx(
    model: BaseModel,
    output: Path,
    *,
    imgsz: int,
    batch: int,
    opset: int,
    dynamic: bool,
    overwrite: bool,
) -> dict[str, int]:
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("ONNX is required. Install it with: pip install onnx") from error

    output = output.expanduser().resolve()
    if output.suffix.lower() != ".onnx":
        raise ValueError("--output must end in .onnx")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Exporter derives the output path from pt_path.
    model.pt_path = str(output.with_suffix(".pt"))
    overrides: dict[str, Any] = {
        "format": "onnx",
        "imgsz": imgsz,
        "batch": batch,
        "device": "cpu",
        "dynamic": dynamic,
        "simplify": False,
        "opset": opset,
        "half": False,
        "int8": False,
        "nms": False,
    }
    if hasattr(DEFAULT_CFG, "quantize"):
        overrides["quantize"] = 32
    exported = Path(Exporter(overrides=overrides)(model=model)).resolve()
    if exported != output:
        if output.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {output}")
        exported.replace(output)

    graph = onnx.load(output)
    onnx.checker.check_model(graph)
    counts = {
        "QuantizeLinear": sum(node.op_type == "QuantizeLinear" for node in graph.graph.node),
        "DequantizeLinear": sum(node.op_type == "DequantizeLinear" for node in graph.graph.node),
        "Conv": sum(node.op_type == "Conv" for node in graph.graph.node),
    }
    if counts["QuantizeLinear"] == 0 or counts["DequantizeLinear"] == 0:
        raise RuntimeError("Exported ONNX contains no explicit Q/DQ nodes")
    return counts


def build_tensorrt_engine(onnx_path: Path, engine_path: Path, workspace_gib: float) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT engine building requires an NVIDIA CUDA GPU")
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError(
            "TensorRT Python bindings are required to build the engine"
        ) from error

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    strongly_typed = hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED")
    if strongly_typed:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parsing failed:\n{errors}")

    config = builder.create_builder_config()
    workspace_bytes = int(workspace_gib * (1 << 30))
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes
    # Strongly typed explicit-Q/DQ networks derive precision from the graph;
    # older TensorRT releases require the INT8 builder flag.
    if not strongly_typed and hasattr(trt.BuilderFlag, "INT8"):
        config.set_flag(trt.BuilderFlag.INT8)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build a serialized engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--engine-output", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--build-engine", action="store_true")
    parser.add_argument("--workspace", type=float, default=4.0, help="TensorRT workspace GiB")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.imgsz < 32 or args.imgsz % 32 or args.batch < 1 or args.workspace <= 0:
        parser.error("imgsz must be a multiple of 32; batch and workspace must be positive")
    if args.dynamic and args.build_engine:
        parser.error("Dynamic TensorRT profiles are not configured; omit --dynamic when building")

    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output or checkpoint.with_name(f"{checkpoint.stem}_qdq.onnx")
    model, source = restore_model(checkpoint, args.base_model)
    counts = export_qdq_onnx(
        model,
        output,
        imgsz=args.imgsz,
        batch=args.batch,
        opset=args.opset,
        dynamic=args.dynamic,
        overwrite=args.overwrite,
    )
    output = output.expanduser().resolve()

    engine = None
    if args.build_engine:
        engine = (args.engine_output or output.with_suffix(".engine")).expanduser().resolve()
        if engine.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite: {engine}")
        build_tensorrt_engine(output, engine, args.workspace)

    report = {
        "checkpoint": str(checkpoint),
        "source_model": str(source),
        "onnx": str(output),
        "engine": str(engine) if engine else None,
        "precision": "explicit_qdq_int8",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "dynamic": args.dynamic,
        "onnx_nodes": counts,
        "tensorrt_built": engine is not None,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
