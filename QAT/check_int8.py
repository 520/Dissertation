#!/usr/bin/env python3
"""Profile one CPU forward pass of a project PyTorch or ONNX model.

Run in PyCharm directly or: python -m QAT.check_int8 --model /path/model.pt
This checks executed operators, not model accuracy or speed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Edit this path to run directly using the IDE's Run button.
DEFAULT_MODEL = PROJECT_ROOT / "QAT/runs/kitti_qat_10e-3/weights/best_cpu_int8.pt"


def check_model(path: Path, imgsz: int = 640) -> bool:
    if not path.is_file():
        raise FileNotFoundError(path)
    if imgsz < 32 or imgsz % 32:
        raise ValueError("imgsz must be a positive multiple of 32")
    if path.suffix.lower() == ".onnx":
        return check_onnx(path, imgsz)
    if path.suffix.lower() != ".pt":
        raise ValueError("Supported model formats: .pt, .onnx")

    import torch
    from QAT.val import load_model
    from QAT.torch_qat import MinMaxFakeQuant

    torch.set_num_threads(4)

    # Project loader freezes QAT observers and preserves packed INT8 layers.
    # Floating-point layers are normalized to FP32 for this CPU check.
    model = load_model(path)
    generator = torch.Generator().manual_seed(0)
    sample = torch.rand(1, 3, imgsz, imgsz, generator=generator)
    with torch.inference_mode():
        model(sample)  # Warm up outside the measured forward pass.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile:
            output = model(sample)

    def check_finite(value):
        if isinstance(value, torch.Tensor):
            value = value.dequantize() if value.is_quantized else value
            if not torch.isfinite(value).all():
                raise RuntimeError("Inference produced NaN or Inf")
        elif isinstance(value, (tuple, list)):
            for item in value:
                check_finite(item)
        elif isinstance(value, dict):
            for item in value.values():
                check_finite(item)

    check_finite(output)
    events = {event.key: event.count for event in profile.key_averages()}
    # Exact names avoid counting prepack, unpack or dynamic operators as
    # evidence of static 8-bit activation/weight convolution execution.
    static_conv_ops = {
        "quantized::conv2d", "quantized::conv2d.new",
        "quantized::conv2d_relu", "quantized::conv2d_relu.new",
    }
    executed = {key: count for key, count in events.items() if key in static_conv_ops}
    fake_quantizers = sum(isinstance(m, MinMaxFakeQuant) for m in model.modules())
    print(f"Model: {path.resolve()}")
    print(f"Device: CPU | backend: {torch.backends.quantized.engine}")
    print("Floating layers run in FP32; this is not an FP16 check.")
    print(f"Custom fake-quant modules: {fake_quantizers}")
    print("Convolution events in ONE forward pass (nested events may overlap):")
    for name, count in sorted(events.items()):
        if "conv" in name.lower():
            print(f"  {name}: {count}")
    if executed:
        print("PASS: real static INT8 quantized convolution was executed.")
        for name, count in sorted(executed.items()):
            print(f"  Evidence: {name} = {count} calls")
        print("This does not mean every model operation is INT8 or measure accuracy.")
        return True
    print("NOT CONFIRMED: no supported static INT8 convolution event was observed.")
    if fake_quantizers:
        print("Custom fake quantizers are present; they alone do not prove INT8 execution.")
    return False


def check_onnx(path: Path, imgsz: int) -> bool:
    """Use executed CPU kernel events after ORT optimization, including QDQ fusion."""
    import numpy as np
    import onnxruntime as ort

    with tempfile.TemporaryDirectory(prefix="check_int8_") as directory:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(directory) / "profile")
        session = ort.InferenceSession(
            str(path.resolve()), sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        try:
            inputs = session.get_inputs()
            if len(inputs) != 1 or len(inputs[0].shape) != 4:
                raise ValueError("ONNX checker expects one NCHW image input")
            spec = inputs[0]
            dtypes = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
                      "tensor(uint8)": np.uint8, "tensor(int8)": np.int8}
            if spec.type not in dtypes:
                raise ValueError(f"Unsupported ONNX input dtype: {spec.type}")
            # Honor fixed export dimensions; --imgsz fills dynamic H/W only.
            shape = tuple(dim if isinstance(dim, int) and dim > 0 else fallback
                          for dim, fallback in zip(spec.shape, (1, 3, imgsz, imgsz)))
            if shape[1] != 3:
                raise ValueError(f"Expected RGB NCHW input, got {shape}")
            dtype = dtypes[spec.type]
            rng = np.random.default_rng(0)
            if np.issubdtype(dtype, np.integer):
                limits = np.iinfo(dtype)
                sample = rng.integers(limits.min, limits.max + 1, size=shape, dtype=dtype)
            else:
                sample = rng.random(shape).astype(dtype)
            # Exactly one run, so the trace counts exclude warmup duplicates.
            output = session.run(None, {spec.name: sample})
            if not all(np.isfinite(value).all() for value in output):
                raise RuntimeError("ONNX inference produced NaN or Inf")
        finally:
            profile_path = session.end_profiling()
        events = json.loads(Path(profile_path).read_text())

    kernels = Counter()
    for event in events:
        args = event.get("args", {})
        # Fence events are scheduling records, not executed kernels.
        if (event.get("cat") == "Node"
                and event.get("name", "").endswith("_kernel_time")
                and args.get("provider")):
            kernels[(args.get("op_name", "unknown"), args["provider"])] += 1
    quantized_ops = {"QLinearConv", "ConvInteger", "QLinearConvTranspose"}
    confirmed = {key: count for key, count in kernels.items()
                 if key[0] in quantized_ops and key[1] == "CPUExecutionProvider"}
    print(f"Model: {path.resolve()}")
    print(f"ONNX Runtime: {ort.__version__} | provider: CPUExecutionProvider")
    print(f"Input: {spec.name}, {shape}, {spec.type}")
    print("Executed convolution / QDQ kernels in ONE forward pass:")
    for (op, provider), count in sorted(kernels.items()):
        if "conv" in op.lower() or op in {"QuantizeLinear", "DequantizeLinear"}:
            print(f"  {op} [{provider}]: {count}")
    if confirmed:
        print("PASS: real INT8 quantized convolution was executed by ONNX Runtime.")
        for (op, provider), count in sorted(confirmed.items()):
            print(f"  Evidence: {op} [{provider}] = {count} calls")
        print("This confirms this CPU run, not every operation, GPU execution or accuracy.")
        return True
    print("NOT CONFIRMED: no recognized INT8 convolution kernel was observed.")
    print("INT8 weights or QuantizeLinear/DequantizeLinear alone are not proof.")
    print("This result does not by itself classify the file as fake quantization.")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    return 0 if check_model(args.model, args.imgsz) else 1


if __name__ == "__main__":
    raise SystemExit(main())
    # python -m QAT.check_int8 --model "original/yolov8n_voc/voc_8n_best_int8.onnx"
    # python -m QAT.check_int8 --model "original/yolov8n_voc/voc_8n_best_int8.onnx"
    # python -m QAT.check_int8 --model "QAT/runs/kitti_qat_10e-3/weights/best_cpu_int8.pt"