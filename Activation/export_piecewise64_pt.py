"""Verify the 6/4 tensor and SIMD kernels, then save a replaced VOC YOLO .pt."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import time

import torch
from torch import nn

from Activation.piecewise64_silu import Piecewise64SiLU, load_cpp, replace_silu
from Activation.yolo_lut_kitti import load_raw_model

ROOT = Path(__file__).resolve().parents[1]


def benchmark(function, x, repeats=30):
    for _ in range(5):
        function(x)
    times = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function(x)
        times.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "original/yolov8n_voc/final_model_factorized_lwi.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "Activation/yolov8n_voc_piecewise64_cpp.pt")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {args.output}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    load_cpp()  # Fail here rather than silently benchmark the fallback.
    activation = Piecewise64SiLU().eval()
    report = {"model": str(args.model.resolve()), "output": str(args.output.resolve()), "threads": args.threads,
              "parameters": "adaptive_silu_6_4.json:selected", "backend": "cpp_cpu_neon_binary_search"}

    with torch.inference_mode():
        k = activation.knots
        boundaries = torch.cat((k, torch.nextafter(k, torch.full_like(k, -torch.inf)),
                                torch.nextafter(k, torch.full_like(k, torch.inf)),
                                torch.tensor([-torch.inf, torch.inf, torch.nan])))
        maximum = 0.0
        for x in [boundaries, torch.linspace(-8, 8, 1_000_001), torch.randn(7, 13).t(),
                  torch.empty(0), torch.tensor(0.3), torch.randn(513) * 10]:
            expected = activation.torch_forward(x)
            actual = activation(x)
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6, equal_nan=True)
            finite = torch.isfinite(expected) & torch.isfinite(actual)
            if finite.any():
                maximum = max(maximum, float((actual[finite] - expected[finite]).abs().max()))
        report["cpp_vs_tensor_max_abs_error"] = maximum
        grid = torch.linspace(-8, 8, 1_000_001)
        error = activation(grid) - torch.nn.functional.silu(grid)
        report["activation_uniform_minus8_plus8"] = {
            "mae": float(error.abs().mean()), "rmse": float(error.square().mean().sqrt()),
            "max_abs_error": float(error.abs().max())}
        sample = torch.randn(1, 64, 80, 80)
        report["activation_median_ms_409600_elements"] = {
            "native_silu": benchmark(torch.nn.functional.silu, sample),
            "tensor_binary_search": benchmark(activation.torch_forward, sample),
            "cpp_simd_binary_search": benchmark(activation, sample)}

    # The portable tensor path must still support training gradients.
    grad_input = torch.tensor([-1.0, 0.3, 2.0], requires_grad=True)
    activation(grad_input).sum().backward()
    assert torch.isfinite(grad_input.grad).all()
    report["gradient_fallback_passed"] = True

    model = load_raw_model(args.model)
    original = copy.deepcopy(model)
    expected_count = sum(isinstance(c, nn.SiLU) for p in model.modules() for c in p._modules.values())
    replaced = replace_silu(model, activation)
    remaining = sum(isinstance(m, nn.SiLU) for m in model.modules())
    assert expected_count > 0 and replaced == expected_count and remaining == 0
    report["replaced_silu_references"] = replaced
    report["remaining_silu"] = remaining
    x = torch.randn(1, 3, 640, 640)
    with torch.inference_mode():
        baseline = original(x)
        cpp = model(x)
        activation.backend = "torch"
        portable = model(x)
        activation.backend = "cpp"
        def flatten(value):
            if isinstance(value, torch.Tensor):
                return [value]
            if isinstance(value, dict):
                return [t for v in value.values() for t in flatten(v)]
            return [t for v in value for t in flatten(v)]
        cpp_tensors, portable_tensors = flatten(cpp), flatten(portable)
        assert len(cpp_tensors) == len(portable_tensors)
        for a, b in zip(cpp_tensors, portable_tensors):
            torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-3)
            assert torch.isfinite(a).all()
        report["forward_output_shapes"] = [list(t.shape) for t in cpp_tensors]
        report["full_forward_cpp_vs_tensor_max_abs_error"] = max(float((a-b).abs().max()) for a,b in zip(cpp_tensors, portable_tensors))
        report["full_forward_median_ms"] = {
            "original_silu": benchmark(original, x, repeats=10),
            "piecewise64_cpp": benchmark(model, x, repeats=10)}
        report["silu_vs_piecewise_prediction_max_abs_change_random_input"] = float((flatten(baseline)[0]-cpp_tensors[0]).abs().max())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model, "piecewise64": report}, args.output)
    reloaded = torch.load(args.output, map_location="cpu", weights_only=False)["model"].eval()
    with torch.inference_mode():
        for a, b in zip(cpp_tensors, flatten(reloaded(x))):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    report["checkpoint_reload_exact_match"] = True
    report["validation_scope"] = "Kernel, gradient, random 640x640 full forward and reload; dataset mAP not measured"
    destination = args.output.with_suffix(".verification.json")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
