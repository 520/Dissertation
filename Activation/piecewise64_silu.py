"""Portable 6-left/4-right SiLU with an optional SIMD C++ inference kernel.

Import this module when loading a saved checkpoint. C++ is built lazily on CPU
inference; training and other devices use differentiable tensor operations.
"""
from pathlib import Path
import json
import warnings
import os
import shutil
import sys

import torch
from torch import nn

_CPP_READY = False
_CPP_ERROR = None


def load_cpp():
    global _CPP_READY, _CPP_ERROR
    if _CPP_READY:
        return
    if _CPP_ERROR is not None:
        raise RuntimeError("Piecewise64 C++ build previously failed") from _CPP_ERROR
    try:
        from torch.utils.cpp_extension import load
        cflags = ["-O3", "-std=c++17"]
        ldflags = []
        if "ATen parallel backend: OpenMP" in torch.__config__.parallel_info():
            if sys.platform == "darwin":
                cflags += ["-Xpreprocessor", "-fopenmp", "-I" + str(Path(sys.prefix) / "include")]
                torch_lib = Path(torch.__file__).parent / "lib"
                ldflags += ["-L" + str(torch_lib), "-lomp", "-Wl,-rpath," + str(torch_lib)]
            else:
                cflags += ["-fopenmp"]
                ldflags += ["-fopenmp"]
        # Conda's Python may be used without its bin directory on PATH.
        import ninja
        previous_path = os.environ.get("PATH", "")
        if shutil.which("ninja") is None:
            os.environ["PATH"] = str(ninja.BIN_DIR) + os.pathsep + previous_path
        try:
            load(
                name="dissertation_piecewise64_cpu",
                sources=[str(Path(__file__).with_name("piecewise64_torch.cpp"))],
                extra_cflags=cflags,
                extra_ldflags=ldflags,
                is_python_module=False,
                verbose=False,
            )
        finally:
            os.environ["PATH"] = previous_path
        _CPP_READY = True
    except Exception as exc:
        _CPP_ERROR = exc
        raise


class Piecewise64SiLU(nn.Module):
    def __init__(self, backend="cpp", parameter_path=None):
        super().__init__()
        if backend not in {"cpp", "torch"}:
            raise ValueError("backend must be cpp or torch")
        path = Path(parameter_path) if parameter_path else Path(__file__).with_name("adaptive_silu_6_4.json")
        selected = json.loads(path.read_text())["selected"]
        knots = torch.tensor(selected["knots"], dtype=torch.float32)
        values = torch.tensor(selected["values"], dtype=torch.float32)
        if len(knots) != 11 or not torch.all(knots[1:] > knots[:-1]):
            raise ValueError("Expected 11 strictly ordered knots")
        self.register_buffer("knots", knots)
        self.register_buffer("values", values)
        self.register_buffer("slopes", values.diff() / knots.diff())
        self.backend = backend

    def torch_forward(self, x):
        # Vectorized binary search over nine interior boundaries: 10 segments.
        index = torch.searchsorted(self.knots[1:-1].contiguous(), x.contiguous(), right=True)
        y = self.values[index] + (x - self.knots[index]) * self.slopes[index]
        y = torch.where(x <= self.knots[0], torch.zeros_like(y), y)
        return torch.where(x >= self.knots[-1], x, y)

    def forward(self, x):
        if (self.backend == "cpp" and x.device.type == "cpu" and x.dtype == torch.float32
                and not (torch.is_grad_enabled() and x.requires_grad)
                and not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export()):
            if _CPP_ERROR is None:
                try:
                    load_cpp()
                except Exception as exc:
                    warnings.warn(f"Piecewise64 C++ unavailable; using tensor fallback: {exc}", RuntimeWarning)
            if _CPP_READY:
                return torch.ops.dissertation_piecewise64.forward(x, self.knots, self.values, self.slopes)
        return self.torch_forward(x)

    def extra_repr(self):
        return f"negative=6, positive=4, zero=1, backend={self.backend}"


def replace_silu(model, replacement):
    count = 0
    for name, child in list(model._modules.items()):
        if child is None:
            continue
        if isinstance(child, nn.SiLU):
            model._modules[name] = replacement
            count += 1
        else:
            count += replace_silu(child, replacement)
    return count
