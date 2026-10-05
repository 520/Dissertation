# 6-left / 4-right piecewise SiLU checkpoint

Final checkpoint: `yolov8n_voc_piecewise64_cpp_final.pt`.
Source: `original/yolov8n_voc/final_model_factorized_lwi.pt` (VOC, 20 classes).
All 56 registered `nn.SiLU` references were replaced; none remain.
The classification sigmoid, DFL softmax and decoding operations remain intact.

## Parameters and implementation

Uses `adaptive_silu_6_4.json:selected`, without fitting new parameters: six
negative knots, four positive knots and zero, giving 11 knots / 10 segments.
Values at and below the lowest knot are zero; values at and above the highest
knot use identity. Interpolation is linear within each segment.

`piecewise64_silu.py` stores knots, values and slopes as FP32 model buffers.
Its portable tensor backend uses vectorized `torch.searchsorted` (binary search).
The inference backend in `piecewise64_torch.cpp` uses a balanced tree with four
levels, NEON byte-table gathers, four FP32 lanes and fused multiply-add on ARM64.
The tree is padded with infinity boundaries to 16 leaves. Other CPU architectures
use scalar binary search. OpenMP builds reuse PyTorch's intra-op thread count.
Training with gradients uses the differentiable tensor backend.

## Loading

Run from the Dissertation project root using the existing ultralytics environment:

```python
import torch
from Activation.piecewise64_silu import Piecewise64SiLU, load_cpp

torch.set_num_threads(10)
load_cpp()  # Explicitly fail if C++ compilation/loading is unavailable.
checkpoint = torch.load(
    "Activation/yolov8n_voc_piecewise64_cpp_final.pt",
    map_location="cpu",
    weights_only=False,
)
model = checkpoint["model"].float().eval()
with torch.inference_mode():
    prediction = model(torch.randn(1, 3, 640, 640))
```

Keep the `Activation` Python package and `piecewise64_torch.cpp` available when
loading. The `.pt` contains model weights and interpolation buffers, not embedded
native machine code. The C++ library is lazily compiled/cached using PyTorch's
extension mechanism. This needs a C++ toolchain and Ninja; OpenMP is needed for
the threaded OpenMP backend. Automatic inference falls back to tensor operations
with a warning if compilation fails. Calling `load_cpp()` first prevents silent
performance fallback. Changing the module's `backend` to `"torch"` explicitly
selects the portable path.

This is a CPU C++ implementation; it is not a CUDA/TensorRT plugin. ONNX/engine
export has not been implemented or verified. The source checkpoint has factorized
Sequential convolutions; the existing standard Ultralytics Conv fuse routine is
incompatible, so use the existing project's skip-fuse validation approach.

## Verification

See `yolov8n_voc_piecewise64_cpp_final.verification.json`. Tests cover a million
inputs, all knots and adjacent FP32 values, tails, infinities, NaN, empty/scalar
and non-contiguous inputs, gradients, full 640x640 model forward, and exact
checkpoint reload. A separate process also loads and runs the final checkpoint.

Activation approximation on a uniform [-8,8] grid: MAE 0.00631037, RMSE 0.00760773,
maximum absolute error 0.02257796. These are activation errors, not detection mAP.
Dataset mAP has not been measured for this checkpoint.

Recorded 10-thread CPU microbenchmark, 409600 elements:

| Implementation | Median ms |
|---|---:|
| Native SiLU | 0.0916 |
| Tensor binary search | 1.6007 |
| C++ SIMD tree | 0.1332 |

Random-input 640x640 full-model forward: native SiLU 41.28 ms, C++ piecewise
45.53 ms. Benchmarks exclude preprocessing and NMS and use short local runs;
this implementation currently accelerates the tensor lookup but does not beat
native SiLU on this machine.

Reproduce export without overwriting an existing checkpoint:

```bash
/opt/anaconda3/envs/ultralytics/bin/python -m Activation.export_piecewise64_pt \
  --output Activation/new_piecewise64.pt --threads 10
```

The earlier `yolov8n_voc_piecewise64_cpp.pt` and
`yolov8n_voc_piecewise64_cpp_optimized.pt` are intermediate verification artifacts;
use the `_final.pt` checkpoint.
