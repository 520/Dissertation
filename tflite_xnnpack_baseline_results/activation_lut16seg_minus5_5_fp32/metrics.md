# TFLite FP32 CPU latency baseline — 8n_float32_lut16seg_minus5_5_float32

- Model: `models\activation\8n_float32_lut16seg_minus5_5_saved_model\8n_float32_lut16seg_minus5_5_float32.tflite` (12.37 MB, input [1, 640, 640, 3] float32, output [1, 12, 8400])
- Runtime: LiteRT (ai-edge-litert) 2.3.0, Python 3.12.7, Windows 11 (10.0.22631)
- CPU: 12th Gen Intel(R) Core(TM) i7-12700 (20 logical CPUs)
- XNNPACK delegation: 531/1429 ops in 93 partitions; not delegated: {'SELECT': 56, 'SELECT_V2': 56, 'LESS_EQUAL': 56, 'GREATER_EQUAL': 56, 'GATHER': 112, 'ADD': 112, 'MUL': 112, 'CAST': 168, 'LESS': 56, 'MAXIMUM': 56, 'MINIMUM': 56, 'RESIZE_NEAREST_NEIGHBOR': 2}
- Protocol: batch 1, 500 images from `PTQ\datasets\kitti\images\val` (000000.png ... 002553.png, letterboxed, preloaded, cycled); 20 warmup invokes, then >= 15.0 s and >= 500 timed `invoke()` calls (every image timed at least once); input copy, pre/post-processing and NMS excluded.

| Backend | Threads | Median (ms) | P75 (ms) | P95 (ms) | Mean (ms) | Std (ms) | Min (ms) | FPS (1/mean) | Samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| xnnpack | 1 | 293.49 | 294.99 | 300.65 | 294.26 | 3.65 | 288.94 | 3.4 | 500 |

Numerical check vs PyTorch `.pt` (same input): box max |diff| 338.1057 px, score max |diff| 9.66e-01, top-100 anchor overlap 1/100.

## Note: latency is used, outputs are not (decision 2026-11-06)

The 293.49 ms median is the reported TFLite/XNNPACK latency for the LUT16-segment SiLU model. The converted model's
outputs are **not** numerically valid; accuracy for this model must come from PyTorch / TensorRT evaluation.

- PyTorch -> ONNX is exact (box max |diff| 0.0024 px vs PyTorch). Both errors come from onnx2tf 1.28.8 (ONNX -> TFLite):
  1. LUT tables shifted by one entry: onnx2tf treats every 1-D constant with > 2 elements as an NCHW shape vector and
     reorders it to NHWC (`[v0, v2, ..., v16, v1]`), so every Gather returns the next table entry. Fixed by
     `onnx2tf_lut_fix.py` (tables stored pre-permuted); all 56 LUT activations then match ONNX to ~1e-5.
  2. Detect-head layout: with the Cast/Gather/Where activation, onnx2tf's layout inference differs per head branch
     before the six `Conv -> Reshape([1, C, H*W])` ops (one branch correct, one missing the NHWC->NCHW transpose, one
     transposed twice). Not fixed.
- Why the latency is still representative: neither error changes the computation graph. (1) only changes constant
  values; (2) only adds/omits transposes of the small head tensors (at most 64 x 6400 floats, << 0.1 ms each), i.e.
  far below 1% of 293 ms. The slowdown vs SiLU (115.87 ms) comes from XNNPACK not supporting
  GATHER / SELECT / CAST / comparison ops: only 531 of 1429 ops are delegated, in 93 partitions.
- The benchmarked file is the original `yolo export format=saved_model` output (sha256 in `results.json`). The
  `_saved_model/` folder was later overwritten by a re-conversion attempt with fix (1) applied, so its current
  `*_float32.tflite` has a different hash.

Suggested thesis footnote: *"TFLite latency of the LUT-SiLU model was measured on the onnx2tf-converted graph, whose
detection-head layout is numerically incorrect due to a converter issue; the error does not alter the operator
graph or its computational cost, so the latency is representative, while accuracy is reported from the PyTorch
model."*
