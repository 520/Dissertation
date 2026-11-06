# TFLite FP32 CPU latency baseline — 8n_float32_float32

- Model: `original\yolov8n_kitti\8n_float32_saved_model\8n_float32_float32.tflite` (12.19 MB, input [1, 640, 640, 3] float32, output [1, 12, 8400])
- Runtime: LiteRT (ai-edge-litert) 2.3.0, Python 3.12.7, Windows 11 (10.0.22631)
- CPU: 12th Gen Intel(R) Core(TM) i7-12700 (20 logical CPUs)
- XNNPACK delegation: 250/252 ops in 3 partitions; not delegated: {'RESIZE_NEAREST_NEIGHBOR': 2}
- Protocol: batch 1, 500 images from `PTQ\datasets\kitti\images\val` (000000.png ... 002553.png, letterboxed, preloaded, cycled); 20 warmup invokes, then >= 15.0 s and >= 500 timed `invoke()` calls (every image timed at least once); input copy, pre/post-processing and NMS excluded.

| Backend | Threads | Median (ms) | P75 (ms) | P95 (ms) | Mean (ms) | Std (ms) | Min (ms) | FPS (1/mean) | Samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| xnnpack | 1 | 115.87 | 116.81 | 118.43 | 116.11 | 1.26 | 113.18 | 8.6 | 500 |
| builtin_no_xnnpack | 1 | 259.10 | 261.68 | 266.68 | 259.66 | 4.98 | 249.92 | 3.9 | 500 |
| xnnpack | 4 | 32.31 | 32.75 | 33.59 | 32.43 | 0.90 | 31.01 | 30.8 | 500 |
| builtin_no_xnnpack | 4 | 140.39 | 141.08 | 142.06 | 139.05 | 4.49 | 123.83 | 7.2 | 500 |
| xnnpack | 8 | 22.47 | 23.36 | 25.59 | 22.76 | 1.61 | 19.68 | 43.9 | 633 |
| builtin_no_xnnpack | 8 | 109.14 | 110.00 | 116.19 | 109.30 | 3.76 | 101.04 | 9.1 | 500 |

Numerical check vs PyTorch `.pt` (same input): box max |diff| 0.0038 px, score max |diff| 4.77e-07, top-100 anchor overlap 100/100.
