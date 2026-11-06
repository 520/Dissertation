# TFLite FP32 CPU latency baseline — 8n_float32_lrf_ft50_ch8_b0.1_ft50_qat_int8_all104_qatw_fp32_float32

- Model: `models\quantization\int8\qat\lrfboth_b1.7ft50_ch8_b0.1_ft50_all\tflite_int8\8n_float32_lrf_ft50_ch8_b0.1_ft50_qat_int8_all104_qatw_fp32_float32.tflite` (5.63 MB, input [1, 640, 640, 3] float32, output [1, 12, 8400])
- Runtime: LiteRT (ai-edge-litert) 2.3.0, Python 3.12.7, Windows 11 (10.0.22631)
- CPU: 12th Gen Intel(R) Core(TM) i7-12700 (20 logical CPUs)
- XNNPACK delegation: 290/292 ops in 3 partitions; not delegated: {'RESIZE_NEAREST_NEIGHBOR': 2}
- Protocol: batch 1, 500 images from `PTQ\datasets\kitti\images\val` (000000.png ... 002553.png, letterboxed, preloaded, cycled); 20 warmup invokes, then >= 15.0 s and >= 500 timed `invoke()` calls (every image timed at least once); input copy, pre/post-processing and NMS excluded.

| Backend | Threads | Median (ms) | P75 (ms) | P95 (ms) | Mean (ms) | Std (ms) | Min (ms) | FPS (1/mean) | Samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| xnnpack | 1 | 87.38 | 88.35 | 90.46 | 87.73 | 1.65 | 85.02 | 11.4 | 500 |
