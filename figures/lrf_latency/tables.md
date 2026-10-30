**Overall comparison of the baseline and the compressed model (KITTI, 640×640, batch 1, RTX A2000).**

| Metric | Baseline | LRF + pruning | Change |
|---|---|---|---|
| Parameters (M) | 3.012 | 1.363 | -54.8% |
| Compute (GMACs) | 4.05 | 2.90 | -28.2% |
| Conv2d layers | 64 | 104 | +62.5% |
| TensorRT layers (after fusion) | 245 | 271 | +10.6% |
| TensorRT FP32 latency (ms) | 2.600 | 2.638 | +1.5% |
| PyTorch eager latency (ms) | 5.244 | 6.728 | +28.3% |
| mAP50-95 (%) | 67.53 | 65.33 | -2.20 pp |

**Per-stage effect of compression. Conv layers, compute (MMACs) and TensorRT time (µs), baseline → compressed.**

| Stage | Size | Conv layers | MMACs | ΔMACs | TRT µs | ΔTime |
|---|---|---|---|---|---|---|
| 4 | 80 | 6 → 7 | 315 → 305 | -3.1% | 165 → 168 | +1.8% |
| 5 | 40 | 1 → 2 | 118 → 62 | -47.1% | 33 → 39 | +18.8% |
| 6 | 40 | 6 → 11 | 315 → 190 | -39.6% | 155 → 202 | +29.7% |
| 7 | 20 | 1 → 2 | 118 → 77 | -35.1% | 54 → 62 | +14.2% |
| 8 | 20 | 4 → 8 | 184 → 96 | -47.5% | 112 → 155 | +38.9% |
| 12 | 40 | 4 → 5 | 236 → 226 | -4.2% | 131 → 147 | +12.5% |
| 16 | 40 | 1 → 2 | 59 → 43 | -27.8% | 30 → 36 | +20.7% |
| 18 | 40 | 4 → 8 | 197 → 114 | -42.0% | 92 → 125 | +35.6% |
| 19 | 20 | 1 → 2 | 59 → 4 | -93.7% | 79 → 51 | -35.7% |
| 21 | 20 | 4 → 12 | 197 → 2 | -98.8% | 116 → 130 | +12.1% |
| 22 | head | 19 → 32 | 1484 → 1019 | -31.3% | 653 → 725 | +11.0% |
| Total |  | 51 → 91 | 3280 → 2138 | -34.8% | 1620 → 1839 | +13.5% |

**TensorRT time by operator type (IProfiler per-layer medians): kernel count, ms and share of the layer sum.**

| Operator | Base kernels | Base ms | % | LRF kernels | LRF ms | % | ΔTime |
|---|---|---|---|---|---|---|---|
| Convolution | 61 | 1.582 | 51.9 | 99 | 1.884 | 57.1 | +19.1% |
| SiLU activation (+ residual Add) | 56 | 0.698 | 22.9 | 58 | 0.676 | 20.5 | -3.1% |
| Tensor copies (split/concat/resize) | 33 | 0.409 | 13.4 | 30 | 0.434 | 13.2 | +6.0% |
| Layout reformatting | 72 | 0.189 | 6.2 | 61 | 0.107 | 3.2 | -43.7% |
| Other (maxpool, DFL, decode) | 23 | 0.168 | 5.5 | 23 | 0.196 | 6.0 | +17.2% |
| Total (IProfiler sum) | 245 | 3.046 | 100.0 | 271 | 3.296 | 100.0 | +8.2% |
