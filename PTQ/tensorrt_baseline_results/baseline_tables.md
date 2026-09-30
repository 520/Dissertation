# Table 1. FP32 baseline accuracy and inference performance

| Model | Dataset | Precision | mAP50-95 (%) | mAP50 (%) | Engine median / P95 (ms) | E2E median / P95 (ms) | E2E FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv8n | KITTI | FP32 | 67.55 | 90.49 | 2.623 / 3.151 | 4.264 / 5.625 | 216.8 |
| YOLOv8n | VOC | FP32 | 62.13 | 82.51 | 2.681 / 3.232 | 4.377 / 5.493 | 216.8 |

Batch = 1; input = 640 x 640; GPU = NVIDIA RTX A2000. mAP is reported as a percentage. Engine latency uses CUDA events and resident GPU buffers. E2E includes preprocessing, data transfer, inference, NMS and result construction; disk I/O is excluded. FPS is completed images divided by measured wall time, not the reciprocal of median latency. P95 describes per-image latency within a run.

# Table 2. FP32 baseline deployment resources and GPU energy

| Model | Dataset | Params (M) | GFLOPs | Weights (MB) | Engine (MB) | Approx. GPU load delta (MiB) | E2E GPU power (W) | E2E GPU energy (J/image) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv8n | KITTI | 3.012 | 8.20 | 6.24 | 79.82 | 147.19 | 63.91 | 0.2947 |
| YOLOv8n | VOC | 3.015 | 8.21 | 6.26 | 93.51 | 138.00 | 63.84 | 0.2945 |

Parameters and GFLOPs describe the original PyTorch checkpoint; 1 MAC is counted as 2 FLOPs. File sizes use decimal MB. GPU load delta uses total-device NVML memory before loading versus after warmup and is approximate, not process-specific model memory. Power and energy use the E2E sustained test; energy includes idle GPU power and covers the GPU board, not the full computer. Results are single-run measurements; repeat under comparable thermal conditions before final publication.
