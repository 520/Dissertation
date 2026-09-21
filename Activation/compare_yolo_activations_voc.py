#!/usr/bin/env python3
"""Compare native SiLU, C LUT21 and PyTorch C++ LUT21 on Pascal VOC."""

from __future__ import annotations

try:
    import Activation.compare_yolo_activations_kitti as comparison
except ModuleNotFoundError:
    import compare_yolo_activations_kitti as comparison


comparison.DEFAULT_GIT_MODEL = (
    "HEAD:original/yolov8n_voc/final_model_factorized_lwi.pt"
)
comparison.DEFAULT_DATA = comparison.PROJECT_ROOT / "datasets/VOC/VOC.yaml"
comparison.DEFAULT_RESULTS = (
    comparison.PROJECT_ROOT / "Activation/yolo_voc_activation_comparison.json"
)
comparison.DEFAULT_DATASET_LABEL = "VOC"


if __name__ == "__main__":
    comparison.main()
