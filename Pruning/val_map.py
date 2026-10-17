"""KITTI val mAP for the baseline and L1-pruned YOLOv8n (pt and TensorRT engine), same settings."""
import json
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'PTQ/datasets/kitti/kitti.yaml'
PRUNED = ROOT / 'models/pruning/l1_20'
MODELS = {
    'baseline_pt': ROOT / 'models/baseline/8n_float32.pt',
    'pruned_pt': PRUNED / '8n_float32_l1_prune20.pt',
    'pruned_engine': PRUNED / '8n_float32_l1_prune20.engine',
    'global_pt': ROOT / 'models/pruning/l1_global20/8n_float32_l1_global20.pt',
    'global_engine': ROOT / 'models/pruning/l1_global20/8n_float32_l1_global20.engine',
}


def main():
    names = sys.argv[1:] or list(MODELS)
    out = PRUNED / 'val_map.json'
    results = json.loads(out.read_text()) if out.is_file() else {}
    for name in names:
        path = MODELS[name]
        engine = path.suffix == '.engine'
        metrics = YOLO(str(path), task='detect').val(
            data=str(DATA), split='val', imgsz=640, batch=1 if engine else 16, conf=0.001, iou=0.7,
            half=False, device=0, rect=False, workers=0, plots=False, verbose=False,
            project=str(PRUNED / 'val_runs'), name=name, exist_ok=True)
        results[name] = {'model': str(path), 'map50_95': metrics.box.map, 'map50': metrics.box.map50,
                         'map75': metrics.box.map75,
                         'per_class_map50_95': dict(zip(metrics.names.values(), map(float, metrics.box.maps))),
                         'speed_ms': metrics.speed}
        print(f"{name}: mAP50-95={metrics.box.map:.4f} mAP50={metrics.box.map50:.4f}", flush=True)
        out.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
