"""Fine-tune the plan-pruned YOLOv8n-KITTI (Pruning/plan_prune.py) in FP32 for 10 epochs and export the best .pt.

Same training recipe as Pruning/finetune.py (SGD, lr0 0.01, cosine to lrf 0.01, 3 warmup epochs, AMP off, RAM
cache, batch 32, seed 42); the pruned model object is handed to the trainer directly so Ultralytics does not rebuild
the unpruned network from its yaml. The checkpoint keeps two C2fSplit blocks (model.8, model.18), so it loads only
with this repository on the import path (run from the repository root with -m).

Both the pruned model before fine-tuning and the exported best checkpoint are validated on the full KITTI val split
with the same protocol as the earlier checkpoints (imgsz 640, batch 1, rect=False, conf 0.001).

Outputs (next to --weights):
  <stem>_ft.pt     best fine-tuned checkpoint (by Ultralytics fitness, mAP50-95)
  finetune.json    recipe, best epoch, mAP before and after fine-tuning

Example:
    .venv/Scripts/python.exe -m Pruning.plan_finetune
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def validate(weights, data_yaml, name):
    box = YOLO(str(weights)).val(data=str(data_yaml), split='val', imgsz=640, batch=1, device=0, rect=False, workers=0,
                                 plots=False, save_json=False, verbose=False, project=str(weights.parent/'val'),
                                 name=name, exist_ok=True).box
    return {'mAP50-95': float(box.map), 'mAP75': float(box.map75), 'mAP50': float(box.map50),
            'precision': float(box.mp), 'recall': float(box.mr)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/pruning/l2ch8_plan/8n_float32_l2ch8_plan.pt')
    parser.add_argument('--data', type=Path, default=ROOT/'PTQ/datasets/kitti/kitti.yaml')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--optimizer', default='SGD')
    parser.add_argument('--lr0', type=float, default=0.01)
    parser.add_argument('--name', default='kitti_l2ch8_plan_ft10')
    args = parser.parse_args()
    weights = args.weights.resolve()

    before = validate(weights, args.data, 'pruned')
    print(f'Pruned, before fine-tuning: {before}', flush=True)

    model = torch.load(weights, map_location='cpu', weights_only=False)['model'].float()
    for p in model.parameters():
        p.requires_grad_(True)
    trainer = DetectionTrainer(overrides={
        'model': str(weights), 'data': str(args.data), 'epochs': args.epochs, 'imgsz': 640,
        'batch': args.batch, 'device': '0', 'workers': args.workers, 'cache': 'ram', 'seed': 42,
        'project': str(ROOT/'Pruning/runs'), 'name': args.name, 'exist_ok': False,
        'optimizer': args.optimizer, 'lr0': args.lr0, 'lrf': 0.01, 'cos_lr': True, 'warmup_epochs': 3.0,
        'close_mosaic': min(10, args.epochs), 'amp': False, 'plots': False, 'val': True, 'save': True})
    trainer.model = model
    begin = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - begin

    out = weights.with_name(f'{weights.stem}_ft.pt')
    shutil.copy2(trainer.best, out)
    after = validate(out, args.data, 'finetuned')
    # The final best.pt is stripped (its epoch is reset), so read the best epoch from results.csv.
    results = trainer.read_results_csv()
    best_epoch = int(max(zip(results['metrics/mAP50-95(B)'], results['epoch']))[1])
    report = {'source': str(weights.relative_to(ROOT)), 'checkpoint': out.name, 'run_dir': str(trainer.save_dir),
              'best_epoch': best_epoch, 'epochs': args.epochs, 'batch': args.batch,
              'optimizer': args.optimizer, 'lr0': args.lr0, 'lrf': 0.01, 'cos_lr': True, 'warmup_epochs': 3.0,
              'close_mosaic': min(10, args.epochs), 'amp': False, 'cache': 'ram', 'train_seconds': seconds,
              'val_batch1_conf0.001': {'before_finetune': before, 'after_finetune': after}}
    (out.parent/'finetune.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved {out} (best epoch {best_epoch}): mAP50-95 {before["mAP50-95"]:.4f} -> {after["mAP50-95"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
