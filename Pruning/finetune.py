"""Fine-tune a pruned YOLOv8n-KITTI model in FP32 with Ultralytics, then validate the best checkpoint.

The pruned model object is handed to the trainer directly: passing a path would make Ultralytics rebuild the
network from its (unpruned) yaml. Speed settings follow QAT/modelopt_qat_all64.py (RAM image cache, 16 workers,
batch 32); AMP is off so training stays FP32. The best epoch is chosen by Ultralytics fitness (mAP50-95).

Example:
    .venv/Scripts/python.exe -m Pruning.finetune
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import SETTINGS

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/pruning/taylor1_25/8n_float32_taylor1_prune25.pt')
    parser.add_argument('--data', type=Path, default=ROOT/'PTQ/datasets/kitti')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--optimizer', default='SGD')
    parser.add_argument('--lr0', type=float, default=0.01)
    parser.add_argument('--name', default='kitti_taylor1_prune25_ft')
    parser.add_argument('--suffix', default='_ft', help="output checkpoint is <weights stem><suffix>.pt, e.g. '_ft50'")
    args = parser.parse_args()
    args.weights, args.data = args.weights.resolve(), args.data.resolve()   # the report stores paths relative to ROOT

    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
    model = ckpt['model'].float()
    for p in model.parameters():
        p.requires_grad_(True)
    data_yaml = args.data/'kitti.yaml'
    trainer = DetectionTrainer(overrides={
        'model': str(args.weights), 'data': str(data_yaml), 'epochs': args.epochs, 'imgsz': 640,
        'batch': args.batch, 'device': '0', 'workers': args.workers, 'cache': 'ram', 'seed': 42,
        'project': str(ROOT/'Pruning/runs'), 'name': args.name, 'exist_ok': True,
        'optimizer': args.optimizer, 'lr0': args.lr0, 'lrf': 0.01, 'cos_lr': True, 'warmup_epochs': 3.0,
        'close_mosaic': min(10, args.epochs), 'amp': False, 'plots': False, 'val': True, 'save': True})
    trainer.model = model
    begin = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - begin

    best = Path(trainer.best)
    out = args.weights.with_name(f'{args.weights.stem}{args.suffix}.pt')
    shutil.copy2(best, out)
    SETTINGS.update({'datasets_dir': str(args.data.parent)})
    box = YOLO(str(out)).val(data='kitti.yaml', split='val', imgsz=640, batch=1, device=0, rect=False, workers=0,
                             plots=False, save_json=False, verbose=False, project=str(out.parent/'val'),
                             name=f'finetuned{args.suffix}', exist_ok=True).box
    val = {'mAP50-95': box.map, 'mAP75': box.map75, 'mAP50': box.map50, 'precision': box.mp, 'recall': box.mr}
    # The final best.pt is stripped (its epoch is reset), so read the best epoch from results.csv.
    results = trainer.read_results_csv()
    best_epoch = int(max(zip(results['metrics/mAP50-95(B)'], results['epoch']))[1])
    report = {'source': str(args.weights.relative_to(ROOT)), 'checkpoint': out.name, 'run_dir': str(trainer.save_dir),
              'best_epoch': best_epoch, 'epochs': args.epochs, 'batch': args.batch,
              'optimizer': args.optimizer, 'lr0': args.lr0, 'lrf': 0.01, 'cos_lr': True, 'warmup_epochs': 3.0,
              'amp': False, 'cache': 'ram', 'train_seconds': seconds,
              'val_batch1_conf0.001': {k: float(v) for k, v in val.items()}}
    report_name = 'finetune.json' if args.suffix == '_ft' else f'finetune{args.suffix}.json'
    (out.parent/report_name).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Fine-tuned (best epoch {report["best_epoch"]}): {report["val_batch1_conf0.001"]}', flush=True)


if __name__ == '__main__':
    main()
