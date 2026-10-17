"""Per-layer INT8 sensitivity of YOLOv8n-KITTI, measured as validation mAP.

All 64 BN-fused convolutions are calibrated once with ModelOpt (same settings as
modelopt_int8_all_convs.py: max calibration, KITTI train, per-channel INT8 weights,
per-tensor INT8 inputs). Then, for each convolution in turn, only that convolution's
quantizers are enabled and the full KITTI validation split is evaluated with Ultralytics.
Reference runs: no quantizer (FP32) and all 64 quantizers (INT8).

Progress is saved after every run; rerunning resumes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ultralytics import settings
import ultralytics.data.utils as data_utils
from ultralytics.models.yolo.detect import DetectionValidator

from modelopt_int8_all_convs import ROOT, calibration_images, load_fused, quantize

DATASETS = ROOT/'PTQ/datasets'


def set_quantized(convs, names):
    for name, module in convs:
        for quantizer in (module.weight_quantizer, module.input_quantizer):
            quantizer.enable() if name in names else quantizer.disable()


def evaluate(model, batch):
    validator = DetectionValidator(args=dict(data='kitti.yaml', imgsz=640, batch=batch, rect=False, half=False,
                                             plots=False, verbose=False, workers=0, device=0, save_json=False))
    validator(model=model)
    box = validator.metrics.box
    return {'map50_95': float(box.map), 'map50': float(box.map50), 'map75': float(box.map75),
            'precision': float(box.mp), 'recall': float(box.mr)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/baseline/8n_float32.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'sensitivity_results/int8_layer_sensitivity')
    parser.add_argument('--calib', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--limit', type=int, default=0, help='evaluate only the first N layers (testing)')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings.update({'datasets_dir': str(DATASETS)})
    data_utils.DATASETS_DIR = DATASETS
    device = torch.device('cuda:0')
    torch.manual_seed(args.seed)

    model = load_fused(args.weights).to(device)
    params = {n: sum(p.numel() for p in m.parameters()) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)}
    hw = {}
    hooks = [m.register_forward_hook(lambda m, i, o, n=n: hw.__setitem__(n, o.shape[2]*o.shape[3]))
             for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        model(torch.zeros(1, 3, 640, 640, device=device))
    for h in hooks:
        h.remove()
    dataset, indices = calibration_images(DATASETS/'kitti', args.calib, args.seed)
    print(f'Calibrating all convolutions on {len(indices)} KITTI training images...', flush=True)
    model, convs, _ = quantize(model, dataset, indices, device, [])
    names = [n for n, _ in convs]

    progress_file = args.output/'progress.json'
    runs = json.loads(progress_file.read_text()) if progress_file.exists() else {}
    plan = [('FP32 (none)', set()), ('INT8 (all 64)', set(names))]
    plan += [(n, {n}) for n in (names[:args.limit] if args.limit else names)]
    for key, quantized in plan:
        if key in runs:
            continue
        begin = time.perf_counter()
        set_quantized(convs, quantized)
        runs[key] = {**evaluate(model, args.batch), 'seconds': time.perf_counter() - begin}
        progress_file.write_text(json.dumps(runs, indent=2), encoding='utf-8')
        print(f"[{len(runs)}/{len(plan)}] {key}: mAP50-95 {runs[key]['map50_95']*100:.2f} "
              f"({runs[key]['seconds']:.0f}s)", flush=True)

    ref = runs['FP32 (none)']
    rows = []
    for index, name in enumerate(names):
        if name not in runs:
            continue
        r = runs[name]
        rows.append({'layer_index': index, 'layer': name, 'params': params[name], 'out_hw': hw[name],
                     'macs_M': params[name]*hw[name]/1e6,
                     **{f'{k} (%)': r[k]*100 for k in ('map50_95', 'map50', 'map75', 'precision', 'recall')},
                     **{f'drop_{k} (pts)': (ref[k]-r[k])*100 for k in ('map50_95', 'map50', 'map75')}})
    table = pd.DataFrame(rows).sort_values('drop_map50_95 (pts)', ascending=False)
    table.insert(0, 'rank', np.arange(1, len(table)+1))
    table.to_csv(args.output/'layer_sensitivity.csv', index=False, encoding='utf-8-sig')
    summary = {'weights': str(args.weights), 'calibration_images': len(indices), 'seed': args.seed,
               'reference': {k: v for k, v in runs.items() if k in ('FP32 (none)', 'INT8 (all 64)')},
               'sum_of_single_layer_drops_map50_95_pts': float(table['drop_map50_95 (pts)'].sum()),
               'layers_evaluated': len(table)}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(table.head(15).to_string(index=False, float_format=lambda v: f'{v:.3f}'))


if __name__ == '__main__':
    main()
