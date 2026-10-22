"""Apply a low-rank factorisation plan to YOLOv8n-KITTI, save the checkpoint and validate it on KITTI val.

The plan is a text file with one line per decomposed layer, "<layer> <option>:<rank>", options as in
Decomposition/knapsack_lowrank.py: tucker2-out / tucker2-in for k x k Conv blocks, svd for 1 x 1 convs. Layers
not listed are kept. Decomposition is plain truncated SVD / Tucker-2 (no data, trained BN kept, no fine-tuning).
The checkpoint stores the model object, as the pruned checkpoints do, so Pruning/finetune.py can train it
(rebuilding from the yaml would undo the factorisation). Loading it needs the repository root on sys.path
(Decomposition.tucker2_sensitivity.TuckerConv).

Validation as Pruning/finetune.py: YOLO(...).val on the full KITTI val split, imgsz 640, batch 1, rect=False.

Example:
    .venv/Scripts/python.exe -m Decomposition.apply_lrf --plan models/lrf/budget2.5/plan.txt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

from Decomposition.knapsack_lowrank import build, load_table
from Pruning.taylor_prune import load_model

ROOT = Path(__file__).resolve().parents[1]


def validate(path, data, out_dir, name):
    SETTINGS.update({'datasets_dir': str(data.parent)})
    box = YOLO(str(path)).val(data='kitti.yaml', split='val', imgsz=640, batch=1, device=0, rect=False, workers=0,
                              plots=False, save_json=False, verbose=False, project=str(out_dir/'val'), name=name,
                              exist_ok=True).box
    return {'mAP50-95': float(box.map), 'mAP75': float(box.map75), 'mAP50': float(box.map50),
            'precision': float(box.mp), 'recall': float(box.mr)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/baseline/8n_float32.pt')
    parser.add_argument('--data', type=Path, default=ROOT/'PTQ/datasets/kitti')
    parser.add_argument('--plan', type=Path, required=True, help='"<layer> <option>:<rank>" per line')
    parser.add_argument('--tucker', type=Path, default=ROOT/'outputs/sensitivity_tucker2/sensitivity.csv')
    parser.add_argument('--svd', type=Path, default=ROOT/'sensitivity_results/kitti_svd_rank_sensitivity/sensitivity.csv')
    parser.add_argument('--output', type=Path, default=None, help='checkpoint path (default: next to --plan)')
    args = parser.parse_args()
    args.weights, args.data, args.plan = args.weights.resolve(), args.data.resolve(), args.plan.resolve()
    out = (args.output or args.plan.with_name(f'{args.weights.stem}_lrf.pt')).resolve()
    device = torch.device('cuda:0')

    wanted = dict(line.split() for line in args.plan.read_text(encoding='utf-8').splitlines() if line.strip())
    table = load_table(args.tucker, args.svd)
    names = {t['layer'] for t in table}
    unknown = set(wanted) - names
    if unknown:
        raise SystemExit(f'not decomposable / not scanned: {sorted(unknown)}')
    sel = []
    for t in table:
        if t['layer'] not in wanted:
            sel.append(0)
            continue
        option, rank = wanted[t['layer']].split(':')
        j = next((j for j, (o, r) in enumerate(zip(t['option'], t['rank'])) if o == option and r == int(rank)), None)
        if j is None:
            raise SystemExit(f'{t["layer"]}: no option {option} rank {rank} in the scans')
        sel.append(j)

    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
    base, _ = load_model(args.weights)
    base.to(device).eval()
    model = build(base, table, sel)
    example = torch.zeros(1, 3, 640, 640, device=device)
    base_macs, base_params = tp.utils.count_ops_and_params(base, example)
    macs, params = tp.utils.count_ops_and_params(model, example)
    with torch.no_grad():
        y0, y1 = base(example + 0.5)[0], model(example + 0.5)[0]
    layers = [{'layer': t['layer'], 'kernel': t['kernel'], 'c_in': t['c_in'], 'c_out': t['c_out'],
               'option': t['option'][j], 'rank': t['rank'][j], 'single_layer_drop_pts': t['drop'][j],
               'single_layer_MFLOPs_saved': t['saved'][j]} for t, j in zip(table, sel) if j]
    decomposition = {'method': 'Tucker-2 (one mode, truncated HOSVD) for k x k Conv blocks, truncated SVD for 1 x 1',
                     'plan': str(args.plan.relative_to(ROOT)), 'layers': layers,
                     'predicted_drop_pts': round(sum(l['single_layer_drop_pts'] for l in layers), 3)}
    model.cpu()
    torch.save({'model': model, 'train_args': ckpt.get('train_args', {}), 'date': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'decomposition': decomposition}, out)
    print(f'Saved {out}: {len(layers)} layers decomposed, params {base_params/1e6:.3f}M -> {params/1e6:.3f}M, '
          f'GFLOPs {2*base_macs/1e9:.3f} -> {2*macs/1e9:.3f}', flush=True)

    val_base = validate(args.weights, args.data, out.parent, 'baseline')
    val_lrf = validate(out, args.data, out.parent, 'lrf')
    print(f'Baseline: {val_base}\nLRF:      {val_lrf}', flush=True)
    manifest = {'source': str(args.weights.relative_to(ROOT)), 'checkpoint': out.name, **decomposition,
                'params': {'base': base_params, 'lrf': params, 'saved_%': 100 * (1 - params / base_params)},
                'GFLOPs': {'base': 2 * base_macs / 1e9, 'lrf': 2 * macs / 1e9, 'saved_%': 100 * (1 - macs / base_macs)},
                'output_rel_diff_constant_input': float((y1 - y0).norm() / y0.norm()),
                'val_batch1_conf0.001': {'baseline': val_base, 'lrf': val_lrf,
                                         'drop_mAP50-95_pts': 100 * (val_base['mAP50-95'] - val_lrf['mAP50-95'])}}
    (out.parent/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
