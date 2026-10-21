"""Re-score chosen candidates of Pruning/pareto_search.py with several adaptive-BN image sets.

Each plan (from outputs/pareto/candidates.csv) is rebuilt from the unpruned model as in pareto_search.py and scored on
the held-out KITTI val images after adaptive BN on --bn-images training-augmented images drawn with each of --seeds.
The unpruned model gets the same treatment, so every drop is against the reference from the same BN image set.

Output: <output>/recheck.csv, one row per plan with the drop for every seed, the mean and the spread.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault('YOLO_VERBOSE', 'False')

import numpy as np
import torch
import torch_pruning as tp

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Pruning.pareto_search import build
from Pruning.sensitivity_l2 import Evaluator, adapt_bn, bn_images
from Pruning.taylor_prune import hyp_from, load_model, split_c2f

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/baseline/8n_float32.pt')
    parser.add_argument('--data', type=Path, default=ROOT/'PTQ/datasets/kitti')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/pareto')
    parser.add_argument('--ids', nargs='+', required=True, help='candidate ids from candidates.csv')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--bn-images', type=int, default=512)
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')

    candidates = {r['id']: r for r in csv.DictReader((args.output/'candidates.csv').open(encoding='utf-8-sig'))}
    missing = [i for i in args.ids if i not in candidates]
    if missing:
        raise SystemExit(f'Unknown candidate ids: {missing}')
    base, _ = load_model(args.weights)
    split_c2f(base)
    for p in base.parameters():
        p.requires_grad_(True)
    base.to(device).eval()
    example = torch.zeros(1, 3, 640, 640, device=device)
    importance = tp.importance.GroupMagnitudeImportance(p=2, group_reduction='mean', normalizer=None)
    evaluate = Evaluator(args.output/'kitti_val_heldout.yaml', args.batch, args.output/'val_heldout')
    models = {cid: build(base, example, importance, [int(c) for c in candidates[cid]['plan']]) for cid in args.ids}

    drops = {cid: {} for cid in args.ids}
    for seed in args.seeds:
        calib = bn_images(args.data, args.bn_images, seed, hyp_from(base.args))
        reference = deepcopy(base)
        adapt_bn(reference, calib, args.batch, device)
        ref_map = 100 * evaluate(reference)['mAP50-95']
        print(f'seed {seed}: unpruned {ref_map:.2f}', flush=True)
        for cid, model in models.items():
            m = deepcopy(model)
            adapt_bn(m, calib, args.batch, device)
            drops[cid][seed] = ref_map - 100 * evaluate(m)['mAP50-95']
            print(f'  {cid:<20} drop {drops[cid][seed]:.2f}', flush=True)

    rows = []
    for cid in args.ids:
        d = np.array([drops[cid][s] for s in args.seeds])
        rows.append({'id': cid, 'saved_%': candidates[cid]['saved_%'], 'drop_heldout_seed42': candidates[cid]['drop_heldout'],
                     **{f'drop_seed{s}': round(drops[cid][s], 3) for s in args.seeds},
                     'drop_mean': round(float(d.mean()), 3), 'drop_spread': round(float(d.max() - d.min()), 3)})
    with (args.output/'recheck.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['id']:<20} saved {float(r['saved_%']):5.1f}%  drop seed42 {float(r['drop_heldout_seed42']):5.2f}  "
              f"mean over seeds {r['drop_mean']:5.2f}  spread {r['drop_spread']:.2f}")


if __name__ == '__main__':
    main()
