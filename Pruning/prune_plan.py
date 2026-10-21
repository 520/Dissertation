"""Prune YOLOv8n-KITTI with one plan of Pruning/knapsack_ch8.py and save it for Pruning/finetune.py.

The plan (channels removed per DepGraph group) is taken from outputs/knapsack_ch8/plans.json by --budget and applied
exactly as in the knapsack evaluation: C2f.cv1 split into cv0/cv1, lowest group-L2 channels (importance on the
unpruned model), all groups pruned together. Split C2f blocks whose halves end up with equal widths are merged back
into standard C2f; unequal ones stay as C2fSplit (Pruning/taylor_prune.py), which Ultralytics trains and validates
like any module. The saved checkpoint is checked against the knapsack evaluation (real GFLOPs).

Outputs (in --output):
  <stem>.pt        Ultralytics-style checkpoint {'model': DetectionModel, ...}, BN as trained (no adaptive BN)
  plan.csv         the plan: per group root, channels, removed, kept
  manifest.json    plan source, params / GFLOPs before and after, check against plans.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch_pruning as tp

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Pruning.knapsack_ch8 import build
from Pruning.taylor_prune import C2fSplit, load_model, split_c2f

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/baseline/8n_float32.pt')
    parser.add_argument('--plans', type=Path, default=ROOT/'outputs/knapsack_ch8/plans.json')
    parser.add_argument('--budget', type=float, required=True)
    parser.add_argument('--output', type=Path, default=None, help='default models/pruning/ch8_b<budget>')
    args = parser.parse_args()
    tag = f'ch8_b{args.budget:g}'
    out_dir = args.output or ROOT/f'models/pruning/{tag}'
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.plans.read_text(encoding='utf-8'))
    plan = next(p for p in data['plans'] if abs(p['budget'] - args.budget) < 1e-9)
    groups = data['groups']

    model, ckpt = load_model(args.weights)
    saved_args = model.args
    split_c2f(model)
    for p in model.parameters():
        p.requires_grad_(True)
    example = torch.zeros(1, 3, 640, 640)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example)
    importance = tp.importance.GroupMagnitudeImportance(p=2, group_reduction='mean', normalizer=None)
    pruned = build(model, example, importance, plan['removed']).eval()

    merged, kept_split = [], []
    for i, m in enumerate(pruned.model):
        if isinstance(m, C2fSplit):
            if m.cv0.conv.out_channels == m.cv1.conv.out_channels:
                pruned.model[i] = m.merge()
                merged.append(i)
            else:
                m.original = []           # drop the reference to the unpruned C2f before saving
                kept_split.append(i)
    macs, params = tp.utils.count_ops_and_params(pruned, example)
    with torch.no_grad():
        out_shape = tuple(pruned(example)[0].shape)
    gflops = 2 * macs / 1e9
    if abs(gflops - plan['real_GFLOPs']) > 1e-3:
        raise RuntimeError(f'GFLOPs {gflops:.4f} differ from the knapsack evaluation {plan["real_GFLOPs"]:.4f}')

    pruned.args = saved_args
    for p in pruned.parameters():
        p.requires_grad_(False)
    stem = f'{args.weights.stem}_{tag}'
    path = out_dir/f'{stem}.pt'
    torch.save({'model': pruned, 'train_args': ckpt.get('train_args', {}), 'date': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'pruning': {'method': 'Torch-Pruning DepGraph, group L2, 8-channel knapsack plan', 'budget': args.budget,
                            'removed': plan['removed']}}, path)

    with (out_dir/'plan.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['group', 'root', 'channels', 'removed', 'kept', 'single_group_drop'])
        writer.writeheader()
        for g, n in zip(groups, plan['removed']):
            writer.writerow({'group': g['group'], 'root': g['root'], 'channels': g['channels'], 'removed': n,
                             'kept': g['channels'] - n, 'single_group_drop': g['drop'][g['removed'].index(n)]})
    manifest = {'source': str(args.weights.relative_to(ROOT)), 'plans': str(args.plans.relative_to(ROOT)),
                'budget': args.budget, 'groups_pruned': plan['groups_pruned'], 'channels_removed': plan['channels_removed'],
                'params_before': base_params, 'params_after': params, 'gflops_before': 2 * base_macs / 1e9,
                'gflops_after': gflops, 'saved_%': 100 * (1 - macs / base_macs), 'output_shape': out_shape,
                'c2f_merged_back': merged, 'c2f_left_split': kept_split,
                'knapsack_evaluation': {k: plan[k] for k in ('pred_drop', 'drop_search', 'drop_heldout', 'pred_saved_%',
                                                              'saved_%', 'real_GFLOPs')},
                'checkpoint': path.name}
    (out_dir/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Saved {path}: params {base_params/1e6:.3f}M -> {params/1e6:.3f}M, GFLOPs {2*base_macs/1e9:.3f} -> {gflops:.3f}, '
          f'C2f merged {merged}, left split {kept_split}, output {out_shape}')


if __name__ == '__main__':
    main()
