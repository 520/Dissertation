"""Prune YOLOv8n-KITTI with a fixed per-group plan picked from the 8-channel L2 sensitivity scan.

The plan (PLAN below) gives, for each DepGraph group of Pruning/sensitivity_l2.py (C2f.cv1 split into cv0/cv1),
how many output channels of its root convolution to remove; the group indices and roots are those of
outputs/sensitivity_l2_ch8. All groups are pruned together, starting from the unpruned model, removing the
lowest-importance channels by GroupMagnitudeImportance(p=2) taken on the unpruned model, exactly as in the scan and
in Pruning/pareto_search.py.

C2f blocks whose two cv1 halves end with the same width are merged back into a standard Ultralytics C2f. Blocks
pruned unequally (here model.8: cv1 coupled with m.0.cv2, and model.18.cv1) cannot be expressed as a C2f, whose
forward chunks cv1 into two equal halves, so they stay as Pruning.taylor_prune.C2fSplit (numerically the same
operator; the checkpoint then needs this repository on the import path to load).

Outputs (in --output):
  <stem>.pt        pruned checkpoint ({'model', 'train_args', ...}), ready for Pruning/finetune.py
  layers.csv       every convolution: channels before / after
  manifest.json    plan, params, GFLOPs

Example:
    .venv/Scripts/python.exe -m Pruning.plan_prune
    .venv/Scripts/python.exe -m Pruning.finetune --weights models/pruning/l2ch8_plan/8n_float32_l2ch8_plan.pt --epochs 10 --name kitti_l2ch8_plan_ft
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault('YOLO_VERBOSE', 'False')

import torch
import torch_pruning as tp
from torch import nn
from ultralytics.nn.modules import Detect

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Pruning.sensitivity_l2 import OUT_CH, groups_of
from Pruning.taylor_prune import C2fSplit, load_model, split_c2f

ROOT = Path(__file__).resolve().parents[1]

# (group, root convolution, channels before, channels removed), from outputs/sensitivity_l2_ch8.
PLAN = [
    (19, 'model.4.m.1.cv1.conv', 32, 8),
    (16, 'model.5.conv', 128, 16),
    (11, 'model.7.conv', 256, 40),
    (8, 'model.8.cv2.conv', 256, 16),
    (10, 'model.8.m.0.cv1.conv', 128, 16),
    (9, 'model.8.m.0.cv2.conv', 128, 16),          # coupled with model.8.cv1 (second half) by the residual
    (41, 'model.16.conv', 64, 32),
    (35, 'model.18.cv1.conv', 64, 40),
    (32, 'model.18.cv2.conv', 128, 40),
    (34, 'model.18.m.0.cv1.conv', 64, 16),
    (33, 'model.18.m.0.cv2.conv', 64, 8),
    (31, 'model.19.conv', 128, 120),
    (48, 'model.21.cv0.conv', 128, 120),
    (5, 'model.21.cv1.conv', 128, 120),
    (2, 'model.21.cv2.conv', 256, 248),
    (4, 'model.21.m.0.cv1.conv', 128, 120),
    (3, 'model.21.m.0.cv2.conv', 128, 120),
    (58, 'model.22.cv2.0.0.conv', 64, 8),
    (54, 'model.22.cv2.2.0.conv', 64, 56),
    (50, 'model.22.cv3.1.0.conv', 64, 16),
    (1, 'model.22.cv3.2.0.conv', 64, 56),
    (0, 'model.22.cv3.2.1.conv', 64, 8),
]


def merge_equal_c2f(model):
    """Merge every C2fSplit whose halves have the same width; return the indices left split."""
    left = []
    for i, m in enumerate(model.model):
        if isinstance(m, C2fSplit):
            if m.cv0.conv.out_channels == m.cv1.conv.out_channels:
                model.model[i] = m.merge()
            else:
                left.append(i)
    return left


def convs(model):
    return {n: m.out_channels for n, m in model.named_modules() if isinstance(m, nn.Conv2d)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/baseline/8n_float32.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'models/pruning/l2ch8_plan')
    parser.add_argument('--stem', default='8n_float32_l2ch8_plan')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model, ckpt = load_model(args.weights)
    saved_args = getattr(model, 'args', None)
    split_c2f(model)
    for p in model.parameters():
        p.requires_grad_(True)     # DepGraph traces autograd; frozen leading convs would be left out of the graph
    model.to(device).eval()
    example = torch.zeros(1, 3, 640, 640, device=device)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example)
    before = convs(model)

    graph, groups = groups_of(model, example)
    if len(groups) != 59:
        raise RuntimeError(f'{len(groups)} groups, the scan has 59')
    names = {m: n for n, m in model.named_modules()}
    importance = tp.importance.GroupMagnitudeImportance(p=2, group_reduction='mean', normalizer=None)
    todo = []
    for g, root, channels, n in PLAN:
        group = groups[g]
        module = group[0].dep.target.module
        if names[module] != root or len(group[0].idxs) != channels:
            raise RuntimeError(f'Group {g}: expected {root} ({channels}), got {names[module]} ({len(group[0].idxs)})')
        todo.append((module, torch.argsort(importance(group))[:n].tolist()))
    for module, idxs in todo:      # importance of every group taken before anything is pruned
        graph.get_pruning_group(module, OUT_CH, idxs).prune()
    for m in model.modules():
        if isinstance(m, Detect):
            m.shape = None                         # rebuild anchors for the pruned model

    for g, root, channels, n in PLAN:
        got = dict(model.named_modules())[root].out_channels
        if got != channels - n:
            raise RuntimeError(f'Group {g} {root}: {got} channels left, expected {channels - n}')
    after_split = convs(model)
    macs, params = tp.utils.count_ops_and_params(model, example)
    split_left = merge_equal_c2f(model)
    with torch.no_grad():
        out_shape = tuple(model(example)[0].shape)

    layers = [{'layer': n, 'channels_before': before[n], 'channels_after': after_split[n],
               'pruned': before[n] - after_split[n]} for n in before]
    with (args.output/'layers.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(layers[0]))
        writer.writeheader()
        writer.writerows(layers)

    model.args = saved_args
    model.cpu()
    out = args.output/f'{args.stem}.pt'
    torch.save({'model': model, 'train_args': ckpt.get('train_args', {}), 'date': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'pruning': {'method': 'Torch-Pruning DepGraph, group L2 magnitude, fixed per-group plan',
                            'plan': [{'group': g, 'root': r, 'channels': c, 'removed': n} for g, r, c, n in PLAN]}}, out)
    manifest = {'source': str(args.weights.relative_to(ROOT)), 'checkpoint': out.name,
                'scan': 'outputs/sensitivity_l2_ch8', 'tool': f'torch-pruning {version("torch-pruning")}',
                'importance': 'GroupMagnitudeImportance(p=2, group_reduction=mean, normalizer=None), unpruned model',
                'plan': [{'group': g, 'root': r, 'channels': c, 'removed': n, 'kept': c - n} for g, r, c, n in PLAN],
                'c2f_left_split': [f'model.{i}' for i in split_left],
                'params': {'before': base_params, 'after': params}, 'GFLOPs': {'before': 2*base_macs/1e9, 'after': 2*macs/1e9},
                'GFLOPs_saved_%': 100 * (1 - macs / base_macs), 'output_shape': out_shape}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Saved {out}: params {base_params/1e6:.3f}M -> {params/1e6:.3f}M, GFLOPs {2*base_macs/1e9:.2f} -> '
          f'{2*macs/1e9:.2f} ({manifest["GFLOPs_saved_%"]:.1f}% saved), output {out_shape}, '
          f'C2f left split: {manifest["c2f_left_split"]}', flush=True)


if __name__ == '__main__':
    main()
