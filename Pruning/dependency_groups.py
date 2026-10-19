"""Torch-Pruning DepGraph dependency groups of YOLOv8n-KITTI (no pruning, no scoring).

The graph is built exactly as in Pruning/taylor_prune.py: C2f.cv1 split into cv0 (first half, straight to the
concat) and cv1 (second half, into the bottlenecks), and the Detect output convolutions (cv2.*.2, cv3.*.2) and the
DFL projection ignored, so the groups are the ones that pruner ranks and prunes. With --no-split C2f stays intact
and its cv1 output is one group whose channels the chunk couples to the bottleneck chain.

A group is every layer whose channels must be removed together: the root convolution's output channels, the BN
after it, the input channels of every consumer convolution (through concat / chunk / residual add), and any other
convolution whose output is added to the same tensor (residual coupling).

Outputs (in --output):
  groups.csv          one row per group: root, channels, coupled output convs, consumer convs, conv MFLOPs touched
  group_members.csv   one row per (group, layer, pruning function, channel count)
  groups.txt          Torch-Pruning's own printout of every group
  manifest.json       settings and totals
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch_pruning as tp
from torch import nn
from ultralytics.nn.modules import C2f

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Pruning.taylor_prune import SPLIT_INDICES, load_model, merged_name, split_c2f

ROOT = Path(__file__).resolve().parents[1]
OUT_CH, IN_CH = tp.pruner.function.prune_conv_out_channels, tp.pruner.function.prune_conv_in_channels


def conv_mflops(model, example):
    """2*MACs of every Conv2d at the example resolution (bias not counted)."""
    flops, handles = {}, []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(lambda mod, _i, out: flops.__setitem__(
                mod, 2 * mod.weight[0].numel() * out.shape[1] * out.shape[2] * out.shape[3] / 1e6)))
    with torch.no_grad():
        model(example)
    for h in handles:
        h.remove()
    return flops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/baseline/8n_float32.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/dependency_groups')
    parser.add_argument('--no-split', action='store_true', help='keep C2f.cv1 as one convolution')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    model, _ = load_model(args.weights)
    for p in model.parameters():
        p.requires_grad_(True)     # DepGraph traces autograd; frozen leading convs would be left out of the graph
    example = torch.zeros(1, 3, 640, 640)
    if not args.no_split:
        SPLIT_INDICES.update(str(i) for i, m in enumerate(model.model) if isinstance(m, C2f))
        split_c2f(model)
    names = {m: n for n, m in model.named_modules()}
    flops = conv_mflops(model, example)
    total_mflops = sum(flops.values())

    head = model.model[-1]
    ignored = [s[-1] for s in head.cv2] + [s[-1] for s in head.cv3] + [head.dfl]
    ignored_convs = {m for layer in ignored for m in layer.modules() if isinstance(m, nn.Conv2d)}
    # As in BasePruner: ignored layers stay in the traced graph (dropping them would wire the cv*.1 convolutions
    # straight into the output concat and DFL) and are only excluded when the groups are collected.
    graph = tp.DependencyGraph().build_dependency(model, example_inputs=example, verbose=False)
    groups = list(graph.get_all_groups(ignored_layers=ignored, root_module_types=[nn.Conv2d]))

    rows, members, text = [], [], []
    for g, group in enumerate(groups):
        root = group[0].dep.target.module
        out_convs, in_convs, touched = [], [], set()
        for item in group:
            dep, idxs = item.dep, item.idxs
            layer = dep.target.module
            fn = getattr(dep.pruning_fn, '__name__', type(dep.handler).__name__)
            members.append({'group': g, 'layer': names.get(layer, dep.target.name), 'type': type(layer).__name__,
                            'pruning_fn': fn, 'channels': len(idxs)})
            if isinstance(layer, nn.Conv2d):
                if dep.pruning_fn == OUT_CH:
                    out_convs.append(names[layer])
                elif dep.pruning_fn == IN_CH:
                    in_convs.append(names[layer])
                touched.add(layer)
        group_flops = sum(flops[m] for m in touched)
        rows.append({'group': g, 'root': names[root], 'channels': len(group[0].idxs),
                     'coupled_out_convs': ' + '.join(out_convs), 'n_out_convs': len(out_convs),
                     'consumer_in_convs': ' + '.join(in_convs), 'n_in_convs': len(in_convs),
                     'residual_coupled': len(out_convs) > 1, 'conv_MFLOPs_touched': round(group_flops, 2),
                     'conv_MFLOPs_touched_%': round(100 * group_flops / total_mflops, 2)})
        text.append(f'===== Group {g}: root {names[root]}, {len(group[0].idxs)} channels =====\n{group}\n')

    for path, table in ((args.output/'groups.csv', rows), (args.output/'group_members.csv', members)):
        with path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    (args.output/'groups.txt').write_text('\n'.join(text), encoding='utf-8')

    # Every non-ignored conv must appear as an output member of exactly one group.
    prunable = [m for m in flops if m not in ignored_convs]
    owners = {}
    for r in rows:
        for name in r['coupled_out_convs'].split(' + '):
            owners.setdefault(name, []).append(r['group'])
    missing = [names[m] for m in prunable if names[m] not in owners]
    duplicated = {n: g for n, g in owners.items() if len(g) > 1}
    manifest = {'source': str(args.weights.relative_to(ROOT)), 'tool': f'torch-pruning {tp.__version__}',
                'c2f_cv1_split': not args.no_split, 'root_module_types': ['Conv2d'],
                'ignored': ['model.22.cv2.*.2', 'model.22.cv3.*.2', 'model.22.dfl'],
                'conv2d_total': len(flops), 'conv2d_ignored': len(ignored_convs), 'conv2d_prunable': len(prunable),
                'groups': len(rows), 'single_conv_groups': sum(r['n_out_convs'] == 1 for r in rows),
                'residual_coupled_groups': [r['group'] for r in rows if r['residual_coupled']],
                'prunable_channels': sum(r['channels'] for r in rows), 'conv_MFLOPs_total': round(total_mflops, 2),
                'check_convs_without_group': missing, 'check_convs_in_several_groups': duplicated}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'ignored'}, indent=1))
    if missing or duplicated:
        raise RuntimeError(f'Group coverage check failed: missing {missing}, duplicated {duplicated}')


if __name__ == '__main__':
    main()
