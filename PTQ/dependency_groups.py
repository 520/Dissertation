"""Torch-Pruning dependency-graph groups of YOLOv8n-KITTI (8n_float32.pt), unfused (Conv + BN).

A group is the set of layers whose channels must be removed together when output channels of its root
convolution are pruned: the root's BN, every consumer's input channels, and, through residual adds and
concatenations, other convolutions' output channels. Built twice:
  original/  the checkpoint as is (C2f splits cv1 with chunk(2))
  c2f_v2/    C2f -> C2f_v2 (cv1 split into cv0/cv1, numerically identical), as used by taylor_global_prune.py
Detect output convolutions (box/class channels) and the fixed DFL projection are ignored as roots.

Outputs per variant:
  groups.csv        one row per group: root, channels, coupled out-channel convs, input-affected convs, BN
  group_items.csv   one row per dependency: group, layer, type, out/in, number of indices, root index span
  groups.txt        Torch-Pruning's own printout of every group
  manifest.json     model, versions, ignored layers, counts
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch_pruning as tp
import ultralytics
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2f_v2 import replace_c2f  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = payload.get('ema') or payload['model'] if isinstance(payload, dict) else payload
    model = model.float().eval()
    for p in model.parameters():
        p.requires_grad_(True)      # Torch-Pruning traces the autograd graph
    return model


def build_groups(model, out_dir, source):
    out_dir.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 3, 640, 640)
    detect = model.model[-1]
    ignored = [seq[-1] for seq in (*detect.cv2, *detect.cv3)] + [detect.dfl.conv]
    graph = tp.DependencyGraph().build_dependency(model, example_inputs=example)
    names = {m: n for n, m in model.named_modules()}
    groups = list(graph.get_all_groups(ignored_layers=ignored, root_module_types=[nn.Conv2d]))

    group_rows, item_rows, text = [], [], []
    covered = set()
    for g, group in enumerate(groups):
        root = group[0].dep.target.module
        out_convs, in_convs, bns = [], [], []
        for item in group:
            layer, handler = item.dep.target.module, item.dep.handler
            is_out = graph.is_out_channel_pruning_fn(handler)
            name = names.get(layer, item.dep.target.name)
            if isinstance(layer, nn.Conv2d):
                (out_convs if is_out else in_convs).append(name)
                if is_out:
                    covered.add(layer)
            elif isinstance(layer, nn.BatchNorm2d):
                bns.append(name)
            roots = [r for r in item.root_idxs if r is not None]
            item_rows.append({'group': g, 'root': names[root], 'layer': name, 'type': type(layer).__name__,
                              'pruning': 'out' if is_out else 'in', 'handler': handler.__name__,
                              'indices': len(item.idxs),
                              'root_index_min': min(roots) if roots else '', 'root_index_max': max(roots) if roots else ''})
        kind = ('residual/concat coupled' if len(out_convs) > 1 else 'single conv')
        group_rows.append({'group': g, 'root': names[root], 'channels': graph.get_out_channels(root),
                           'coupling': kind, 'out_convs': len(out_convs), 'out_conv_layers': ' '.join(out_convs),
                           'in_convs': len(in_convs), 'in_conv_layers': ' '.join(in_convs),
                           'batchnorms': ' '.join(bns), 'dependencies': len(group)})
        text.append(f'===== group {g}: root {names[root]} ({graph.get_out_channels(root)} channels) =====\n{group}\n')

    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    uncovered = [names[m] for m in convs if m not in covered and m not in ignored]
    write_csv(out_dir/'groups.csv', group_rows)
    write_csv(out_dir/'group_items.csv', item_rows)
    (out_dir/'groups.txt').write_text('\n'.join(text), encoding='utf-8')
    manifest = {'source': str(source), 'torch': torch.__version__, 'torch_pruning': tp.__version__,
                'ultralytics': ultralytics.__version__, 'convs': len(convs), 'groups': len(groups),
                'coupled_groups': sum(r['out_convs'] > 1 for r in group_rows),
                'group_channels_total': sum(r['channels'] for r in group_rows),
                'conv_out_channels_in_groups': sum(graph.get_out_channels(m) for m in covered),
                'ignored_layers': [names[m] for m in ignored], 'convs_not_in_any_group': uncovered}
    (out_dir/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest, group_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/baseline/8n_float32.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'models/pruning/dependency_groups')
    args = parser.parse_args()

    for variant in ('original', 'c2f_v2'):
        model = load(args.weights)
        if variant == 'c2f_v2':
            with torch.no_grad():
                before = model(torch.zeros(1, 3, 640, 640))[0]
                replace_c2f(model)
                diff = float((model(torch.zeros(1, 3, 640, 640))[0] - before).abs().max())
            print(f'C2f_v2 max output difference: {diff:.3g}')
        manifest, rows = build_groups(model, args.output/variant, args.weights)
        print(f"[{variant}] {manifest['convs']} convs -> {manifest['groups']} groups "
              f"({manifest['coupled_groups']} coupled), {manifest['group_channels_total']} prunable group channels; "
              f"not in any group: {manifest['convs_not_in_any_group']}")
        for r in rows:
            if r['out_convs'] > 1:
                print(f"  group {r['group']:2d} {r['root']:<22} {r['channels']:4d} ch  couples {r['out_conv_layers']}")
    print(f'Done: {args.output}')


if __name__ == '__main__':
    main()
