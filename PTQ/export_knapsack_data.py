"""Export the Taylor-1 sensitivity scan as knapsack input (knapsack_data.json, next to sensitivity.csv).

Per group: root, coupled members, channels, and one option per distinct pruned-channel count with its measured
mAP50-95 drop. Per convolution: kernel area, groups, output pixels at 640x640, in/out channels, bias, BN, and how
many of its input/output channels each group removes per pruned root channel. With that table the MACs and
params of any combination of group choices are exact, rather than a sum of single-group savings.
The formula is checked here against Torch-Pruning's own counts on every single-group trial.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch_pruning as tp
from torch import nn

from taylor_global_prune import ROOT, dependency_groups, load_unfused, replace_c2f

P5 = ('model.19.', 'model.21.', 'model.22.cv2.2.', 'model.22.cv3.2.')


def cost(convs, removed):
    """(params, MACs) of the network with removed[g] root channels taken out of each group g."""
    params = macs = 0.0
    for c in convs:
        cout = c['cout'] - sum(f * removed.get(g, 0) for g, f in c['out'])
        cin = c['cin'] - sum(f * removed.get(g, 0) for g, f in c['in'])
        w = c['k'] * cin * cout / c['groups']
        params += w + (cout if c['bias'] else 0) + (2 * cout if c['bn'] else 0)
        macs += w * c['hw'] + (cout * c['hw'] if c['bias'] else 0) + (2 * cout * c['hw'] if c['bn'] else 0)
    # Torch-Pruning also counts SPPF max-pool and upsample outputs (~0.03% of MACs); not modelled here.
    return params, macs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/baseline/8n_float32.pt')
    parser.add_argument('--results', type=Path, default=ROOT/'sensitivity_results/kitti_taylor1_group_sensitivity')
    args = parser.parse_args()
    model = load_unfused(args.weights, torch.device('cpu'))
    replace_c2f(model)
    example = torch.zeros(1, 3, 640, 640)
    names = {m: n for n, m in model.named_modules()}

    hw = {}
    hooks = [m.register_forward_hook(lambda mod, i, o: hw.__setitem__(mod, o.shape[2] * o.shape[3]))
             for m in model.modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        model(example)
    for h in hooks:
        h.remove()

    graph, groups, _ = dependency_groups(model, example)
    convs = {}
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            name = names[m]
            bn_name = name[:-len('conv')] + 'bn' if name.endswith('.conv') else None
            convs[m] = {'name': name, 'k': m.kernel_size[0] * m.kernel_size[1], 'groups': m.groups,
                        'cin': m.in_channels, 'cout': m.out_channels, 'hw': hw[m], 'bias': m.bias is not None,
                        'bn': bool(bn_name and isinstance(dict(model.named_modules()).get(bn_name), nn.BatchNorm2d)),
                        'out': [], 'in': []}
    group_info = []
    for g, group in enumerate(groups):
        root = group[0].dep.target.module
        channels = graph.get_out_channels(root)
        members = []
        for item in group:
            layer = item.dep.target.module
            if isinstance(layer, nn.Conv2d):
                side = 'out' if graph.is_out_channel_pruning_fn(item.dep.handler) else 'in'
                convs[layer][side].append([g, len(item.idxs) / channels])
                if side == 'out':
                    members.append(names[layer])
        group_info.append({'id': g, 'root': names[root], 'channels': channels, 'members': members,
                           'p5': names[root].startswith(P5)})
    conv_list = list(convs.values())
    base_macs_tp, base_params_tp = tp.utils.count_ops_and_params(model, example)
    base_params, base_macs = cost(conv_list, {})

    # Options from the scan, one per distinct channel count; check the formula on every trial.
    by_root = {gi['root']: gi for gi in group_info}
    worst = {'params': 0.0, 'macs': 0.0}
    with (args.results/'sensitivity.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    for gi in group_info:
        gi['options'] = [{'ratio': 0.0, 'n': 0, 'drop': 0.0}]
    for r in sorted(rows, key=lambda r: float(r['ratio'])):
        gi = by_root[r['root']]
        n = int(r['pruned'])
        p, m = cost(conv_list, {gi['id']: n})
        worst['params'] = max(worst['params'], abs(p - int(r['params'])) / base_params_tp)
        worst['macs'] = max(worst['macs'], abs(m - float(r['gmacs']) * 1e9) / base_macs_tp)
        if all(o['n'] != n for o in gi['options']):
            gi['options'].append({'ratio': float(r['ratio']), 'n': n, 'drop': float(r['drop_mAP50-95'])})
    manifest = json.loads((args.results/'manifest.json').read_text(encoding='utf-8'))
    data = {'source': str(args.weights), 'reference_mAP50_95': manifest['reference_unpruned_adaptive_bn']['mAP50-95'],
            'val_images': manifest['val_subset']['images'], 'trials': len(rows),
            'base': {'params': base_params, 'macs': base_macs, 'params_tp': base_params_tp, 'macs_tp': base_macs_tp},
            'formula_max_rel_error_vs_torch_pruning': worst,
            'p5_joint': [{'ratio': 0.9, 'drop_full_val': 3.08}, {'ratio': 0.95, 'drop_full_val': 5.98}],
            'groups': group_info,
            'convs': [{k: c[k] for k in ('name', 'k', 'groups', 'cin', 'cout', 'hw', 'bias', 'bn', 'out', 'in')}
                      for c in conv_list]}
    out = args.results/'knapsack_data.json'
    out.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
    print(f'base params {base_params:.0f} (tp {base_params_tp}), MACs {base_macs/1e9:.4f} G (tp {base_macs_tp/1e9:.4f} G)')
    print(f'max relative error vs Torch-Pruning over {len(rows)} trials: {worst}')
    print(f'{len(group_info)} groups, {len(conv_list)} convs, {sum(len(g["options"]) for g in group_info)} options '
          f'-> {out} ({out.stat().st_size/1024:.0f} KB)')


if __name__ == '__main__':
    main()
