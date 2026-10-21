"""Knapsack over the 8-channel-step sensitivity scan: best plan per drop budget, evaluated for real.

Options per group are the scan's removal counts (0, 8, 16, ... up to channels - 8) from
outputs/sensitivity_l2_ch8/sensitivity.csv; cost = single-group mAP50-95 drop (points, cumulative max over the
removal count, negative counted as 0), value = single-group MFLOPs saved. For every budget in --budgets the best plan
(top 1, cost resolution 0.01 point) is pruned for real from the unpruned model as in Pruning/budget_curve.py: real
FLOPs, adaptive BN (512 training-augmented images, seed 42), mAP50-95 on the 500-image search subset and the 996
held-out KITTI val images. Identical plans share one evaluation.

Outputs (in --output):
  plans.csv     one row per budget: removal counts, predicted and real drop and FLOPs
  plans.json    the same plus per-group names, channels and single-group drops, for the slider page
  manifest.json settings and reference
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

os.environ.setdefault('YOLO_VERBOSE', 'False')

import numpy as np
import torch
import torch_pruning as tp

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Pruning.budget_curve import evaluators
from Pruning.sensitivity_l2 import OUT_CH, adapt_bn, bn_images, groups_of
from Pruning.taylor_prune import hyp_from, load_model, split_c2f
from ultralytics.nn.modules import Detect

ROOT = Path(__file__).resolve().parents[1]
UNIT = 0.01


def load_options(path):
    rows = list(csv.DictReader(path.open(encoding='utf-8-sig')))
    base = json.loads((path.parent/'manifest.json').read_text())['base_GFLOPs']
    groups = {}
    for r in rows:
        g = groups.setdefault(int(r['group']), {'group': int(r['group']), 'root': r['root'].removesuffix('.conv'),
                                                'channels': int(r['channels']),
                                                'coupled': r['coupled_out_convs'].count(' + '), 'points': {0: (0.0, 0.0)}})
        g['points'][int(r['removed'])] = (100 * float(r['drop_mAP50-95']), 1000 * (base - float(r['GFLOPs'])))
    table = []
    for k in sorted(groups):
        g = groups[k]
        removed = sorted(g['points'])
        run, drop = 0.0, []
        for n in removed:
            run = max(run, max(0.0, g['points'][n][0]))
            drop.append(round(run, 4))
        table.append({**{k2: g[k2] for k2 in ('group', 'root', 'channels', 'coupled')}, 'removed': removed,
                      'drop': drop, 'raw_drop': [round(g['points'][n][0], 4) for n in removed],
                      'saved': [round(g['points'][n][1], 2) for n in removed]})
    return table, base


def best_plans(table, budgets):
    """Multiple-choice knapsack, top 1 per budget; returns the option index per group for each budget."""
    B = int(round(max(budgets) / UNIT))
    prev = np.zeros(B + 1)
    choices, costs = [], []
    for t in table:
        cost = [int(round(d / UNIT)) for d in t['drop']]
        cur, pick = np.full(B + 1, -np.inf), np.zeros(B + 1, np.int16)
        for j, (c, v) in enumerate(zip(cost, t['saved'])):
            if c > B:
                continue
            cand = np.full(B + 1, -np.inf)
            cand[c:] = prev[:B + 1 - c] + v
            better = cand > cur
            cur[better], pick[better] = cand[better], j
        choices.append(pick)
        costs.append(cost)
        prev = cur
    plans = {}
    for budget in budgets:
        b, sel = int(round(budget / UNIT)), [0] * len(table)
        for i in range(len(table) - 1, -1, -1):
            sel[i] = int(choices[i][b])
            b -= costs[i][sel[i]]
        plans[budget] = sel
    return plans


def build(base, example, importance, removals):
    model = deepcopy(base)
    graph, groups = groups_of(model, example)
    todo = [(groups[g][0].dep.target.module, torch.argsort(importance(groups[g]))[:n].tolist())
            for g, n in enumerate(removals) if n]
    for root, idxs in todo:
        graph.get_pruning_group(root, OUT_CH, idxs).prune()
    for m in model.modules():
        if isinstance(m, Detect):
            m.shape = None
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/baseline/8n_float32.pt')
    parser.add_argument('--data', type=Path, default=ROOT/'PTQ/datasets/kitti')
    parser.add_argument('--scan', type=Path, default=ROOT/'outputs/sensitivity_l2_ch8/sensitivity.csv')
    parser.add_argument('--search-subset', type=Path, default=ROOT/'outputs/sensitivity_l2_ch8/val_subset.txt')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/knapsack_ch8')
    parser.add_argument('--budgets', type=float, nargs='+', default=[round(0.1 * k, 1) for k in range(0, 51)])
    parser.add_argument('--bn-images', type=int, default=512)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    begin = time.perf_counter()

    table, base_gflops = load_options(args.scan)
    plans = best_plans(table, args.budgets)
    distinct = list(dict.fromkeys(tuple(p) for p in plans.values()))
    print(f'{len(args.budgets)} budgets -> {len(distinct)} distinct plans', flush=True)

    base, _ = load_model(args.weights)
    split_c2f(base)
    for p in base.parameters():
        p.requires_grad_(True)
    base.to(device).eval()
    example = torch.zeros(1, 3, 640, 640, device=device)
    base_macs, base_params = tp.utils.count_ops_and_params(base, example)
    importance = tp.importance.GroupMagnitudeImportance(p=2, group_reduction='mean', normalizer=None)
    evaluate, n_search, n_held = evaluators(args.data, args.search_subset, args.output, args.batch)
    calib = bn_images(args.data, args.bn_images, args.seed, hyp_from(base.args))
    ref = deepcopy(base)
    adapt_bn(ref, calib, args.batch, device)
    reference = {'GFLOPs': 2 * base_macs / 1e9, 'params': base_params,
                 'mAP_search': 100 * evaluate['search'](ref)['mAP50-95'],
                 'mAP_heldout': 100 * evaluate['heldout'](ref)['mAP50-95']}
    del ref
    print(f'Unpruned after adaptive BN: {reference}', flush=True)

    measured = {}
    for k, sel in enumerate(distinct):
        start = time.perf_counter()
        removals = [table[i]['removed'][j] for i, j in enumerate(sel)]
        if not any(removals):
            measured[sel] = {'real_GFLOPs': reference['GFLOPs'], 'params': base_params,
                             'mAP_search': reference['mAP_search'], 'mAP_heldout': reference['mAP_heldout']}
            continue
        model = build(base, example, importance, removals)
        macs, params = tp.utils.count_ops_and_params(model, example)
        adapt_bn(model, calib, args.batch, device)
        measured[sel] = {'real_GFLOPs': 2 * macs / 1e9, 'params': params,
                         'mAP_search': 100 * evaluate['search'](model)['mAP50-95'],
                         'mAP_heldout': 100 * evaluate['heldout'](model)['mAP50-95']}
        del model
        print(f'[{k + 1}/{len(distinct)}] saved {100 * (1 - macs / base_macs):5.2f}%  held-out drop '
              f'{reference["mAP_heldout"] - measured[sel]["mAP_heldout"]:5.2f}  ({time.perf_counter() - start:.1f}s)', flush=True)

    rows = []
    for budget in args.budgets:
        sel = tuple(plans[budget])
        m = measured[sel]
        removals = [table[i]['removed'][j] for i, j in enumerate(sel)]
        rows.append({'budget': budget, 'removed': removals, 'groups_pruned': sum(1 for n in removals if n),
                     'channels_removed': sum(removals),
                     'pred_drop': round(sum(table[i]['drop'][j] for i, j in enumerate(sel)), 3),
                     'drop_search': round(reference['mAP_search'] - m['mAP_search'], 3),
                     'drop_heldout': round(reference['mAP_heldout'] - m['mAP_heldout'], 3),
                     'pred_saved_%': round(100 * sum(table[i]['saved'][j] for i, j in enumerate(sel)) / 1000 / base_gflops, 2),
                     'saved_%': round(100 * (1 - m['real_GFLOPs'] / reference['GFLOPs']), 2),
                     'real_GFLOPs': round(m['real_GFLOPs'], 4), 'params': m['params']})
    with (args.output/'plans.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{**r, 'removed': ' '.join(map(str, r['removed']))} for r in rows])
    (args.output/'plans.json').write_text(json.dumps({'reference': reference, 'groups': table, 'plans': rows},
                                                     separators=(',', ':')), encoding='utf-8')
    manifest = {'source': str(args.weights.relative_to(ROOT)), 'scan': str(args.scan.relative_to(ROOT)),
                'knapsack': 'top 1 per budget; options = removal counts of the scan; cost = single-group drop '
                            '(cumulative max, >= 0), value = single-group MFLOPs saved', 'unit_points': UNIT,
                'budgets': args.budgets, 'distinct_plans': len(distinct),
                'adaptive_bn': {'images': args.bn_images, 'seed': args.seed},
                'validation': {'search_images': n_search, 'heldout_images': n_held},
                'reference_unpruned_adaptive_bn': reference, 'seconds': time.perf_counter() - begin}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Done: {args.output}')


if __name__ == '__main__':
    main()
