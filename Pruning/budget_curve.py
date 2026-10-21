"""Knapsack-optimal plan at evenly spaced drop budgets, evaluated for real: predicted vs real FLOPs and mAP drop.

For every budget in --budgets (default 0.1 to 5.0 mAP points in steps of 0.1) the best plan of the multiple-choice
knapsack of Pruning/pareto_search.py (cap --cap) is pruned for real from the unpruned model; real FLOPs are counted,
adaptive BN is run as in the sensitivity scan (512 training-augmented images, seed 42), and mAP50-95 is measured on the
500-image search subset and the 996 held-out KITTI val images. Budgets that give the same plan share one evaluation.

Outputs (in --output):
  curve.csv        one row per budget: plan, predicted and real FLOPs saved, predicted and real drop
  curve.png/.pdf   two line charts against the budget: drop (predicted / search / held-out) and FLOPs saved
  tradeoff.png/.pdf  FLOPs saved against mAP drop, predicted and real
  manifest.json    settings and reference
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

from Pruning.pareto_search import RATIOS, UNIT, build, kbest_knapsack, load_table
from Pruning.sensitivity_l2 import Evaluator, adapt_bn, bn_images
from Pruning.taylor_prune import KITTI, hyp_from, load_model, split_c2f

ROOT = Path(__file__).resolve().parents[1]


def evaluators(data_dir, search_list, out_dir, batch):
    search = [Path(p).resolve() for p in search_list.read_text(encoding='utf-8').split()]
    all_val = sorted((data_dir/'images/val').glob('*.png')) + sorted((data_dir/'images/val').glob('*.jpg'))
    held = [p.resolve() for p in all_val if p.resolve() not in set(search)]
    out = {}
    for name, files in (('search', search), ('heldout', held)):
        lst = out_dir/f'val_{name}.txt'
        lst.write_text('\n'.join(map(str, files)) + '\n', encoding='utf-8')
        yaml = out_dir/f'kitti_val_{name}.yaml'
        yaml.write_text(json.dumps({'path': str(data_dir.resolve()), 'train': 'images/train', 'val': str(lst.resolve()),
                                    'nc': len(KITTI), 'names': KITTI}), encoding='utf-8')
        out[name] = Evaluator(yaml, batch, out_dir/f'val_{name}')
    return out, len(search), len(held)


def plot(rows, reference, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, muted, grid = '#1f1f1e', '#6b6a66', '#e3e1dc'
    gray, blue, orange = '#8f8d87', '#2a78d6', '#d9692b'
    b = [r['budget'] for r in rows]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200)
    series = [(left, 'pred_drop', 'Predicted (sum of single-group drops)', gray, '--'),
              (left, 'drop_search', 'Real, search subset (500 images)', blue, '-'),
              (left, 'drop_heldout', 'Real, held-out val (996 images)', orange, '-'),
              (right, 'pred_saved_%', 'Predicted (sum of single-group savings)', gray, '--'),
              (right, 'saved_%', 'Real (pruned model)', blue, '-')]
    for ax, key, label, color, style in series:
        ax.plot(b, [r[key] for r in rows], style, color=color, linewidth=2, label=label)
        ax.plot(b[-1], rows[-1][key], 'o', color=color, markersize=5, markeredgecolor='white')
        ax.annotate(f'{rows[-1][key]:.1f}', (b[-1], rows[-1][key]), textcoords='offset points', xytext=(6, -3),
                    fontsize=8, color=ink)
    left.plot([0, max(b)], [0, max(b)], ':', color=grid, linewidth=1)
    left.set_ylabel('mAP50-95 drop (points)', color=ink, fontsize=9)
    left.set_title('mAP50-95 drop after adaptive BN', loc='left', fontsize=10, color=ink)
    right.set_ylabel('FLOPs saved (% of 8.149 GFLOPs)', color=ink, fontsize=9)
    right.set_title('FLOPs saved', loc='left', fontsize=10, color=ink)
    for ax in (left, right):
        ax.set_xlabel('Knapsack drop budget (points)', color=ink, fontsize=9)
        ax.grid(color=grid, linewidth=0.8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
        ax.set_xlim(0, max(b) + 0.35)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.suptitle(f'YOLOv8n-KITTI: knapsack-optimal plan per budget, predicted vs real '
                 f'(unpruned after adaptive BN: {reference["mAP_heldout"]:.2f} held-out mAP50-95)',
                 x=0.01, ha='left', fontsize=10.5, color=ink)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(out_dir/f'curve.{suffix}', facecolor='white')


def plot_tradeoff(rows, reference, out_dir):
    """FLOPs saved against mAP drop for the same plans: the additive prediction and the two real measurements."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, muted, grid = '#1f1f1e', '#6b6a66', '#e3e1dc'
    gray, blue, orange = '#8f8d87', '#2a78d6', '#d9692b'
    fig, ax = plt.subplots(figsize=(8.5, 5.6), dpi=200)
    series = [('pred_drop', 'pred_saved_%', 'Predicted (sums of single-group drops and savings)', gray, '--', 'o'),
              ('drop_search', 'saved_%', 'Real, search subset (500 images)', blue, '-', 'o'),
              ('drop_heldout', 'saved_%', 'Real, held-out val (996 images)', orange, '-', 'o')]
    for xkey, ykey, label, color, style, marker in series:
        xs, ys = [r[xkey] for r in rows], [r[ykey] for r in rows]
        ax.plot(xs, ys, style, color=color, linewidth=1.6, zorder=2)
        ax.scatter(xs, ys, s=16, color=color, edgecolor='white', linewidth=0.6, label=label, zorder=3)
    for r in rows:
        if r['budget'] in (0.5, 1.0, 2.0, 3.0, 5.0):
            ax.annotate(f"budget {r['budget']:g}", (r['drop_heldout'], r['saved_%']), textcoords='offset points',
                        xytext=(6, -11), fontsize=7.5, color=muted)
    ax.set_xlabel('mAP50-95 drop after adaptive BN (points)', color=ink, fontsize=9)
    ax.set_ylabel('FLOPs saved (% of 8.149 GFLOPs)', color=ink, fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(color=grid, linewidth=0.8)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.set_title('Knapsack-optimal plans at 50 budgets (0.1–5.0): FLOPs saved vs mAP drop', loc='left',
                 fontsize=10.5, color=ink)
    fig.text(0.01, 0.005, f'One point per plan, joined in budget order. Reference: unpruned after the same adaptive BN, '
             f'{reference["mAP_heldout"]:.2f} held-out / {reference["mAP_search"]:.2f} search.',
             fontsize=7.5, color=muted)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for suffix in ('png', 'pdf'):
        fig.savefig(out_dir/f'tradeoff.{suffix}', facecolor='white')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/baseline/8n_float32.pt')
    parser.add_argument('--data', type=Path, default=ROOT/'PTQ/datasets/kitti')
    parser.add_argument('--scan', type=Path, default=ROOT/'outputs/sensitivity_l2_eighths/sensitivity.csv')
    parser.add_argument('--search-subset', type=Path, default=ROOT/'outputs/sensitivity_l2_eighths/val_subset.txt')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/budget_curve')
    parser.add_argument('--budgets', type=float, nargs='+', default=[round(0.1 * k, 1) for k in range(1, 51)])
    parser.add_argument('--cap', type=float, default=0.875, help='maximum ratio per group')
    parser.add_argument('--bn-images', type=int, default=512)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--replot', action='store_true', help='only redraw curve.png/.pdf from curve.csv')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.replot:
        rows = [{k: (float(v) if k not in ('plan',) else v) for k, v in r.items()}
                for r in csv.DictReader((args.output/'curve.csv').open(encoding='utf-8-sig'))]
        reference = json.loads((args.output/'manifest.json').read_text())['reference_unpruned_adaptive_bn']
        plot(rows, reference, args.output)
        plot_tradeoff(rows, reference, args.output)
        return
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    begin = time.perf_counter()

    table, base_gflops = load_table(args.scan)
    plans_at, _ = kbest_knapsack(table, RATIOS.index(args.cap), int(round(max(args.budgets) / UNIT)), 1)
    plans = {b: plans_at(int(round(b / UNIT)))[0] for b in args.budgets}
    distinct = list(dict.fromkeys(''.join(map(str, p)) for p in plans.values()))
    print(f'{len(args.budgets)} budgets -> {len(distinct)} distinct plans', flush=True)

    base, _ = load_model(args.weights)
    split_c2f(base)
    for p in base.parameters():
        p.requires_grad_(True)
    base.to(device).eval()
    example = torch.zeros(1, 3, 640, 640, device=device)
    base_macs, _ = tp.utils.count_ops_and_params(base, example)
    importance = tp.importance.GroupMagnitudeImportance(p=2, group_reduction='mean', normalizer=None)
    evaluate, n_search, n_held = evaluators(args.data, args.search_subset, args.output, args.batch)
    calib = bn_images(args.data, args.bn_images, args.seed, hyp_from(base.args))
    ref = deepcopy(base)
    adapt_bn(ref, calib, args.batch, device)
    reference = {'GFLOPs': 2 * base_macs / 1e9, 'mAP_search': 100 * evaluate['search'](ref)['mAP50-95'],
                 'mAP_heldout': 100 * evaluate['heldout'](ref)['mAP50-95']}
    del ref
    print(f'Unpruned after adaptive BN: {reference}', flush=True)

    measured = {}
    for k, plan in enumerate(distinct):
        start = time.perf_counter()
        model = build(base, example, importance, [int(c) for c in plan])
        macs, params = tp.utils.count_ops_and_params(model, example)
        adapt_bn(model, calib, args.batch, device)
        measured[plan] = {'real_GFLOPs': 2 * macs / 1e9, 'params': params,
                          'mAP_search': 100 * evaluate['search'](model)['mAP50-95'],
                          'mAP_heldout': 100 * evaluate['heldout'](model)['mAP50-95']}
        del model
        m = measured[plan]
        print(f'[{k + 1}/{len(distinct)}] saved {100 * (1 - macs / base_macs):5.2f}%  held-out drop '
              f'{reference["mAP_heldout"] - m["mAP_heldout"]:5.2f}  ({time.perf_counter() - start:.1f}s)', flush=True)

    rows = []
    for b in args.budgets:
        plan = plans[b]
        key = ''.join(map(str, plan))
        m = measured[key]
        rows.append({'budget': b, 'plan': key, 'groups_pruned': sum(1 for j in plan if j),
                     'pred_drop': round(sum(t['drop'][j] for t, j in zip(table, plan)), 3),
                     'drop_search': round(reference['mAP_search'] - m['mAP_search'], 3),
                     'drop_heldout': round(reference['mAP_heldout'] - m['mAP_heldout'], 3),
                     'pred_saved_%': round(100 * sum(t['saved'][j] for t, j in zip(table, plan)) / 1000 / base_gflops, 2),
                     'saved_%': round(100 * (1 - m['real_GFLOPs'] / reference['GFLOPs']), 2),
                     'real_GFLOPs': round(m['real_GFLOPs'], 4), 'params': m['params']})
    with (args.output/'curve.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {'source': str(args.weights.relative_to(ROOT)), 'scan': str(args.scan.relative_to(ROOT)),
                'knapsack': 'best plan per budget (k=1), costs = single-group drops (cumulative max, >= 0), '
                            'values = single-group MFLOPs saved', 'cap': args.cap, 'budgets': args.budgets,
                'distinct_plans': len(distinct), 'adaptive_bn': {'images': args.bn_images, 'seed': args.seed},
                'validation': {'search_images': n_search, 'heldout_images': n_held},
                'reference_unpruned_adaptive_bn': reference, 'seconds': time.perf_counter() - begin}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    plot(rows, reference, args.output)
    plot_tradeoff(rows, reference, args.output)
    print(f'Done: {args.output}')


if __name__ == '__main__':
    main()
