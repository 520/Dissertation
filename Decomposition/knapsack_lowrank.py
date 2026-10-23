"""Knapsack over the low-rank sensitivity scans (Tucker-2 + SVD): best decomposition plan per drop budget, evaluated.

One group per convolution, exactly one option per group (multiple-choice knapsack):
  k x k Conv blocks (39): keep, Tucker-2 'out' ranks, Tucker-2 'in' ranks  (outputs/sensitivity_tucker2),
                          full Tucker-2 'both' ranks, R_in = R_out           (outputs/sensitivity_tucker2_both)
  1 x 1 convs     (24): keep, SVD ranks                                   (sensitivity_results/kitti_svd_rank_sensitivity)
The SVD rows of k x k convs are not used: they are the same decomposition as Tucker-2 'out'.
Cost = single-layer mAP50-95 drop with the trained BN ("raw", the one measured the same way in both scans), in
points, negative counted as 0 and made monotone (a rank costs at least as much as every larger rank). Value =
single-layer MFLOPs saved; options that save nothing (rank too high for the factorisation to pay off) are dropped.
For every budget in --budgets the best plan (top 1, cost resolution 0.01 point) is built for real from the original
model and evaluated on the 500-image search subset (the scans' subset) and the 996 held-out KITTI val images,
both with the trained BN (raw) and after adaptive BN (512 training-augmented images, seed 42), as
Pruning/knapsack_ch8.py. Identical plans share one evaluation.

Outputs (in --output):
  options.csv   every option of every group: cost, value, source
  plans.csv     one row per budget: chosen rank per layer, predicted and real drop and FLOPs
  plans.json    the same plus the option table
  curve.png/pdf predicted vs real drop and FLOPs saved per budget
  manifest.json settings and references
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

import torch
import torch_pruning as tp
from torch import nn

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Decomposition.tucker2_sensitivity import TuckerConv, decompose, set_module
from Pruning.budget_curve import evaluators
from Pruning.knapsack_ch8 import best_plans
from Pruning.sensitivity_l2 import adapt_bn, bn_images
from Pruning.taylor_prune import hyp_from, load_model

ROOT = Path(__file__).resolve().parents[1]
HOOI_ITERS = 10                                       # as the Tucker-2 scan (only matters for 'both')


def monotone(points):
    """points: [(rank, raw drop pts, MFLOPs saved, label)] -> same with cost = max(0, drops at this and larger ranks)."""
    out, run = [], 0.0
    for rank, drop, saved, label in sorted(points, key=lambda p: -p[0]):
        run = max(run, drop, 0.0)
        out.append((rank, round(run, 4), round(drop, 4), saved, label))
    return out[::-1]


def load_table(tucker_csvs, svd_csv):
    groups = {}
    tucker_csvs = [tucker_csvs] if isinstance(tucker_csvs, Path) else tucker_csvs
    for r in (r for path in tucker_csvs for r in csv.DictReader(path.open(encoding='utf-8-sig'))):
        g = groups.setdefault(r['layer'], {'layer': r['layer'], 'kernel': f"{r['kernel']}x{r['kernel']}",
                                           'c_in': int(r['c_in']), 'c_out': int(r['c_out']), 'modes': {}})
        saved = 1000 * (float(r['layer_GFLOPs_base']) - float(r['layer_GFLOPs']))
        g['modes'].setdefault(r['mode'], []).append((int(r['rank']), 100 * float(r['drop_raw']), round(saved, 3),
                                                     f"tucker2-{r['mode']}"))
    for r in csv.DictReader(svd_csv.open(encoding='utf-8-sig')):
        if r['kernel'] != '1x1':
            continue                                  # k x k: identical to Tucker-2 'out'
        name = r['layer'].removesuffix('.conv')
        g = groups.setdefault(name, {'layer': name, 'kernel': '1x1', 'c_in': int(r['cin']), 'c_out': int(r['cout']),
                                     'modes': {}})
        saved = 2 * (int(r['macs_orig']) - int(r['macs_new'])) / 1e6
        g['modes'].setdefault('svd', []).append((int(r['rank']), float(r['plain_drop_mAP50-95']), round(saved, 3),
                                                 'svd'))
    table = []
    for name, g in groups.items():
        options = [('keep', 0, 0.0, 0.0, 0.0)]
        for mode, points in g['modes'].items():
            for rank, cost, raw, saved, label in monotone(points):
                if saved > 0:
                    options.append((label, rank, cost, raw, saved))
        table.append({'layer': name, 'kernel': g['kernel'], 'c_in': g['c_in'], 'c_out': g['c_out'],
                      'option': [o[0] for o in options], 'rank': [o[1] for o in options],
                      'drop': [o[2] for o in options], 'raw_drop': [o[3] for o in options],
                      'saved': [o[4] for o in options]})
    order = {n: i for i, n in enumerate(n for n, _ in load_model(ROOT/'models/baseline/8n_float32.pt')[0].named_modules())}
    table.sort(key=lambda t: order[t['layer']])
    return table


def low_rank_conv(conv, rank):
    """Plain nn.Conv2d with bias (Detect output convs) -> 1x1 conv to rank, 1x1 conv back with the bias."""
    w = conv.weight.detach().double().flatten(1)
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    first = nn.Conv2d(conv.in_channels, rank, conv.kernel_size, conv.stride, conv.padding, conv.dilation, bias=False)
    second = nn.Conv2d(rank, conv.out_channels, 1, bias=conv.bias is not None)
    first.weight.data = (s[:rank, None].sqrt() * vh[:rank]).reshape(first.weight.shape).float()
    second.weight.data = (u[:, :rank] * s[:rank].sqrt()).reshape(second.weight.shape).float()
    if conv.bias is not None:
        second.bias.data = conv.bias.detach().clone()
    return nn.Sequential(first, second).to(conv.weight.device)


def build(base, table, sel):
    model = deepcopy(base)
    for t, j in zip(table, sel):
        option, rank = t['option'][j], t['rank'][j]
        if option == 'keep':
            continue
        module = model.get_submodule(t['layer'])
        if isinstance(module, nn.Conv2d):
            set_module(model, t['layer'], low_rank_conv(module, rank))
        else:
            # svd of a 1 x 1 Conv block is the 'out' mode
            r_out, r_in = {'tucker2-in': (t['c_out'], rank), 'tucker2-out': (rank, t['c_in']),
                           'svd': (rank, t['c_in']), 'tucker2-both': (rank, rank)}[option]
            new, _ = decompose(module, r_out, r_in, HOOI_ITERS)
            set_module(model, t['layer'], new)
    return model


def plot(rows, base_gflops, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, grid = '#1f1f1e', '#e3e1dc'
    gray, blue, orange, green = '#8f8d87', '#2a78d6', '#d9692b', '#3a9a5b'
    b = [r['budget'] for r in rows]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200)
    for key, label, color, style in (('pred_drop', 'Predicted (sum of single-layer raw drops)', gray, '--'),
                                     ('raw_drop_search', 'Real, trained BN, search subset (500)', green, ':'),
                                     ('drop_search', 'Real, adaptive BN, search subset (500)', blue, '-'),
                                     ('drop_heldout', 'Real, adaptive BN, held-out val (996)', orange, '-')):
        left.plot(b, [r[key] for r in rows], style, color=color, linewidth=2, label=label)
    left.plot([0, max(b)], [0, max(b)], ':', color=grid, linewidth=1)
    for key, label, color, style in (('pred_saved_%', 'Predicted (sum of single-layer savings)', gray, '--'),
                                     ('saved_%', 'Real (decomposed model)', blue, '-')):
        right.plot(b, [r[key] for r in rows], style, color=color, linewidth=2, label=label)
    left.set_ylabel('mAP50-95 drop (points)', color=ink, fontsize=9)
    left.set_title('mAP50-95 drop', loc='left', fontsize=10, color=ink)
    right.set_ylabel(f'FLOPs saved (% of {base_gflops:.3f} GFLOPs)', color=ink, fontsize=9)
    right.set_title('FLOPs saved', loc='left', fontsize=10, color=ink)
    for ax in (left, right):
        ax.set_xlabel('Knapsack drop budget (points)', color=ink, fontsize=9)
        ax.grid(color=grid, linewidth=0.8)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out_dir/f'curve.{ext}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/baseline/8n_float32.pt')
    parser.add_argument('--data', type=Path, default=ROOT/'PTQ/datasets/kitti')
    parser.add_argument('--tucker', type=Path, nargs='+', default=[ROOT/'outputs/sensitivity_tucker2/sensitivity.csv',
                                                                   ROOT/'outputs/sensitivity_tucker2_both/sensitivity.csv'])
    parser.add_argument('--svd', type=Path, default=ROOT/'sensitivity_results/kitti_svd_rank_sensitivity/sensitivity.csv')
    parser.add_argument('--search-subset', type=Path, default=ROOT/'outputs/sensitivity_tucker2/val_subset.txt')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/knapsack_lowrank_both')
    parser.add_argument('--budgets', type=float, nargs='+', default=[round(0.1 * k, 1) for k in range(1, 51)])
    parser.add_argument('--bn-images', type=int, default=512)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    begin = time.perf_counter()

    table = load_table(args.tucker, args.svd)
    with (args.output/'options.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['layer', 'kernel', 'c_in', 'c_out', 'option', 'rank', 'cost_pts', 'raw_drop_pts', 'MFLOPs_saved'])
        for t in table:
            for j in range(len(t['option'])):
                writer.writerow([t['layer'], t['kernel'], t['c_in'], t['c_out'], t['option'][j], t['rank'][j],
                                 t['drop'][j], t['raw_drop'][j], t['saved'][j]])
    plans = best_plans(table, args.budgets)
    distinct = list(dict.fromkeys(tuple(p) for p in plans.values()))
    print(f'{len(table)} layers, {sum(len(t["option"]) for t in table)} options; '
          f'{len(args.budgets)} budgets -> {len(distinct)} distinct plans', flush=True)

    base, _ = load_model(args.weights)
    base.to(device).eval()
    example = torch.zeros(1, 3, 640, 640, device=device)
    base_macs, base_params = tp.utils.count_ops_and_params(base, example)
    evaluate, n_search, n_held = evaluators(args.data, args.search_subset, args.output, args.batch)
    calib = bn_images(args.data, args.bn_images, args.seed, hyp_from(base.args))
    ref = deepcopy(base)
    adapt_bn(ref, calib, args.batch, device)
    reference = {'GFLOPs': 2 * base_macs / 1e9, 'params': base_params,
                 'raw_mAP_search': 100 * evaluate['search'](base)['mAP50-95'],
                 'raw_mAP_heldout': 100 * evaluate['heldout'](base)['mAP50-95'],
                 'mAP_search': 100 * evaluate['search'](ref)['mAP50-95'],
                 'mAP_heldout': 100 * evaluate['heldout'](ref)['mAP50-95']}
    del ref
    print(f'Original: {reference}', flush=True)

    measured = {}
    for k, sel in enumerate(distinct):
        start = time.perf_counter()
        if not any(sel):
            measured[sel] = {key: reference[key] for key in ('params', 'raw_mAP_search', 'raw_mAP_heldout',
                                                              'mAP_search', 'mAP_heldout')}
            measured[sel]['real_GFLOPs'] = reference['GFLOPs']
            continue
        model = build(base, table, sel)
        macs, params = tp.utils.count_ops_and_params(model, example)
        m = {'real_GFLOPs': 2 * macs / 1e9, 'params': params,
             'raw_mAP_search': 100 * evaluate['search'](model)['mAP50-95'],
             'raw_mAP_heldout': 100 * evaluate['heldout'](model)['mAP50-95']}
        adapt_bn(model, calib, args.batch, device)
        m['mAP_search'] = 100 * evaluate['search'](model)['mAP50-95']
        m['mAP_heldout'] = 100 * evaluate['heldout'](model)['mAP50-95']
        measured[sel] = m
        del model
        print(f'[{k + 1}/{len(distinct)}] saved {100 * (1 - macs / base_macs):5.2f}%  held-out drop raw '
              f'{reference["raw_mAP_heldout"] - m["raw_mAP_heldout"]:5.2f} adaptive BN '
              f'{reference["mAP_heldout"] - m["mAP_heldout"]:5.2f}  ({time.perf_counter() - start:.1f}s)', flush=True)

    rows = []
    for budget in args.budgets:
        sel = tuple(plans[budget])
        m = measured[sel]
        chosen = [f"{t['option'][j]}:{t['rank'][j]}" if j else 'keep' for t, j in zip(table, sel)]
        rows.append({'budget': budget, 'layers_decomposed': sum(1 for j in sel if j),
                     'pred_drop': round(sum(t['drop'][j] for t, j in zip(table, sel)), 3),
                     'raw_drop_search': round(reference['raw_mAP_search'] - m['raw_mAP_search'], 3),
                     'raw_drop_heldout': round(reference['raw_mAP_heldout'] - m['raw_mAP_heldout'], 3),
                     'drop_search': round(reference['mAP_search'] - m['mAP_search'], 3),
                     'drop_heldout': round(reference['mAP_heldout'] - m['mAP_heldout'], 3),
                     'pred_saved_%': round(100 * sum(t['saved'][j] for t, j in zip(table, sel)) / 1000 / reference['GFLOPs'], 2),
                     'saved_%': round(100 * (1 - m['real_GFLOPs'] / reference['GFLOPs']), 2),
                     'real_GFLOPs': round(m['real_GFLOPs'], 4), 'params': m['params'],
                     'params_saved_%': round(100 * (1 - m['params'] / base_params), 2), 'choice': ' '.join(chosen)})
    with (args.output/'plans.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output/'plans.json').write_text(json.dumps({'reference': reference, 'layers': table, 'plans': rows},
                                                     separators=(',', ':')), encoding='utf-8')
    plot(rows, reference['GFLOPs'], args.output)
    manifest = {'source': str(args.weights.relative_to(ROOT)),
                'tucker_scan': [str(p.relative_to(ROOT)) for p in args.tucker], 'svd_scan': str(args.svd.relative_to(ROOT)),
                'knapsack': 'top 1 per budget; k x k: keep / Tucker-2 out / in / both ranks, 1 x 1: keep / SVD ranks; '
                            'cost = single-layer raw drop (trained BN), >= 0, monotone in rank; value = single-layer '
                            'MFLOPs saved, options saving nothing dropped', 'unit_points': 0.01,
                'budgets': args.budgets, 'layers': len(table), 'distinct_plans': len(distinct),
                'adaptive_bn': {'images': args.bn_images, 'seed': args.seed},
                'validation': {'search_images': n_search, 'heldout_images': n_held},
                'reference_original': reference, 'seconds': time.perf_counter() - begin}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Done: {args.output}')


if __name__ == '__main__':
    main()
