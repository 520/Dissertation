"""Candidate pruning plans for YOLOv8n-KITTI from the L2 sensitivity scan, evaluated for real, and their Pareto front.

Candidates
  For every pruning-ratio cap and every drop budget, a K-best multiple-choice knapsack over the 59 DepGraph groups
  (value: FLOPs saved by the group alone, cost: its mAP50-95 drop alone, made non-decreasing in the ratio, negative
  drops counted as 0) gives the --k best plans under the additive model. From those, up to --per-budget plans are kept
  greedily in value order, each differing from every kept plan in at least --min-diff groups whose chosen options do
  not both cost zero (plans that only differ on free groups add no information). Uniform plans (every group at the
  same ratio) are added as baselines.

Real evaluation (every candidate, from the unpruned model)
  prune all chosen groups together (lowest group-L2 channels, importance taken on the unpruned model as in the scan),
  count the real FLOPs, adaptive BN exactly as in Pruning/sensitivity_l2.py (512 training-augmented images, same
  seed), then mAP50-95 on (a) the 500-image search subset used by the scan and (b) the held-out remainder of KITTI val
  (996 images, never used to rank anything). The Pareto front is taken on real GFLOPs and held-out mAP50-95.

Outputs (in --output):
  candidates.csv   every candidate: source, plan, predicted and real FLOPs, predicted and measured drop
  pareto.csv       the non-dominated knapsack candidates, by real GFLOPs (uniform plans are baselines only)
  pareto.png/.pdf  held-out mAP50-95 against real GFLOPs, front highlighted
  manifest.json    settings and references
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

from Pruning.sensitivity_l2 import OUT_CH, Evaluator, adapt_bn, bn_images, groups_of
from Pruning.taylor_prune import KITTI, hyp_from, load_model, split_c2f
from ultralytics.nn.modules import Detect

ROOT = Path(__file__).resolve().parents[1]
RATIOS = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
UNIT = 0.01        # knapsack cost resolution, mAP points


def load_table(path):
    """Per group: name, channels, and per ratio index the drop (points, cumulative max, >= 0) and MFLOPs saved."""
    rows = list(csv.DictReader(path.open(encoding='utf-8-sig')))
    base = json.loads((path.parent/'manifest.json').read_text())['base_GFLOPs']
    table = {}
    for r in rows:
        g = table.setdefault(int(r['group']), {'root': r['root'], 'channels': int(r['channels']),
                                               'drop': {0: 0.0}, 'saved': {0: 0.0}})
        k = RATIOS.index(float(r['ratio']))
        g['drop'][k] = 100 * float(r['drop_mAP50-95'])
        g['saved'][k] = 1000 * (base - float(r['GFLOPs']))
    out = []
    for g in sorted(table):
        t = table[g]
        run, drop = 0.0, []
        for k in range(len(RATIOS)):
            run = max(run, max(0.0, t['drop'][k]))
            drop.append(run)
        out.append({'group': g, 'root': t['root'], 'channels': t['channels'], 'drop': drop,
                    'saved': [t['saved'][k] for k in range(len(RATIOS))]})
    return out, base


def kbest_knapsack(table, max_idx, budget_cells, k):
    """Top-k plans by total value with total cost <= b, for every b up to budget_cells. Returns a backtracking fn."""
    n, B = len(table), budget_cells
    costs = [[int(round(t['drop'][j] / UNIT)) for j in range(max_idx + 1)] for t in table]
    values = [t['saved'][:max_idx + 1] for t in table]
    prev = np.full((B + 1, k), -np.inf)
    prev[:, 0] = 0.0
    choice = np.zeros((n, B + 1, k), np.int8)
    rank = np.zeros((n, B + 1, k), np.int16)
    for i in range(n):
        cur = np.full((B + 1, k), -np.inf)
        for b in range(B + 1):
            vals, js, rs = [], [], []
            for j, c in enumerate(costs[i]):
                if c > b:
                    continue
                vals.append(prev[b - c] + values[i][j])
                js.append(np.full(k, j))
                rs.append(np.arange(k))
            vals, js, rs = np.concatenate(vals), np.concatenate(js), np.concatenate(rs)
            top = np.argsort(-vals, kind='stable')[:k]
            cur[b], choice[i, b], rank[i, b] = vals[top], js[top], rs[top]
        prev = cur

    def plans(b):
        out = []
        for r0 in range(k):
            if not np.isfinite(prev[b, r0]):
                break
            sel, bb, r = [0] * n, b, r0
            for i in range(n - 1, -1, -1):
                j = int(choice[i, bb, r])
                sel[i], r, bb = j, int(rank[i, bb, r]), bb - costs[i][j]
            out.append(sel)
        return out
    return plans, costs


def diverse(plans, costs, keep, min_diff):
    kept = []
    for p in plans:
        def distance(q):
            return sum(1 for i, (a, b) in enumerate(zip(p, q)) if a != b and (costs[i][a] or costs[i][b]))
        if all(distance(q) >= min_diff for q in kept):
            kept.append(p)
        if len(kept) == keep:
            break
    return kept


def build(base, example, importance, plan):
    """Prune every group of the plan together; channel choice from the unpruned model's group L2."""
    model = deepcopy(base)
    graph, groups = groups_of(model, example)
    todo = []
    for gi, j in enumerate(plan):
        if j == 0:
            continue
        group = groups[gi]
        channels = len(group[0].idxs)
        n = min(int(round(RATIOS[j] * channels)), channels - 1)
        todo.append((group[0].dep.target.module, torch.argsort(importance(group))[:n].tolist()))
    for root, idxs in todo:
        graph.get_pruning_group(root, OUT_CH, idxs).prune()
    for m in model.modules():
        if isinstance(m, Detect):
            m.shape = None
    return model


def pareto(rows, x='real_GFLOPs', y='mAP_heldout'):
    """Non-dominated rows: no other row has fewer (or equal) GFLOPs and higher (or equal) mAP, one strictly."""
    front = [r for r in rows if not any((o[x] <= r[x] and o[y] >= r[y]) and (o[x] < r[x] or o[y] > r[y]) for o in rows)]
    return sorted(front, key=lambda r: r[x])


def plot(rows, front, reference, out_dir, marked=()):
    """Left: every candidate with the uniform baselines. Right: the knapsack region, front and marked plans."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, muted, grid = '#1f1f1e', '#6b6a66', '#e3e1dc'
    blue, orange, gray = '#2a78d6', '#d9692b', '#a8a6a0'
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), dpi=200, gridspec_kw={'width_ratios': [1, 1.35]})
    other = [r for r in rows if r not in front and r['source'] != 'uniform']
    uni = [r for r in rows if r['source'] == 'uniform']
    x = lambda rs: [r['real_GFLOPs'] for r in rs]
    y = lambda rs: [r['mAP_heldout'] for r in rs]
    for k, ax in enumerate(axes):
        ax.scatter(x(other), y(other), s=20, color=gray, edgecolor='white', linewidth=0.8, label='Knapsack candidates', zorder=2)
        ax.plot(x(front), y(front), color=blue, linewidth=2, zorder=3)
        ax.scatter(x(front), y(front), s=36, color=blue, edgecolor='white', linewidth=1.2, label='Pareto front', zorder=4)
        ax.scatter([reference['GFLOPs']], [reference['mAP_heldout']], s=90, marker='*', color=ink, zorder=5,
                   label='Unpruned (adaptive BN)')
        ax.grid(color=grid, linewidth=0.8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
        ax.set_xlabel('Real GFLOPs (640×640)', color=ink, fontsize=9)
    left, right = axes
    left.scatter(x(uni), y(uni), s=60, marker='D', color=orange, edgecolor='white', linewidth=1.2,
                 label='Uniform ratio (all 59 groups)', zorder=4)
    for r in uni:
        left.annotate(r['id'].replace('uniform_', 'uniform '), (r['real_GFLOPs'], r['mAP_heldout']),
                      textcoords='offset points', xytext=(7, 4), fontsize=8, color=ink)
    left.set_ylabel('mAP50-95 on held-out KITTI val (%)', color=ink, fontsize=9)
    left.set_title('All candidates', loc='left', fontsize=10, color=ink)
    left.legend(frameon=False, fontsize=7.5, loc='center right')
    knap = [r for r in rows if r['source'] == 'knapsack'] + [{'real_GFLOPs': reference['GFLOPs'], 'mAP_heldout': reference['mAP_heldout']}]
    xs, ys = x(knap), y(knap)
    right.set_xlim(min(xs) - 0.08, max(xs) + 0.12)
    right.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    right.set_title('Knapsack region (zoom)', loc='left', fontsize=10, color=ink)
    by_id = {r['id']: r for r in rows}
    for label, cid in marked:
        r = by_id[cid]
        right.scatter([r['real_GFLOPs']], [r['mAP_heldout']], s=120, facecolor='none', edgecolor=orange, linewidth=1.8, zorder=6)
        right.annotate(f"{label}: {r['saved_%']:.1f}% fewer FLOPs, -{r['drop_heldout']:.2f} mAP",
                       (r['real_GFLOPs'], r['mAP_heldout']), textcoords='offset points', xytext=(-12, -16),
                       ha='right', fontsize=8, color=ink)
    fig.suptitle('YOLOv8n-KITTI pruning plans after adaptive BN, before fine-tuning', x=0.01, ha='left', fontsize=11, color=ink)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(out_dir/f'pareto.{suffix}', facecolor='white')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/baseline/8n_float32.pt')
    parser.add_argument('--data', type=Path, default=ROOT/'PTQ/datasets/kitti')
    parser.add_argument('--scan', type=Path, default=ROOT/'outputs/sensitivity_l2_eighths/sensitivity.csv')
    parser.add_argument('--search-subset', type=Path, default=ROOT/'outputs/sensitivity_l2_eighths/val_subset.txt')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/pareto')
    parser.add_argument('--budgets', type=float, nargs='+', default=[0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5])
    parser.add_argument('--caps', type=float, nargs='+', default=[0.5, 0.75, 0.875], help='maximum ratio per group')
    parser.add_argument('--uniform', type=float, nargs='*', default=[0.125, 0.25, 0.375])
    parser.add_argument('--k', type=int, default=200, help='knapsack plans kept per budget before diversity')
    parser.add_argument('--per-budget', type=int, default=4)
    parser.add_argument('--min-diff', type=int, default=3)
    parser.add_argument('--bn-images', type=int, default=512)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0, help='evaluate only the first N candidates (smoke test)')
    parser.add_argument('--replot', action='store_true', help='redo front and plot from an existing candidates.csv')
    parser.add_argument('--mark', nargs='*', default=[], help='LABEL=candidate_id pairs to circle on the zoomed plot')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')

    marked = [m.split('=', 1) for m in args.mark]
    if args.replot:
        rows = list(csv.DictReader((args.output/'candidates.csv').open(encoding='utf-8-sig')))
        for r in rows:
            for key in ('real_GFLOPs', 'mAP_heldout', 'saved_%', 'drop_heldout'):
                r[key] = float(r[key])
        manifest = json.loads((args.output/'manifest.json').read_text())
        reference = manifest['reference_unpruned_adaptive_bn']
        front = pareto([r for r in rows if r['source'] == 'knapsack'])
        with (args.output/'pareto.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(front)
        plot(rows, front, reference, args.output, marked)
        manifest['pareto_front'] = len(front)
        manifest['pareto_front_over'] = 'knapsack candidates (uniform baselines plotted separately)'
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'Pareto front: {len(front)} knapsack candidates')
        return

    table, base_gflops = load_table(args.scan)
    n = len(table)
    candidates, seen = [], set()

    def add(cid, source, budget, cap, plan, costs=None):
        key = tuple(plan)
        if key in seen:
            return
        seen.add(key)
        candidates.append({'id': cid, 'source': source, 'budget': budget, 'cap': cap, 'plan': ''.join(map(str, plan)),
                           'groups_pruned': sum(1 for j in plan if j),
                           'pred_drop': round(sum(table[i]['drop'][j] for i, j in enumerate(plan)), 3),
                           'pred_GFLOPs_saved': round(sum(table[i]['saved'][j] for i, j in enumerate(plan)) / 1000, 4)})

    begin = time.perf_counter()
    cells = int(round(max(args.budgets) / UNIT))
    for cap in args.caps:
        max_idx = RATIOS.index(cap)
        plans_at, costs = kbest_knapsack(table, max_idx, cells, args.k)
        for budget in args.budgets:
            for r, plan in enumerate(diverse(plans_at(int(round(budget / UNIT))), costs, args.per_budget, args.min_diff)):
                add(f'cap{round(cap*1000)}_b{budget:g}_r{r}', 'knapsack', budget, cap, plan)
    for ratio in args.uniform:
        add(f'uniform_{ratio*100:g}%', 'uniform', '', ratio, [RATIOS.index(ratio)] * n)
    print(f'{len(candidates)} distinct candidates ({time.perf_counter() - begin:.0f}s knapsack)', flush=True)
    if args.limit:
        candidates = candidates[:args.limit]

    base, _ = load_model(args.weights)
    split_c2f(base)
    for p in base.parameters():
        p.requires_grad_(True)
    base.to(device).eval()
    example = torch.zeros(1, 3, 640, 640, device=device)
    base_macs, base_params = tp.utils.count_ops_and_params(base, example)
    importance = tp.importance.GroupMagnitudeImportance(p=2, group_reduction='mean', normalizer=None)

    search = [Path(p) for p in args.search_subset.read_text(encoding='utf-8').split()]
    all_val = sorted((args.data/'images/val').glob('*.png')) + sorted((args.data/'images/val').glob('*.jpg'))
    in_search = {p.resolve() for p in search}
    heldout = [p.resolve() for p in all_val if p.resolve() not in in_search]
    evaluators = {}
    for name, files in (('search', search), ('heldout', heldout)):
        lst = args.output/f'val_{name}.txt'
        lst.write_text('\n'.join(str(f) for f in files) + '\n', encoding='utf-8')
        yaml = args.output/f'kitti_val_{name}.yaml'
        yaml.write_text(json.dumps({'path': str(args.data.resolve()), 'train': 'images/train', 'val': str(lst.resolve()),
                                    'nc': len(KITTI), 'names': KITTI}), encoding='utf-8')
        evaluators[name] = Evaluator(yaml, args.batch, args.output/f'val_{name}')
    calib = bn_images(args.data, args.bn_images, args.seed, hyp_from(base.args))

    ref_model = deepcopy(base)
    adapt_bn(ref_model, calib, args.batch, device)
    reference = {'GFLOPs': 2 * base_macs / 1e9, 'params': base_params,
                 'mAP_search': 100 * evaluators['search'](ref_model)['mAP50-95'],
                 'mAP_heldout': 100 * evaluators['heldout'](ref_model)['mAP50-95']}
    del ref_model
    print(f'Unpruned after adaptive BN: {reference}', flush=True)

    fields = list(candidates[0]) + ['real_GFLOPs', 'real_GFLOPs_saved', 'saved_%', 'params', 'mAP_search', 'mAP_heldout',
                                    'drop_search', 'drop_heldout', 'seconds']
    out_csv = args.output/'candidates.csv'
    for c, cand in enumerate(candidates):
        start = time.perf_counter()
        model = build(base, example, importance, [int(ch) for ch in cand['plan']])
        macs, params = tp.utils.count_ops_and_params(model, example)
        adapt_bn(model, calib, args.batch, device)
        m_search = 100 * evaluators['search'](model)['mAP50-95']
        m_held = 100 * evaluators['heldout'](model)['mAP50-95']
        cand.update({'real_GFLOPs': round(2 * macs / 1e9, 4), 'real_GFLOPs_saved': round(reference['GFLOPs'] - 2 * macs / 1e9, 4),
                     'saved_%': round(100 * (1 - macs / base_macs), 2), 'params': params,
                     'mAP_search': round(m_search, 3), 'mAP_heldout': round(m_held, 3),
                     'drop_search': round(reference['mAP_search'] - m_search, 3),
                     'drop_heldout': round(reference['mAP_heldout'] - m_held, 3),
                     'seconds': round(time.perf_counter() - start, 1)})
        del model
        with out_csv.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(candidates[:c + 1])
        print(f"[{c + 1}/{len(candidates)}] {cand['id']:<22} saved {cand['saved_%']:5.1f}% "
              f"(pred {100 * cand['pred_GFLOPs_saved'] / base_gflops:5.1f}%)  drop pred {cand['pred_drop']:5.2f}  "
              f"search {cand['drop_search']:5.2f}  held-out {cand['drop_heldout']:5.2f}  ({cand['seconds']}s)", flush=True)

    done = [c for c in candidates if 'mAP_heldout' in c]
    front = pareto([c for c in done if c['source'] == 'knapsack'])
    with (args.output/'pareto.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(front)
    plot(done, front, reference, args.output, marked)
    knap = [c for c in done if c['source'] == 'knapsack']
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else None
    rank = lambda x: np.argsort(np.argsort(x))
    manifest = {'source': str(args.weights.relative_to(ROOT)), 'scan': str(args.scan.relative_to(ROOT)),
                'importance': 'GroupMagnitudeImportance(p=2), taken on the unpruned model',
                'knapsack': {'value': 'MFLOPs saved by the group alone', 'cost': 'mAP50-95 drop alone, cumulative max, >= 0',
                             'unit_points': UNIT, 'budgets': args.budgets, 'caps': args.caps, 'k': args.k,
                             'per_budget': args.per_budget, 'min_diff_groups': args.min_diff},
                'uniform_baselines': args.uniform,
                'adaptive_bn': {'images': args.bn_images, 'augment': 'training hyperparameters', 'seed': args.seed},
                'validation': {'search_images': len(search), 'heldout_images': len(heldout)},
                'reference_unpruned_adaptive_bn': reference, 'candidates': len(done), 'pareto_front': len(front),
                'pareto_front_over': 'knapsack candidates (uniform baselines plotted separately)',
                'checks': {'pearson_pred_vs_search_drop': corr([c['pred_drop'] for c in knap], [c['drop_search'] for c in knap]),
                           'spearman_search_vs_heldout_mAP': corr(rank([c['mAP_search'] for c in done]),
                                                                  rank([c['mAP_heldout'] for c in done])),
                           'mean_pred_over_real_GFLOPs_saved': float(np.mean([c['pred_GFLOPs_saved'] / c['real_GFLOPs_saved']
                                                                              for c in knap if c['real_GFLOPs_saved'] > 0]))},
                'seconds': time.perf_counter() - begin}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest['checks'], indent=1))
    print(f'Pareto front: {len(front)} of {len(done)} candidates. Done: {args.output}')


if __name__ == '__main__':
    main()
