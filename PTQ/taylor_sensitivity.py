"""Per-group pruning sensitivity of YOLOv8n-KITTI: Taylor-1 ranking, adaptive BN, small validation subset.

For every Torch-Pruning dependency group (C2f_v2 model, see dependency_groups.py) and every ratio in --ratios,
only that group is pruned: its lowest-Taylor-1 channels (group score = sum of the member convolutions'
Taylor-1, as in taylor_global_prune.py) are removed. Then adaptive BN: all BN running statistics are reset and
re-estimated (cumulative average) on a fixed set of --bn-images KITTI training images passed through the training
augmentation pipeline (mosaic, as when the checkpoint's statistics were accumulated; generated once), and
mAP is measured on a fixed random subset of --val-images KITTI validation images. No fine-tuning.
The reference is the unpruned model after the same adaptive-BN step, so a drop reflects pruning only.

Outputs (in --output):
  taylor_scores.pt         Taylor-1 per original convolution (reused on later runs)
  val_subset.txt           validation images used
  sensitivity.csv          one row per (group, ratio); appended as it runs, so an interrupted scan resumes
  drop_mAP50-95.csv        pivot: group x ratio, mAP50-95 drop (percentage points)
  sensitivity_heatmap.png  the same pivot as a heatmap
  manifest.json            settings and references
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch_pruning as tp
import yaml
from torch import nn
from ultralytics.data import build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionValidator

from taylor_global_prune import (KITTI, ROOT, dependency_groups, group_scores, load_unfused, sha256, taylor_scores,
                                 to_c2f_v2)

FIELDS = ['group', 'root', 'members', 'channels', 'ratio', 'pruned', 'kept', 'taylor_mass_pruned', 'params',
          'gmacs', 'mAP50-95', 'mAP50', 'precision', 'recall', 'drop_mAP50-95', 'drop_mAP50', 'seconds']


def bn_batches(data_dir, count, batch, seed, hyp):
    """Fixed adaptive-BN batches, generated once (seeded) and kept in memory as uint8.
    They come from the training pipeline with the checkpoint's own hyperparameters (mosaic on): the checkpoint's
    BN statistics were accumulated on mosaic batches. Re-estimating them on plain letterboxed images costs the
    unpruned model 3-4 pp mAP50-95 on KITTI val, while mosaic batches reproduce its accuracy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = {'path': str(data_dir), 'train': 'images/train', 'val': 'images/val', 'names': KITTI, 'nc': len(KITTI),
            'channels': 3}
    hyp = copy.copy(hyp)
    hyp.imgsz = 640
    dataset = build_yolo_dataset(hyp, str(data_dir/'images/train'), batch, data, mode='train')
    order = np.random.default_rng(seed).permutation(len(dataset))[:count]
    return [torch.stack([dataset[int(i)]['img'] for i in order[k:k + batch]]) for k in range(0, len(order), batch)]


@torch.no_grad()
def adapt_bn(model, batches, device):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None                 # cumulative average over all adaptation batches
            m.train()
    for x in batches:
        model(x.to(device).float() / 255)
    model.eval()


class SubsetValidator:
    """DetectionValidator on a fixed val subset; the dataloader (and its workers) is built once and reused."""
    def __init__(self, data_yaml, batch):
        self.args = dict(data=str(data_yaml), imgsz=640, batch=batch, half=False, plots=False, save_json=False,
                         verbose=False, workers=4)
        self.dataloader = None

    def __call__(self, model):
        validator = DetectionValidator(dataloader=self.dataloader, args=self.args)
        stats = validator(model=copy.deepcopy(model).eval())     # the validator fuses BN in place
        self.dataloader = validator.dataloader
        return {'mAP50-95': stats['metrics/mAP50-95(B)'], 'mAP50': stats['metrics/mAP50(B)'],
                'precision': stats['metrics/precision(B)'], 'recall': stats['metrics/recall(B)']}


def heatmap(pivot, ratios, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    roots = list(pivot)
    values = np.array([[pivot[r].get(q, np.nan) for q in ratios] for r in roots])
    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * len(ratios), 0.6 + 0.22 * len(roots)))
    vmax = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    im = ax.imshow(values, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(ratios)), [f'{q:.0%}' for q in ratios])
    ax.set_yticks(range(len(roots)), roots, fontsize=7)
    ax.set_xlabel('channels pruned in the group (lowest Taylor-1 first)')
    fig.colorbar(im, ax=ax, label='mAP50-95 drop (pp) vs. unpruned + adaptive BN')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/baseline/8n_float32.pt')
    parser.add_argument('--data', type=Path, default=ROOT/'PTQ/datasets/kitti')
    parser.add_argument('--output', type=Path, default=ROOT/'sensitivity_results/kitti_taylor1_group_sensitivity')
    parser.add_argument('--ratios', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument('--val-images', type=int, default=500)
    parser.add_argument('--bn-images', type=int, default=1280)
    parser.add_argument('--bn-batch', type=int, default=16)
    parser.add_argument('--batch', type=int, default=32, help='validation batch size')
    parser.add_argument('--groups', type=int, nargs='*', default=None, help='only these group indices')
    parser.add_argument('--workers', type=int, default=8, help='dataloader workers for Taylor scoring')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')
    example = torch.zeros(1, 3, 640, 640, device=device)

    # Small validation subset (fixed) and adaptive-BN images (fixed, disjoint split: train).
    val_files = sorted((args.data/'images/val').glob('*.png'))
    subset = [val_files[int(i)] for i in np.random.default_rng(args.seed).permutation(len(val_files))[:args.val_images]]
    (args.output/'val_subset.txt').write_text('\n'.join(str(p) for p in sorted(subset)) + '\n', encoding='utf-8')
    data_yaml = args.output/'kitti_val_subset.yaml'
    data_yaml.write_text(yaml.safe_dump({'path': str(args.data), 'train': 'images/train',
                                         'val': str(args.output/'val_subset.txt'), 'names': KITTI,
                                         'nc': len(KITTI)}), encoding='utf-8')
    validate = SubsetValidator(data_yaml, args.batch)

    # Taylor-1 scores (all training images), cached.
    model = load_unfused(args.weights, device)
    batches = bn_batches(args.data, args.bn_images, args.bn_batch, args.seed, model.args)
    cache = args.output/'taylor_scores.pt'
    if cache.is_file() and torch.load(cache)['source_sha256'] == sha256(args.weights):
        payload = torch.load(cache)
        scores, taylor_info = payload['scores'], payload['info']
        print(f'Taylor-1 scores loaded from {cache.name}', flush=True)
    else:
        print('Taylor-1 scores on all KITTI training images...', flush=True)
        scores, taylor_info = taylor_scores(model, args.data, device, args.workers)
        torch.save({'scores': scores, 'info': taylor_info, 'source_sha256': sha256(args.weights)}, cache)

    # References on the val subset.
    base = load_unfused(args.weights, device)
    score_of, diff = to_c2f_v2(base, scores, example)
    ref_plain = validate(base)
    ref_model = copy.deepcopy(base)
    adapt_bn(ref_model, batches, device)
    ref = validate(ref_model)
    base_macs, base_params = tp.utils.count_ops_and_params(base, example)
    print(f'Unpruned on {len(subset)} val images: {ref_plain}\n  after adaptive BN: {ref}', flush=True)

    names_of = {m: n for n, m in base.named_modules()}
    graph, groups, _ = dependency_groups(base, example)
    rows = group_scores(graph, groups, score_of, names_of)
    selected = [r for r in rows if args.groups is None or r['group'] in args.groups]

    out_csv = args.output/'sensitivity.csv'
    done = set()
    if out_csv.is_file():
        with out_csv.open(encoding='utf-8-sig') as stream:
            done = {(r['root'], float(r['ratio'])) for r in csv.DictReader(stream)}
    else:
        with out_csv.open('w', encoding='utf-8-sig', newline='') as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writeheader()
    trials = [(r, q) for r in selected for q in args.ratios if (r['root'], q) not in done]
    print(f'{len(groups)} groups; {len(trials)} trials to run ({len(done)} already done)', flush=True)
    manifest = {'source': str(args.weights), 'source_sha256': sha256(args.weights), 'torch': torch.__version__,
                'torch_pruning': tp.__version__, 'ranking': 'Taylor-1 group score (sum over coupled member convs)',
                'taylor': taylor_info, 'c2f_split_max_abs_diff': diff, 'ratios': args.ratios,
                'adaptive_bn': {'images': sum(len(b) for b in batches), 'split': 'KITTI train', 'batch': args.bn_batch,
                                'augment': 'training pipeline, checkpoint hyperparameters (mosaic on), generated once',
                                'statistics': 'reset, cumulative average', 'seed': args.seed},
                'val_subset': {'images': len(subset), 'split': 'KITTI val', 'seed': args.seed,
                               'file': 'val_subset.txt'},
                'reference_unpruned': ref_plain, 'reference_unpruned_adaptive_bn': ref,
                'unpruned_params': base_params, 'unpruned_gmacs': base_macs / 1e9, 'fine_tuned': False,
                'groups': len(groups)}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    begin = time.perf_counter()
    for k, (row, ratio) in enumerate(trials):
        start = time.perf_counter()
        n = min(int(round(ratio * row['channels'])), row['channels'] - 1)
        if n == 0:
            continue
        model = copy.deepcopy(base)
        trial_graph, trial_groups, _ = dependency_groups(model, example)
        trial_names = {m: name for name, m in model.named_modules()}
        group = next(g for g in trial_groups if trial_names[g[0].dep.target.module] == row['root'])
        order = torch.argsort(row['imp'])[:n].tolist()          # lowest Taylor-1 first
        pruning_group = trial_graph.get_pruning_group(group[0].dep.target.module, group[0].dep.handler, sorted(order))
        if not trial_graph.check_pruning_group(pruning_group):
            raise RuntimeError(f'Torch-Pruning rejected {row["root"]} at {ratio}')
        pruning_group.prune()
        adapt_bn(model, batches, device)
        metrics = validate(model)
        macs, params = tp.utils.count_ops_and_params(model, example)
        record = {'group': row['group'], 'root': row['root'], 'members': row['members'], 'channels': row['channels'],
                  'ratio': ratio, 'pruned': n, 'kept': row['channels'] - n,
                  'taylor_mass_pruned': float(row['imp'][order].sum() / row['imp'].sum()), 'params': params,
                  'gmacs': macs / 1e9, **metrics,
                  'drop_mAP50-95': 100 * (ref['mAP50-95'] - metrics['mAP50-95']),
                  'drop_mAP50': 100 * (ref['mAP50'] - metrics['mAP50']), 'seconds': time.perf_counter() - start}
        with out_csv.open('a', encoding='utf-8-sig', newline='') as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writerow(record)
        elapsed = time.perf_counter() - begin
        print(f"[{k + 1}/{len(trials)}] {row['root']:<24} {ratio:.0%} ({n}/{row['channels']}) "
              f"mAP50-95={metrics['mAP50-95']:.4f} drop={record['drop_mAP50-95']:+.2f}pp "
              f"ETA {elapsed / (k + 1) * (len(trials) - k - 1) / 60:.0f} min", flush=True)

    # Pivot and heatmap over everything in sensitivity.csv.
    with out_csv.open(encoding='utf-8-sig') as stream:
        results = list(csv.DictReader(stream))
    ratios = sorted({float(r['ratio']) for r in results})
    pivot = {}
    for r in sorted(results, key=lambda r: int(r['group'])):
        pivot.setdefault(r['root'], {})[float(r['ratio'])] = float(r['drop_mAP50-95'])
    with (args.output/'drop_mAP50-95.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['root'] + [f'{q:g}' for q in ratios])
        for root, values in pivot.items():
            writer.writerow([root] + [f'{values[q]:.3f}' if q in values else '' for q in ratios])
    heatmap(pivot, ratios, args.output/'sensitivity_heatmap.png')
    print(f'Done: {args.output}')


if __name__ == '__main__':
    main()
