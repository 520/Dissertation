"""Per-layer low-rank (truncated SVD) sensitivity of YOLOv8n-KITTI on a small validation subset.

For every convolution (DFL excluded) and every rank r = 8, 16, ... up to its full rank min(Cout, Cin*kh*kw), only
that convolution is factorised: W (Cout x Cin*kh*kw) ~ U_r S_r V_r^T becomes a kh x kw conv Cin -> r (sqrt(S_r) V_r^T,
same stride/padding) followed by a 1x1 conv r -> Cout (U_r sqrt(S_r), original bias). No fine-tuning.
mAP is measured twice per trial on the same fixed --val-images KITTI val subset as taylor_sensitivity.py:
  plain       factorised as is
  adaptive BN BN running statistics re-estimated (calibration, as in taylor_sensitivity.py) on --bn-images
              KITTI training images, then BN folded; references are the unfactorised model with/without the same step

Outputs (in --output):
  layers.csv               every convolution: shape, full rank, ranks scanned, output size
  sensitivity.csv          one row per (layer, rank); appended as it runs, so an interrupted scan resumes
  drop_mAP50-95.csv        pivot: layer x rank, mAP50-95 drop after adaptive BN (percentage points)
  manifest.json            settings and references
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from ultralytics.nn.modules import Conv
from ultralytics.utils.torch_utils import fuse_conv_and_bn

from taylor_global_prune import KITTI, ROOT, load_unfused, sha256
from taylor_sensitivity import SubsetValidator, adapt_bn, bn_batches

FIELDS = ['layer', 'cout', 'cin', 'kernel', 'full_rank', 'rank', 'energy_kept', 'rel_error', 'params_orig',
          'params_new', 'macs_orig', 'macs_new', 'model_params', 'model_gmacs', 'plain_mAP50-95', 'plain_mAP50',
          'plain_drop_mAP50-95', 'mAP50-95', 'mAP50', 'precision', 'recall', 'drop_mAP50-95', 'drop_mAP50', 'seconds']


def conv_layers(model, example):
    """Every factorisable conv: (name, module, parent Conv block or None, output H*W)."""
    blocks = {m.conv: m for m in model.modules() if isinstance(m, Conv)}
    sizes, hooks = {}, []
    convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d) and '.dfl.' not in n
             and m.groups == 1]
    for n, m in convs:
        hooks.append(m.register_forward_hook(lambda mod, i, o, n=n: sizes.__setitem__(n, o.shape[2] * o.shape[3])))
    with torch.no_grad():
        model(example)
    for h in hooks:
        h.remove()
    return [(n, m, blocks.get(m), sizes[n]) for n, m in convs]


def factorise(conv, rank):
    w = conv.weight.detach().double()
    cout, cin, kh, kw = w.shape
    u, s, vh = torch.linalg.svd(w.reshape(cout, -1), full_matrices=False)
    root = s[:rank].sqrt()
    first = nn.Conv2d(cin, rank, (kh, kw), conv.stride, conv.padding, conv.dilation, bias=False)
    second = nn.Conv2d(rank, cout, 1, bias=conv.bias is not None)
    first.weight.data = (root[:, None] * vh[:rank]).reshape(rank, cin, kh, kw).float()
    second.weight.data = (u[:, :rank] * root).reshape(cout, rank, 1, 1).float()
    if conv.bias is not None:
        second.bias.data = conv.bias.detach().clone()
    energy = float((s[:rank] ** 2).sum() / (s ** 2).sum())
    return nn.Sequential(first, second).to(conv.weight.device), energy, float(np.sqrt(max(0.0, 1 - energy)))


def apply(model, name, rank):
    """Factorise `name` in `model` in place; returns (energy kept, relative Frobenius error, parent Conv block)."""
    parent_name, _, attr = name.rpartition('.')
    parent = model.get_submodule(parent_name)
    seq, energy, err = factorise(getattr(parent, attr), rank)
    setattr(parent, attr, seq)
    return energy, err, parent if isinstance(parent, Conv) else None


def fold_bn(block):
    """Fold the block's BN into the second (1x1) conv, so the validator's fuse() leaves the block alone."""
    if block is not None and hasattr(block, 'bn'):
        block.conv[1] = fuse_conv_and_bn(block.conv[1], block.bn)
        delattr(block, 'bn')
        block.forward = block.forward_fuse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/baseline/8n_float32.pt')
    parser.add_argument('--data', type=Path, default=ROOT/'PTQ/datasets/kitti')
    parser.add_argument('--output', type=Path, default=ROOT/'sensitivity_results/kitti_svd_rank_sensitivity')
    parser.add_argument('--step', type=int, default=8, help='ranks are multiples of this')
    parser.add_argument('--val-images', type=int, default=500)
    parser.add_argument('--bn-images', type=int, default=1280)
    parser.add_argument('--bn-batch', type=int, default=16)
    parser.add_argument('--batch', type=int, default=32, help='validation batch size')
    parser.add_argument('--layers', nargs='*', default=None, help='only these conv names')
    parser.add_argument('--list', action='store_true', help='write layers.csv, print the trial count and exit')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')
    example = torch.zeros(1, 3, 640, 640, device=device)

    base = load_unfused(args.weights, device)
    layers = conv_layers(base, example)
    plan = []
    with (args.output/'layers.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['layer', 'cout', 'cin', 'kernel', 'full_rank', 'ranks', 'out_hw', 'params', 'macs'])
        for name, conv, _, hw in layers:
            cout, cin, kh, kw = conv.weight.shape
            full = min(cout, cin * kh * kw)
            ranks = list(range(args.step, full + 1, args.step))
            writer.writerow([name, cout, cin, f'{kh}x{kw}', full, len(ranks), hw, conv.weight.numel(),
                             conv.weight.numel() * hw])
            if args.layers is None or name in args.layers:
                plan += [(name, r) for r in ranks]
    print(f'{len(layers)} convolutions, {len(plan)} (layer, rank) trials', flush=True)
    if args.list:
        return

    # Same fixed val subset as taylor_sensitivity.py (same seed and size).
    val_files = sorted((args.data/'images/val').glob('*.png'))
    subset = [val_files[int(i)] for i in np.random.default_rng(args.seed).permutation(len(val_files))[:args.val_images]]
    (args.output/'val_subset.txt').write_text('\n'.join(str(p) for p in sorted(subset)) + '\n', encoding='utf-8')
    data_yaml = args.output/'kitti_val_subset.yaml'
    data_yaml.write_text(yaml.safe_dump({'path': str(args.data), 'train': 'images/train',
                                         'val': str(args.output/'val_subset.txt'), 'names': KITTI,
                                         'nc': len(KITTI)}), encoding='utf-8')
    validate = SubsetValidator(data_yaml, args.batch)
    batches = bn_batches(args.data, args.bn_images, args.bn_batch, args.seed, base.args)

    ref_plain = validate(base)
    ref_model = copy.deepcopy(base)
    adapt_bn(ref_model, batches, device)
    ref = validate(ref_model)
    base_params = sum(p.numel() for p in base.parameters())
    base_macs = sum(c.weight.numel() * hw for _, c, _, hw in layers)       # conv MACs (all convs scanned)
    print(f'Unfactorised on {len(subset)} val images: {ref_plain}\n  after adaptive BN: {ref}', flush=True)

    out_csv = args.output/'sensitivity.csv'
    done = set()
    if out_csv.is_file():
        with out_csv.open(encoding='utf-8-sig') as stream:
            done = {(r['layer'], int(r['rank'])) for r in csv.DictReader(stream)}
    else:
        with out_csv.open('w', encoding='utf-8-sig', newline='') as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writeheader()
    trials = [t for t in plan if t not in done]
    print(f'{len(trials)} trials to run ({len(done)} already done)', flush=True)
    manifest = {'source': str(args.weights), 'source_sha256': sha256(args.weights), 'torch': torch.__version__,
                'method': 'truncated SVD of W reshaped to Cout x Cin*kh*kw; kxk conv Cin->r then 1x1 conv r->Cout, '
                          'singular values split as sqrt into both factors', 'rank_step': args.step,
                'layers': len(layers), 'excluded': 'model.22.dfl.conv (fixed)',
                'adaptive_bn': {'images': sum(len(b) for b in batches), 'split': 'KITTI train', 'batch': args.bn_batch,
                                'augment': 'training pipeline, checkpoint hyperparameters (mosaic on), generated once',
                                'statistics': 'reset, cumulative average', 'seed': args.seed},
                'val_subset': {'images': len(subset), 'split': 'KITTI val', 'seed': args.seed,
                               'file': 'val_subset.txt'},
                'reference_plain': ref_plain, 'reference_adaptive_bn': ref, 'model_params': base_params,
                'conv_gmacs': base_macs / 1e9, 'fine_tuned': False}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    info = {n: (c, hw) for n, c, _, hw in layers}
    begin = time.perf_counter()
    for k, (name, rank) in enumerate(trials):
        start = time.perf_counter()
        conv, hw = info[name]
        cout, cin, kh, kw = conv.weight.shape
        model = copy.deepcopy(base)
        energy, err, block = apply(model, name, rank)
        plain_model = copy.deepcopy(model)
        fold_bn(plain_model.get_submodule(name.rpartition('.')[0]) if block is not None else None)
        plain = validate(plain_model)
        adapt_bn(model, batches, device)
        fold_bn(model.get_submodule(name.rpartition('.')[0]) if block is not None else None)
        metrics = validate(model)
        params_orig, params_new = conv.weight.numel(), rank * (cin * kh * kw + cout)
        record = {'layer': name, 'cout': cout, 'cin': cin, 'kernel': f'{kh}x{kw}', 'full_rank': min(cout, cin * kh * kw),
                  'rank': rank, 'energy_kept': energy, 'rel_error': err, 'params_orig': params_orig,
                  'params_new': params_new, 'macs_orig': params_orig * hw, 'macs_new': params_new * hw,
                  'model_params': base_params - params_orig + params_new,
                  'model_gmacs': (base_macs - (params_orig - params_new) * hw) / 1e9,
                  'plain_mAP50-95': plain['mAP50-95'], 'plain_mAP50': plain['mAP50'],
                  'plain_drop_mAP50-95': 100 * (ref_plain['mAP50-95'] - plain['mAP50-95']), **metrics,
                  'drop_mAP50-95': 100 * (ref['mAP50-95'] - metrics['mAP50-95']),
                  'drop_mAP50': 100 * (ref['mAP50'] - metrics['mAP50']), 'seconds': time.perf_counter() - start}
        with out_csv.open('a', encoding='utf-8-sig', newline='') as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writerow(record)
        elapsed = time.perf_counter() - begin
        print(f"[{k + 1}/{len(trials)}] {name:<24} r={rank:<4}/{record['full_rank']:<4} energy={energy:.3f} "
              f"plain drop={record['plain_drop_mAP50-95']:+.2f}pp  adaptBN drop={record['drop_mAP50-95']:+.2f}pp "
              f"ETA {elapsed / (k + 1) * (len(trials) - k - 1) / 60:.0f} min", flush=True)

    with out_csv.open(encoding='utf-8-sig') as stream:
        results = list(csv.DictReader(stream))
    order = [n for n, *_ in layers]
    ranks = sorted({int(r['rank']) for r in results})
    pivot = {}
    for r in sorted(results, key=lambda r: order.index(r['layer'])):
        pivot.setdefault(r['layer'], {})[int(r['rank'])] = float(r['drop_mAP50-95'])
    with (args.output/'drop_mAP50-95.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['layer'] + ranks)
        for name, values in pivot.items():
            writer.writerow([name] + [f'{values[q]:.3f}' if q in values else '' for q in ranks])
    print(f'Done: {args.output}')


if __name__ == '__main__':
    main()
