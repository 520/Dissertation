"""Per-layer Tucker-2 sensitivity of YOLOv8n-KITTI: rank step 8, small validation subset, adaptive BN when needed.

Every Ultralytics Conv block with a k x k kernel (k > 1) is a candidate (1x1 convs and the Detect output convs are
not decomposed). For every layer, both modes are swept separately, starting each time from the original model:
  mode 'out': R_out = 8, 16, ..., C_out with R_in = C_in (full);
  mode 'in' : R_in  = 8, 16, ..., C_in  with R_out = C_out (full)  (skipped when C_in <= 8, i.e. model.0).
The kernel W (C_out, C_in, k, k) is replaced by its Tucker-2 approximation
  W ~= G x_0 U_out x_1 U_in,  U_out (C_out, R_out), U_in (C_in, R_in), core G (R_out, R_in, k, k),
computed by truncated HOSVD of the mode-0/mode-1 unfoldings refined by --hooi-iters HOOI sweeps (with one mode at
full rank this is the exact Frobenius-optimal truncated SVD). It runs as conv1x1(C_in->R_in) -> core kxk (stride,
padding as the original) -> conv1x1(R_out->C_out), followed by the original BN and SiLU; a factor whose mode is at
full rank is folded into the core, so a one-mode scheme is two convolutions.
Each scheme is evaluated on a fixed random subset of --val-images KITTI validation images (imgsz 640, rect=False,
conf 0.001) with the trained BN statistics ("raw"). If the raw mAP50-95 drop exceeds --calib-threshold, all BN
running statistics are re-estimated (adaptive BN, as Pruning/sensitivity_l2.py) on --bn-images augmented KITTI
training images and the model is evaluated again ("cal", drop against the unmodified model after the same step).

Outputs (in --output):
  sensitivity.csv     one row per (layer, mode, rank)
  summary.csv         one row per (layer, mode): smallest rank within each drop tolerance, mean drop
  val_subset.txt      the validation images used
  manifest.json       settings, reference metrics, timing
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
from torch import nn
from ultralytics.nn.modules import Conv

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Pruning.sensitivity_l2 import Evaluator, adapt_bn, bn_images
from Pruning.taylor_prune import KITTI, hyp_from, load_model

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['layer', 'kernel', 'stride', 'c_in', 'c_out', 'mode', 'rank', 'r_in', 'r_out', 'ratio', 'rel_error',
          'layer_params', 'layer_params_base', 'layer_GFLOPs', 'layer_GFLOPs_base', 'model_GFLOPs',
          'mAP50-95_raw', 'mAP50_raw', 'drop_raw', 'calibrated', 'mAP50-95_cal', 'mAP50_cal', 'drop_cal',
          'drop', 'seconds']
TOLERANCES = (0.002, 0.005, 0.01)


class TuckerConv(nn.Module):
    """Ultralytics Conv with its convolution replaced by a Tucker-2 chain (not a Conv, so fuse() leaves it alone)."""

    def __init__(self, block: Conv, convs: list[nn.Conv2d]):
        super().__init__()
        self.conv, self.bn, self.act = nn.Sequential(*convs), block.bn, block.act
        for key in ('f', 'i', 'type', 'np'):
            if hasattr(block, key):
                setattr(self, key, getattr(block, key))

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def leading(matrix, rank):
    return torch.linalg.svd(matrix, full_matrices=False)[0][:, :rank]


def tucker2(weight, r_out, r_in, hooi_iters):
    """Partial Tucker on modes 0 (out) and 1 (in); a factor is None when its mode is kept at full rank."""
    w = weight.double()
    c_out, c_in = w.shape[:2]
    u_out = leading(w.reshape(c_out, -1), r_out) if r_out < c_out else None
    u_in = leading(w.transpose(0, 1).reshape(c_in, -1), r_in) if r_in < c_in else None
    if u_out is not None and u_in is not None:
        for _ in range(hooi_iters):
            u_out = leading(torch.einsum('oihw,ir->orhw', w, u_in).reshape(c_out, -1), r_out)
            u_in = leading(torch.einsum('oihw,os->ishw', w, u_out).reshape(c_in, -1), r_in)
    core = w if u_out is None else torch.einsum('oihw,os->sihw', w, u_out)
    core = core if u_in is None else torch.einsum('sihw,ir->srhw', core, u_in)
    approx = core if u_in is None else torch.einsum('srhw,ir->sihw', core, u_in)
    approx = approx if u_out is None else torch.einsum('sihw,os->oihw', approx, u_out)
    rel_error = (torch.linalg.norm(approx - w) / torch.linalg.norm(w)).item()
    return u_out, u_in, core, rel_error


def decompose(block: Conv, r_out, r_in, hooi_iters):
    conv = block.conv
    if conv.bias is not None or conv.groups != 1:
        raise NotImplementedError(f'conv with bias or groups: {conv}')
    u_out, u_in, core, rel_error = tucker2(conv.weight.data, r_out, r_in, hooi_iters)
    dtype, device = conv.weight.dtype, conv.weight.device
    convs = []
    if u_in is not None:
        first = nn.Conv2d(conv.in_channels, r_in, 1, bias=False)
        first.weight.data = u_in.T[:, :, None, None].to(device, dtype).contiguous()
        convs.append(first)
    middle = nn.Conv2d(core.shape[1], core.shape[0], conv.kernel_size, conv.stride, conv.padding, conv.dilation, bias=False)
    middle.weight.data = core.to(device, dtype).contiguous()
    convs.append(middle)
    if u_out is not None:
        last = nn.Conv2d(r_out, conv.out_channels, 1, bias=False)
        last.weight.data = u_out[:, :, None, None].to(device, dtype).contiguous()
        convs.append(last)
    return TuckerConv(block, convs), rel_error


def chain_cost(convs, hw_in, hw_out):
    """(params, MACs) of a list of convolutions; 1x1 convs before the strided core run at the input resolution."""
    params = macs = 0
    hw, seen_core = hw_in, False
    for c in convs:
        n = c.weight.numel()
        if c.kernel_size != (1, 1) or c.stride != (1, 1):
            seen_core, hw = True, hw_out
        params, macs = params + n, macs + n * (hw if seen_core else hw_in)
    return params, macs


def set_module(model, name, module):
    parent, _, child = name.rpartition('.')
    setattr(model.get_submodule(parent) if parent else model, child, module)


def spatial_sizes(model, example):
    """Input/output H*W of every conv block and the total conv MACs of the model at 640x640."""
    sizes, macs, hooks = {}, [0], []
    names = {m: n for n, m in model.named_modules()}
    for m in model.modules():
        if isinstance(m, Conv):
            hooks.append(m.register_forward_hook(
                lambda mod, i, o: sizes.__setitem__(names[mod], (i[0].shape[-2] * i[0].shape[-1], o.shape[-2] * o.shape[-1]))))
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(
                lambda mod, i, o: macs.__setitem__(0, macs[0] + mod.weight.numel() * o.shape[-2] * o.shape[-1])))
    with torch.no_grad():
        model(example)
    for h in hooks:
        h.remove()
    return sizes, macs[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/baseline/8n_float32.pt')
    parser.add_argument('--data', type=Path, default=ROOT/'PTQ/datasets/kitti')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/sensitivity_tucker2')
    parser.add_argument('--step', type=int, default=8, help='rank step; ranks step, 2*step, ..., C')
    parser.add_argument('--hooi-iters', type=int, default=10)
    parser.add_argument('--calib-threshold', type=float, default=0.002,
                        help='adaptive BN when the raw mAP50-95 drop exceeds this')
    parser.add_argument('--bn-images', type=int, default=512, help='KITTI training images for adaptive BN')
    parser.add_argument('--val-images', type=int, default=500, help='KITTI validation images for mAP')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--layers', nargs='*', default=None, help='only these layer names')
    parser.add_argument('--modes', nargs='+', default=['out', 'in'], choices=['out', 'in'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')

    base, _ = load_model(args.weights)
    base.to(device).eval()
    example = torch.zeros(1, 3, 640, 640, device=device)
    sizes, base_macs = spatial_sizes(base, example)
    layers = [(n, m) for n, m in base.named_modules()
              if isinstance(m, Conv) and m.conv.kernel_size != (1, 1) and m.conv.groups == 1]
    if args.layers:
        layers = [(n, m) for n, m in layers if n in args.layers]
    print(f'{len(layers)} layers, base conv GFLOPs {2*base_macs/1e9:.4f}', flush=True)

    val_files = sorted((args.data/'images/val').glob('*.png')) + sorted((args.data/'images/val').glob('*.jpg'))
    val_files = [val_files[i] for i in sorted(np.random.default_rng(args.seed).permutation(len(val_files))[:args.val_images])]
    subset = args.output/'val_subset.txt'
    subset.write_text('\n'.join(str(f.resolve()) for f in val_files) + '\n', encoding='utf-8')
    data_yaml = args.output/'kitti_val_subset.yaml'
    data_yaml.write_text(json.dumps({'path': str(args.data.resolve()), 'train': 'images/train', 'val': str(subset.resolve()),
                                     'nc': len(KITTI), 'names': KITTI}), encoding='utf-8')
    evaluate = Evaluator(data_yaml, args.batch, args.output/'val')
    calib = bn_images(args.data, args.bn_images, args.seed, hyp_from(base.args))

    begin = time.perf_counter()
    trained_bn = evaluate(base)
    reference_model = deepcopy(base)
    adapt_bn(reference_model, calib, args.batch, device)
    adaptive_bn = evaluate(reference_model)
    del reference_model
    print(f'Original, trained BN (raw reference): {trained_bn}\nOriginal, adaptive BN (cal reference): {adaptive_bn}', flush=True)

    out_csv = args.output/'sensitivity.csv'
    rows = []
    if out_csv.exists():                       # resume
        with out_csv.open(encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        print(f'Resuming: {len(rows)} rows already done', flush=True)
    done = {(r['layer'], r['mode'], int(r['rank'])) for r in rows}

    for name, block in layers:
        conv = block.conv
        c_out, c_in, k = conv.out_channels, conv.in_channels, conv.kernel_size[0]
        hw_in, hw_out = sizes[name]
        base_params, base_layer_macs = chain_cost([conv], hw_in, hw_out)
        for mode in args.modes:
            channels = c_out if mode == 'out' else c_in
            if channels <= args.step:
                continue
            for rank in range(args.step, channels + 1, args.step):
                if (name, mode, rank) in done:
                    continue
                start = time.perf_counter()
                r_out, r_in = (rank, c_in) if mode == 'out' else (c_out, rank)
                model = deepcopy(base)
                new, rel_error = decompose(model.get_submodule(name), r_out, r_in, args.hooi_iters)
                set_module(model, name, new)
                params, macs = chain_cost(list(new.conv), hw_in, hw_out)
                raw = evaluate(model)
                drop_raw = trained_bn['mAP50-95'] - raw['mAP50-95']
                cal, drop_cal = None, None
                if drop_raw > args.calib_threshold:
                    adapt_bn(model, calib, args.batch, device)
                    cal = evaluate(model)
                    drop_cal = adaptive_bn['mAP50-95'] - cal['mAP50-95']
                row = {'layer': name, 'kernel': k, 'stride': conv.stride[0], 'c_in': c_in, 'c_out': c_out, 'mode': mode,
                       'rank': rank, 'r_in': r_in, 'r_out': r_out, 'ratio': round(rank / channels, 5),
                       'rel_error': round(rel_error, 6), 'layer_params': params, 'layer_params_base': base_params,
                       'layer_GFLOPs': round(2*macs/1e9, 5), 'layer_GFLOPs_base': round(2*base_layer_macs/1e9, 5),
                       'model_GFLOPs': round(2*(base_macs - base_layer_macs + macs)/1e9, 5),
                       'mAP50-95_raw': round(raw['mAP50-95'], 5), 'mAP50_raw': round(raw['mAP50'], 5),
                       'drop_raw': round(drop_raw, 5), 'calibrated': cal is not None,
                       'mAP50-95_cal': '' if cal is None else round(cal['mAP50-95'], 5),
                       'mAP50_cal': '' if cal is None else round(cal['mAP50'], 5),
                       'drop_cal': '' if cal is None else round(drop_cal, 5),
                       'drop': round(drop_raw if cal is None else min(drop_raw, drop_cal), 5),
                       'seconds': round(time.perf_counter() - start, 1)}
                rows.append(row)
                with out_csv.open('w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"{name:<22} {mode:<3} rank {rank:3d}/{channels:<3d} err {rel_error:.3f} "
                      f"params {params:7d}/{base_params:<7d} mAP50-95 raw {raw['mAP50-95']:.4f} ({drop_raw:+.4f})"
                      + ('' if cal is None else f" cal {cal['mAP50-95']:.4f} ({drop_cal:+.4f})")
                      + f" ({row['seconds']}s)", flush=True)
                del model, new

    summary = []
    for name, _ in layers:
        for mode in args.modes:
            mine = sorted((r for r in rows if r['layer'] == name and r['mode'] == mode), key=lambda r: int(r['rank']))
            if not mine:
                continue
            drops = [float(r['drop']) for r in mine]
            entry = {'layer': name, 'mode': mode, 'c_in': mine[0]['c_in'], 'c_out': mine[0]['c_out'],
                     'channels': mine[-1]['rank'], 'layer_params_base': mine[0]['layer_params_base'],
                     'layer_GFLOPs_base': mine[0]['layer_GFLOPs_base'],
                     'mean_drop': round(float(np.mean(drops)), 5), 'max_drop': round(max(drops), 5)}
            for tol in TOLERANCES:
                # smallest rank such that it and every larger rank stay within the tolerance
                ok = [int(r['rank']) for i, r in enumerate(mine) if all(d <= tol for d in drops[i:])]
                entry[f'min_rank@{tol:g}'] = ok[0] if ok else ''
            summary.append(entry)
    summary.sort(key=lambda r: -r['mean_drop'])
    if summary:
        with (args.output/'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)

    manifest = {'source': str(args.weights.relative_to(ROOT)), 'method': 'Tucker-2 (partial Tucker on out/in modes), HOSVD + HOOI',
                'hooi_iters': args.hooi_iters, 'layers': [n for n, _ in layers], 'modes': args.modes,
                'ranks': f'{args.step}, {2*args.step}, ..., C (other mode at full rank)',
                'calibration': {'when': f'raw mAP50-95 drop > {args.calib_threshold}', 'method': 'adaptive BN',
                                'images': args.bn_images, 'split': 'KITTI train',
                                'augment': 'training hyperparameters (mosaic etc.)', 'seed': args.seed,
                                'momentum': None, 'batch': args.batch},
                'validation': {'images': len(val_files), 'split': 'KITTI val', 'seed': args.seed, 'imgsz': 640,
                               'rect': False, 'conf': 0.001, 'iou': 0.7},
                'drop': 'drop_raw vs original with trained BN; drop_cal vs original after adaptive BN; drop = min of the two',
                'base_conv_GFLOPs': 2*base_macs/1e9, 'original_trained_bn': trained_bn,
                'original_adaptive_bn': adaptive_bn, 'seconds': time.perf_counter() - begin}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, default=float), encoding='utf-8')
    print(f'Done: {args.output}')


if __name__ == '__main__':
    main()
