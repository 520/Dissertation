"""Per-group pruning sensitivity of YOLOv8n-KITTI: L2 importance, adaptive BN, small validation subset.

Groups are the 59 Torch-Pruning DepGraph groups of Pruning/dependency_groups.py (C2f.cv1 split into two halves;
Detect output convolutions cv2.*.2 / cv3.*.2 and the DFL projection not pruned). For every group and every ratio,
starting each time from the unpruned model:
  1. importance = Torch-Pruning GroupMagnitudeImportance(p=2): per root channel, the mean over the group's layers
     (coupled conv filters, their BN, consumer conv input slices) of the L2 norm of the weights for that channel;
  2. remove the round(ratio * channels) lowest-importance channels of that group only (at least one channel kept),
     or with --channel-step S: remove S, 2S, ... channels (S at least kept), e.g. 31 points for a 256-channel group;
  3. adaptive BN: reset all BN running statistics and re-estimate them (cumulative average, train-mode BN, no
     gradient, no weight update) on --bn-images KITTI training images with the training augmentation;
  4. mAP on a fixed random subset of --val-images KITTI validation images (imgsz 640, rect=False, conf 0.001).
The reference is the unpruned model after the same adaptive BN step; the unpruned model with its trained BN
statistics is also evaluated on the subset.

Outputs (in --output):
  sensitivity.csv     one row per (group, ratio): channels removed, params, GFLOPs, metrics, drop vs reference
  summary.csv         one row per group: mAP50-95 drop at every ratio, ranked by mean drop
  val_subset.txt      the validation images used
  manifest.json       settings, reference metrics, timing
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

os.environ.setdefault('YOLO_VERBOSE', 'False')

import numpy as np
import torch
import torch_pruning as tp
from torch import nn
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.modules import C2f, Detect

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Pruning.taylor_prune import KITTI, hyp_from, load_model, split_c2f

ROOT = Path(__file__).resolve().parents[1]
OUT_CH = tp.pruner.function.prune_conv_out_channels
FIELDS = ['group', 'root', 'coupled_out_convs', 'channels', 'ratio', 'removed', 'params', 'GFLOPs',
          'mAP50-95', 'mAP50', 'mAP75', 'precision', 'recall', 'drop_mAP50-95', 'drop_mAP50', 'seconds']


def ignored_layers(model):
    head = model.model[-1]
    # .modules(): a low-rank output conv is a Sequential of two convs, both kept
    return [m for s in (*head.cv2, *head.cv3) for m in s[-1].modules() if isinstance(m, nn.Conv2d)] + [head.dfl]


def groups_of(model, example):
    """DepGraph groups as in Pruning/dependency_groups.py (ignored layers stay in the traced graph)."""
    graph = tp.DependencyGraph().build_dependency(model, example_inputs=example, verbose=False)
    return graph, list(graph.get_all_groups(ignored_layers=ignored_layers(model), root_module_types=[nn.Conv2d]))


def bn_images(data_dir, count, seed, hyp):
    """Training-augmented images (mosaic etc., the model's training hyperparameters), generated once with a fixed
    seed. The trained BN statistics come from this distribution: re-estimating them on plain letterboxed KITTI
    frames costs the unpruned model 3.4 mAP50-95 on the subset, augmented images about none."""
    data = {'path': str(data_dir), 'train': 'images/train', 'val': 'images/val', 'names': KITTI, 'nc': len(KITTI)}
    dataset = YOLODataset(img_path=str(data_dir/'images/train'), data=data, imgsz=640, augment=True, rect=False,
                          cache=False, stride=32, batch_size=1, hyp=hyp, prefix='Adaptive BN: ')
    random.seed(seed)
    order = np.random.default_rng(seed).permutation(len(dataset))[:count]
    return torch.stack([dataset[int(i)]['img'] for i in order])


@torch.no_grad()
def adapt_bn(model, images, batch, device):
    """Re-estimate every BN's running mean/var on the given images; all weights stay unchanged."""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None                     # cumulative moving average over all batches
    model.eval()
    for bn in bns:
        bn.train()
    for k in range(0, len(images), batch):
        model(images[k:k + batch].to(device).float() / 255)
    model.eval()


class Evaluator:
    """Ultralytics DetectionValidator on a fixed image list; the dataloader is built once and reused."""

    def __init__(self, data_yaml, batch, save_dir):
        self.args = dict(data=str(data_yaml), split='val', imgsz=640, batch=batch, conf=0.001, iou=0.7, rect=False,
                         device=0, workers=0, cache='ram', plots=False, verbose=False, save_json=False, half=False)
        self.save_dir, self.dataloader = save_dir, None

    def __call__(self, model):
        validator = DetectionValidator(dataloader=self.dataloader, save_dir=self.save_dir, args=self.args)
        validator(model=deepcopy(model))      # AutoBackend fuses BN into the convolutions of the model it gets
        self.dataloader = validator.dataloader
        box = validator.metrics.box
        return {'mAP50-95': box.map, 'mAP50': box.map50, 'mAP75': box.map75, 'precision': box.mp, 'recall': box.mr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/baseline/8n_float32.pt')
    parser.add_argument('--data', type=Path, default=ROOT/'PTQ/datasets/kitti')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/sensitivity_l2')
    parser.add_argument('--ratios', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument('--bn-images', type=int, default=512, help='KITTI training images for adaptive BN')
    parser.add_argument('--val-images', type=int, default=500, help='KITTI validation images for mAP')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--groups', type=int, nargs='*', default=None, help='only these group indices')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--channel-step', type=int, default=0,
                        help='remove step, 2*step, ... channels per group (at least step kept) instead of --ratios')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')

    base, _ = load_model(args.weights)
    split_c2f(base)
    for p in base.parameters():
        p.requires_grad_(True)     # DepGraph traces autograd; frozen leading convs would be left out of the graph
    base.to(device).eval()
    example = torch.zeros(1, 3, 640, 640, device=device)
    _, base_groups = groups_of(base, example)
    names = {m: n for n, m in base.named_modules()}
    info = []
    for group in base_groups:
        coupled = [names[d.target.module] for d, _ in group
                   if d.pruning_fn == OUT_CH and isinstance(d.target.module, nn.Conv2d)]
        info.append({'root': names[group[0].dep.target.module], 'coupled_out_convs': ' + '.join(coupled),
                     'channels': len(group[0].idxs)})
    print(f'{len(info)} groups', flush=True)
    base_macs, base_params = tp.utils.count_ops_and_params(base, example)

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
    reference = evaluate(reference_model)
    del reference_model
    print(f'Unpruned, trained BN: {trained_bn}\nUnpruned, adaptive BN (reference): {reference}', flush=True)

    importance = tp.importance.GroupMagnitudeImportance(p=2, group_reduction='mean', normalizer=None)
    out_csv = args.output/'sensitivity.csv'
    rows = []
    selected = range(len(info)) if args.groups is None else args.groups
    for g in selected:
        channels = info[g]['channels']
        removals = (list(range(args.channel_step, channels, args.channel_step)) if args.channel_step
                    else [min(int(round(r * channels)), channels - 1) for r in args.ratios])
        for n in removals:
            ratio = round(n / channels, 5)
            start = time.perf_counter()
            model = deepcopy(base)
            graph, groups = groups_of(model, example)
            group = groups[g]
            root = group[0].dep.target.module
            if dict(model.named_modules()).get(info[g]['root']) is not root or len(groups) != len(info):
                raise RuntimeError(f'Group {g} does not match between copies')
            scores = importance(group)
            idxs = torch.argsort(scores)[:n].tolist()
            graph.get_pruning_group(root, OUT_CH, idxs).prune()
            for m in model.modules():
                if isinstance(m, Detect):
                    m.shape = None
            macs, params = tp.utils.count_ops_and_params(model, example)
            adapt_bn(model, calib, args.batch, device)
            metrics = evaluate(model)
            row = {'group': g, **info[g], 'ratio': ratio, 'removed': n, 'params': params, 'GFLOPs': round(2*macs/1e9, 4),
                   **{k: round(v, 5) for k, v in metrics.items()},
                   'drop_mAP50-95': round(reference['mAP50-95'] - metrics['mAP50-95'], 5),
                   'drop_mAP50': round(reference['mAP50'] - metrics['mAP50'], 5),
                   'seconds': round(time.perf_counter() - start, 1)}
            rows.append(row)
            with out_csv.open('w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            print(f"group {g:2d} {info[g]['root']:<26} ratio {ratio:.2f} removed {n:3d}/{channels:<3d} "
                  f"GFLOPs {row['GFLOPs']:.3f} mAP50-95 {metrics['mAP50-95']:.4f} drop {row['drop_mAP50-95']:+.4f} "
                  f"({row['seconds']}s)", flush=True)
            del model, graph, groups, group

    def safe_removal(mine, limit):
        """Channels removable before the first point whose drop exceeds limit (mAP50-95 points)."""
        best = 0
        for r in mine:
            if 100 * r['drop_mAP50-95'] > limit:
                break
            best = r['removed']
        return best

    summary = []
    for g in selected:
        mine = [r for r in rows if r['group'] == g]
        if not mine:
            continue                           # group of channel_step channels or fewer: nothing to remove
        per_point = {} if args.channel_step else {f"drop@{r['ratio']:g}": r['drop_mAP50-95'] for r in mine}
        summary.append({'group': g, **info[g], **per_point,
                        'mean_drop': round(float(np.mean([r['drop_mAP50-95'] for r in mine])), 5),
                        'removable@0.5pt': safe_removal(mine, 0.5), 'removable@1pt': safe_removal(mine, 1.0),
                        'GFLOPs_saved@max_removed': round(2*base_macs/1e9 - mine[-1]['GFLOPs'], 4)})
    summary.sort(key=lambda r: -r['mean_drop'])
    with (args.output/'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    manifest = {'source': str(args.weights.relative_to(ROOT)), 'tool': f'torch-pruning {tp.__version__}',
                'importance': 'GroupMagnitudeImportance(p=2, group_reduction=mean, normalizer=None)',
                'groups': len(info), 'c2f_cv1_split': True,
                'ignored': ['model.22.cv2.*.2', 'model.22.cv3.*.2', 'model.22.dfl'],
                'ratios': None if args.channel_step else args.ratios,
                'channel_step': args.channel_step or None,
                'channels_removed': (f'{args.channel_step}, {2*args.channel_step}, ... below the group size'
                                     if args.channel_step else 'round(ratio*channels), at least one kept'),
                'adaptive_bn': {'images': args.bn_images, 'split': 'KITTI train', 'augment': 'training hyperparameters (mosaic etc.)', 'seed': args.seed,
                                'momentum': None, 'batch': args.batch},
                'validation': {'images': len(val_files), 'split': 'KITTI val', 'seed': args.seed, 'imgsz': 640,
                               'rect': False, 'conf': 0.001, 'iou': 0.7},
                'base_params': base_params, 'base_GFLOPs': 2*base_macs/1e9,
                'unpruned_trained_bn': trained_bn, 'reference_unpruned_adaptive_bn': reference,
                'seconds': time.perf_counter() - begin}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Done: {args.output}')


if __name__ == '__main__':
    main()
