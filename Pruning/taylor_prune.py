"""Per-channel Taylor-1 scoring of YOLOv8n-KITTI and 25 % per-layer structured pruning with Torch-Pruning.

Taylor-1 (same definition as PTQ/channel_importance.py):
    score[c] = mean over KITTI training images of | sum_{h,w} A[c,h,w] * dL/dA[c,h,w] |
where A is the convolution output after BatchNorm and before SiLU (= the BN-fused convolution output) and
L = 7.5*box + 0.5*cls + 1.5*DFL (Ultralytics v8DetectionLoss), one image per backward pass, no augmentation,
FP32 without TF32, BN in eval mode, no weight update.

Pruning: Torch-Pruning DepGraph, global_pruning=False, so every dependency group (a convolution's output
channels together with all layers that must change with them) loses exactly --ratio of its channels. Channels
coupled by a residual addition form one group and are ranked by the mean Taylor-1 of the coupled convolutions.
The Detect output convolutions (cv2.*.2 box, cv3.*.2 class) and the fixed DFL projection are not pruned, so the
output format (1, 4+nc, 8400) is unchanged. C2f.cv1 is split into two convolutions while pruning (its output is
chunked into two halves that play different roles) and merged back afterwards, so the saved model contains only
standard Ultralytics modules. No fine-tuning is done here.

Outputs (in --output):
  <stem>.pt               pruned model, Ultralytics checkpoint format ({'model': DetectionModel, ...})
  channel_scores.csv      every conv output channel: Taylor-1, rank within layer, pruned yes/no
  layers.csv              per conv: channels before/after, group, kept fraction
  manifest.json           settings, checks against the 2026-10-14 scores, params/GFLOPs, val metrics
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch_pruning as tp
from torch import nn
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.modules import C2f, Conv, Detect
from ultralytics.utils import SETTINGS
from ultralytics.utils.loss import v8DetectionLoss

ROOT = Path(__file__).resolve().parents[1]
KITTI = {0: 'car', 1: 'van', 2: 'truck', 3: 'pedestrian', 4: 'person_sitting', 5: 'cyclist', 6: 'tram', 7: 'misc'}


def slice_conv(block: Conv, idx: slice) -> Conv:
    """Copy of an Ultralytics Conv (conv+BN+act) keeping only output channels idx; for a low-rank block
    (Decomposition TuckerConv, conv is a Sequential) only its last convolution is sliced."""
    new = deepcopy(block)
    src = block.conv[-1] if isinstance(block.conv, nn.Sequential) else block.conv
    dst = new.conv[-1] if isinstance(new.conv, nn.Sequential) else new.conv
    dst.weight = nn.Parameter(src.weight.data[idx].clone())
    dst.out_channels = dst.weight.shape[0]
    bn = new.bn
    bn.weight = nn.Parameter(block.bn.weight.data[idx].clone())
    bn.bias = nn.Parameter(block.bn.bias.data[idx].clone())
    bn.running_mean = block.bn.running_mean[idx].clone()
    bn.running_var = block.bn.running_var[idx].clone()
    bn.num_features = bn.weight.shape[0]
    return new


def concat_convs(a: Conv, b: Conv) -> Conv:
    new = deepcopy(a)
    new.conv.weight = nn.Parameter(torch.cat([a.conv.weight.data, b.conv.weight.data]))
    new.conv.out_channels = new.conv.weight.shape[0]
    for key in ('weight', 'bias'):
        setattr(new.bn, key, nn.Parameter(torch.cat([getattr(a.bn, key).data, getattr(b.bn, key).data])))
    new.bn.running_mean = torch.cat([a.bn.running_mean, b.bn.running_mean])
    new.bn.running_var = torch.cat([a.bn.running_var, b.bn.running_var])
    new.bn.num_features = new.bn.weight.shape[0]
    return new


class C2fSplit(nn.Module):
    """C2f with cv1 split into cv0 (first half, straight to the concat) and cv1 (second half, into the
    bottlenecks); numerically identical to C2f, but lets each half be pruned as its own group."""

    def __init__(self, m: C2f):
        super().__init__()
        self.original = [m]                       # list: not registered as a submodule
        self.cv0, self.cv1 = slice_conv(m.cv1, slice(0, m.c)), slice_conv(m.cv1, slice(m.c, 2 * m.c))
        self.cv2, self.m = m.cv2, m.m
        for key in ('f', 'i', 'type', 'np'):
            setattr(self, key, getattr(m, key))

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(b(y[-1]) for b in self.m)
        return self.cv2(torch.cat(y, 1))

    def merge(self) -> C2f:
        if self.cv0.conv.out_channels != self.cv1.conv.out_channels:
            raise RuntimeError(f'C2f halves pruned unequally: {self.cv0.conv.out_channels} vs {self.cv1.conv.out_channels}')
        m = self.original[0]
        m.cv1, m.cv2, m.m, m.c = concat_convs(self.cv0, self.cv1), self.cv2, self.m, self.cv0.conv.out_channels
        return m


def split_c2f(model):
    for i, m in enumerate(model.model):
        if isinstance(m, C2f):
            model.model[i] = C2fSplit(m)


def merge_c2f(model):
    for i, m in enumerate(model.model):
        if isinstance(m, C2fSplit):
            model.model[i] = m.merge()


def load_model(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = (ckpt.get('ema') or ckpt['model']) if isinstance(ckpt, dict) else ckpt
    return model.float().eval(), (ckpt if isinstance(ckpt, dict) else {})


def hyp_from(saved):
    hyp = get_cfg()
    for key, value in (saved or {}).items():
        if hasattr(hyp, key):
            setattr(hyp, key, value)
    return hyp


def taylor1_scores(model, data_dir, device, workers, limit):
    """Per-channel Taylor-1 for every Conv block (hook on BN output = fused conv output before SiLU)."""
    data = {'path': str(data_dir), 'train': 'images/train', 'val': 'images/val', 'names': KITTI, 'nc': len(KITTI)}
    dataset = YOLODataset(img_path=str(data_dir/'images/train'), data=data, imgsz=640, augment=False, rect=False,
                          cache=False, stride=32, batch_size=1, hyp=get_cfg(), prefix='Taylor-1: ')
    count = min(limit, len(dataset)) if limit else len(dataset)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, range(count)), batch_size=1, shuffle=False,
                                         num_workers=workers, pin_memory=True, collate_fn=YOLODataset.collate_fn)
    blocks = {name: m for name, m in model.named_modules() if isinstance(m, Conv)}
    sums = {name: torch.zeros(m.conv.out_channels, dtype=torch.float64, device=device) for name, m in blocks.items()}

    def hook(name):
        def forward_hook(_module, _inputs, output):
            activation = output.detach().clone()

            def grad_hook(gradient):
                sums[name] += (activation * gradient.detach()).sum(dim=(0, 2, 3), dtype=torch.float64).abs()
            output.register_hook(grad_hook)
        return forward_hook

    handles = [m.bn.register_forward_hook(hook(name)) for name, m in blocks.items()]
    model.to(device).eval()
    model.model[-1].training = True              # raw head outputs for the loss; BN stays in eval mode
    criterion = v8DetectionLoss(model)
    for p in model.parameters():
        p.requires_grad_(True)
    begin = time.perf_counter()
    for k, raw in enumerate(loader):
        batch = {'img': raw['img'].to(device).float() / 255, 'cls': raw['cls'].to(device),
                 'bboxes': raw['bboxes'].to(device), 'batch_idx': raw['batch_idx'].to(device)}
        model.zero_grad(set_to_none=True)
        loss, _ = criterion(model(batch['img']), batch)
        loss.sum().backward()
        if (k + 1) % 1000 == 0:
            print(f'  scored {k + 1}/{count} images ({time.perf_counter() - begin:.0f}s)', flush=True)
    for h in handles:
        h.remove()
    model.model[-1].training = False
    model.zero_grad(set_to_none=True)
    return {blocks[n].conv: (s / count).cpu() for n, s in sums.items()}, count


class ActivationTaylorImportance(tp.importance.GroupMagnitudeImportance):
    """Group importance = mean Taylor-1 of the convolutions whose output channels are in the group."""

    def __init__(self, scores):
        super().__init__(p=1, group_reduction='mean', normalizer=None)
        self.scores = scores

    def __call__(self, group):
        imps, root = [], []
        for i, (dep, idxs) in enumerate(group):
            if dep.pruning_fn == tp.pruner.function.prune_conv_out_channels and dep.layer in self.scores:
                imps.append(self.scores[dep.layer][idxs].float())
                root.append(group[i].root_idxs)
        return self._reduce(imps, root) if imps else None


def compare_with_reference(names, scores, reference_csv):
    """Map split C2f halves back to the original fused-conv names and compare with an earlier scoring run."""
    if not reference_csv.is_file():
        return {'reference': str(reference_csv), 'status': 'not found'}
    ref = {}
    with reference_csv.open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if row['taylor1']:
                ref.setdefault(row['layer'], {})[int(row['channel'])] = float(row['taylor1'])
    a, b, worst = [], [], 0.0
    for conv, name in names.items():
        if conv not in scores:
            continue
        layer, offset = merged_name(name, scores[conv].numel())
        for c, v in enumerate(scores[conv].tolist()):
            r = ref.get(layer, {}).get(c + offset)
            if r is not None:
                a.append(v); b.append(r)
                worst = max(worst, abs(v - r) / max(abs(r), 1e-12))
    a, b = np.array(a), np.array(b)
    rank = lambda x: np.argsort(np.argsort(x))
    return {'reference': str(reference_csv.relative_to(ROOT)), 'channels_compared': int(a.size),
            'pearson': float(np.corrcoef(a, b)[0, 1]), 'spearman': float(np.corrcoef(rank(a), rank(b))[0, 1]),
            'max_relative_difference': float(worst)}


SPLIT_INDICES: set[str] = set()


def merged_name(name, half_channels):
    """'model.2.cv0.conv' -> ('model.2.cv1.conv', 0); 'model.2.cv1.conv' -> ('model.2.cv1.conv', c) for split
    C2f modules; any other conv keeps its name with offset 0."""
    parts = name.split('.')
    if len(parts) == 4 and parts[1] in SPLIT_INDICES and parts[2] in ('cv0', 'cv1'):
        return f'model.{parts[1]}.cv1.conv', 0 if parts[2] == 'cv0' else half_channels
    return name, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/baseline/8n_float32.pt')
    parser.add_argument('--data', type=Path, default=ROOT/'PTQ/datasets/kitti')
    parser.add_argument('--output', type=Path, default=ROOT/'models/pruning/taylor1_25')
    parser.add_argument('--ratio', type=float, default=0.25, help='fraction of channels removed per layer/group')
    parser.add_argument('--limit', type=int, default=0, help='scoring images; 0 = all KITTI training images')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--reference', type=Path,
                        default=ROOT/'sensitivity_results/kitti_channel_scores_20261014/channels.csv')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    stem = f'{args.weights.stem}_taylor1_prune{round(args.ratio * 100)}'

    model, ckpt = load_model(args.weights)
    saved_args = model.args
    model.args = hyp_from(saved_args)
    example = torch.zeros(1, 3, 640, 640)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example)
    with torch.no_grad():
        reference_out = model(example)[0]
    SPLIT_INDICES.update(str(i) for i, m in enumerate(model.model) if isinstance(m, C2f))
    split_c2f(model)
    with torch.no_grad():
        split_error = float((model(example)[0] - reference_out).abs().max())
    print(f'C2f split check: max |diff| = {split_error:.2e}', flush=True)

    names = {m.conv: name + '.conv' for name, m in model.named_modules() if isinstance(m, Conv)}
    original_channels = {conv: conv.out_channels for conv in names}
    print('Scoring Taylor-1 on KITTI training images...', flush=True)
    scores, n_images = taylor1_scores(model, args.data, device, args.workers, args.limit)
    check = compare_with_reference(names, scores, args.reference)
    print(f'Comparison with earlier scores: {check}', flush=True)

    model.cpu().eval()
    head = model.model[-1]
    ignored = [s[-1] for s in head.cv2] + [s[-1] for s in head.cv3] + [head.dfl]
    pruner = tp.pruner.BasePruner(model, example, importance=ActivationTaylorImportance(scores),
                                  pruning_ratio=args.ratio, global_pruning=False, ignored_layers=ignored)
    pruned, groups = {}, []
    for g, group in enumerate(pruner.step(interactive=True)):
        members = []
        for dep, idxs in group:
            if dep.pruning_fn == tp.pruner.function.prune_conv_out_channels and dep.layer in names:
                pruned[dep.layer] = sorted(int(i) for i in idxs)
                members.append(names[dep.layer])
        groups.append(members)
        group.prune()
    for m in model.modules():
        if isinstance(m, Detect):
            m.shape = None                         # rebuild anchors for the pruned model
    merge_c2f(model)

    # Per-conv report under the final (merged) names; cv0/cv1 halves map back to C2f.cv1 channels.
    rows, layers = [], []
    group_of = {name: k for k, members in enumerate(groups) for name in members}
    for conv, name in names.items():
        if conv not in scores:
            continue
        final, offset = merged_name(name, original_channels[conv])
        s = scores[conv]
        order = torch.argsort(s, descending=True).tolist()
        rank = {c: r + 1 for r, c in enumerate(order)}
        cut = set(pruned.get(conv, []))
        for c in range(s.numel()):
            rows.append({'layer': final, 'channel': c + offset, 'split_half': name if final != name else '',
                         'taylor1': float(s[c]), 'rank_within_layer': rank[c], 'pruned': c in cut,
                         'group': group_of.get(name, '')})
        layers.append({'layer': final, 'split_half': name if final != name else '', 'group': group_of.get(name, ''),
                       'group_members': ' + '.join(groups[group_of[name]]) if name in group_of else '',
                       'channels_before': original_channels[conv], 'pruned': len(cut),
                       'channels_after': original_channels[conv] - len(cut),
                       'pruned_fraction': len(cut) / original_channels[conv],
                       'pruned_are_lowest_taylor1': cut == set(order[len(order) - len(cut):]) if cut else None})
    for path, table in ((args.output/'channel_scores.csv', rows), (args.output/'layers.csv', layers)):
        with path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)

    prunable = [l for l in layers if l['pruned']]
    untouched = [l['layer'] for l in layers if not l['pruned']]
    macs, params = tp.utils.count_ops_and_params(model, example)
    with torch.no_grad():
        out_shape = tuple(model(example)[0].shape)
    model.args = saved_args
    out = args.output/f'{stem}.pt'
    torch.save({'model': model, 'train_args': ckpt.get('train_args', {}), 'date': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'pruning': {'method': 'Torch-Pruning DepGraph, activation Taylor-1', 'ratio_per_layer': args.ratio}}, out)
    print(f'Saved {out}: params {base_params/1e6:.3f}M -> {params/1e6:.3f}M, '
          f'GFLOPs {2*base_macs/1e9:.2f} -> {2*macs/1e9:.2f}, output {out_shape}', flush=True)

    SETTINGS.update({'datasets_dir': str(args.data.parent)})
    metrics = YOLO(str(out)).val(data='kitti.yaml', split='val', imgsz=640, batch=1, device=0, rect=False, workers=0,
                                 plots=False, save_json=False, verbose=False, project=str(args.output/'val'),
                                 name='no_finetune', exist_ok=True).box
    val = {'mAP50-95': metrics.map, 'mAP75': metrics.map75, 'mAP50': metrics.map50,
           'precision': metrics.mp, 'recall': metrics.mr}
    print(f'Pruned, no fine-tuning: {val}', flush=True)

    manifest = {'source': str(args.weights.relative_to(ROOT)), 'tool': f'torch-pruning {tp.__version__}',
                'importance': 'activation Taylor-1, mean |sum_hw A*dL/dA| over images, BN output before SiLU; '
                              'group score = mean over coupled convolutions',
                'scoring_images': n_images, 'scoring_split': 'KITTI train', 'loss_gains': {k: getattr(hyp_from(saved_args), k) for k in ('box', 'cls', 'dfl')},
                'ratio_per_layer': args.ratio, 'global_pruning': False, 'round_to': None,
                'ignored': ['model.22.cv2.*.2', 'model.22.cv3.*.2', 'model.22.dfl'],
                'c2f_split_check_max_abs_diff': split_error, 'reference_check': check,
                'groups': len(groups), 'pruned_convs': len(prunable), 'unpruned_convs': untouched,
                'all_pruned_channels_are_lowest_taylor1_in_single_conv_groups':
                    all(l['pruned_are_lowest_taylor1'] for l in prunable if ' + ' not in l['group_members']),
                'params_before': base_params, 'params_after': params,
                'gflops_before': 2*base_macs/1e9, 'gflops_after': 2*macs/1e9, 'output_shape': out_shape,
                'val_no_finetune': val, 'checkpoint': out.name}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Done: {args.output}')


if __name__ == '__main__':
    main()
