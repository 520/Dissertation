"""Global first-order Taylor channel pruning of YOLOv8n-KITTI with Torch-Pruning.

1. Taylor-1 score of every convolution output channel on all KITTI training images, as in channel_importance.py:
   mean over images of |sum_hw A * dL/dA|, A = pre-activation Conv+BN output (BN in eval mode, so equal to the
   fused convolution), L = Ultralytics v8 detection loss, one backward pass per image, no augmentation.
2. C2f -> C2f_v2 (cv1 split into cv0/cv1, numerically identical) so the two halves can be pruned independently.
3. Torch-Pruning dependency graph: channels coupled by residual adds/concats form one group; a group unit's
   score is the sum of the Taylor scores of all convolution output channels it removes (raw, no per-layer
   normalisation, so one ranking spans the whole network).
4. The lowest --ratio of all group units network-wide is removed (at least --min-channels kept per group).
   Not pruned: Detect output convolutions (box/class channels) and the fixed DFL projection.
No fine-tuning is done here: the reported mAP is the pruned model straight after pruning.

Outputs (in --output):
  taylor_channels.csv   per original channel: layer, channel, taylor1, global rank, pruned
  groups.csv            per Torch-Pruning group: root layer, members, channels before/kept/pruned
  layers.csv            per convolution: out channels before/after
  <stem>_pruned.pt      pruned model, Ultralytics checkpoint layout (needs PTQ/c2f_v2.py importable)
  manifest.json         settings, params/GFLOPs, val mAP before/after, check against the 2026-10-14 scores
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch_pruning as tp
import ultralytics
import yaml
from torch import nn
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.modules import Conv
from ultralytics.utils.loss import v8DetectionLoss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2f_v2 import replace_c2f  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KITTI = {0: 'car', 1: 'van', 2: 'truck', 3: 'pedestrian', 4: 'person_sitting', 5: 'cyclist', 6: 'tram', 7: 'misc'}
REFERENCE = ROOT/'sensitivity_results/kitti_channel_scores_20261014/channels.csv'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_unfused(path, device):
    """FP32 DetectionModel with BN kept (BN is pruned together with its convolution)."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = payload.get('ema') or payload['model'] if isinstance(payload, dict) else payload
    model = model.float().eval().to(device)
    hyp = get_cfg()
    if isinstance(model.args, dict):
        for key, value in model.args.items():
            if hasattr(hyp, key):
                setattr(hyp, key, value)
    model.args = hyp
    for name, p in model.named_parameters():
        p.requires_grad_('.dfl.' not in name)
    return model


def taylor_scores(model, data_dir, device, workers, limit=None):
    """{Conv module name: tensor[out_channels]} of mean |sum_hw A dL/dA| over the KITTI training images."""
    data = {'path': str(data_dir), 'train': 'images/train', 'val': 'images/val', 'names': KITTI, 'nc': len(KITTI)}
    dataset = YOLODataset(img_path=str(data_dir/'images/train'), data=data, imgsz=640, augment=False, rect=False,
                          cache=False, stride=32, batch_size=1, hyp=get_cfg(), prefix='Taylor: ')
    indices = list(range(len(dataset)))[:limit]
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=1, shuffle=False,
                                         num_workers=workers, collate_fn=YOLODataset.collate_fn, pin_memory=True)
    convs = {name: m for name, m in model.named_modules() if isinstance(m, Conv) and hasattr(m, 'bn')}
    acts = {}

    def keep(name):
        def hook(module, inputs, output):
            output.retain_grad()
            acts[name] = output
        return hook
    handles = [m.bn.register_forward_hook(keep(name)) for name, m in convs.items()]
    criterion = v8DetectionLoss(model)
    model.eval()
    model.model[-1].training = True      # raw head outputs for the loss; BN stays in eval mode
    sums = {name: torch.zeros(m.bn.num_features, dtype=torch.float64, device=device) for name, m in convs.items()}
    losses, begin = [], time.perf_counter()
    for k, raw in enumerate(loader):
        batch = {'img': raw['img'].to(device, non_blocking=True).float() / 255, 'cls': raw['cls'].to(device),
                 'bboxes': raw['bboxes'].to(device), 'batch_idx': raw['batch_idx'].to(device)}
        acts.clear()
        model.zero_grad(set_to_none=True)
        loss = criterion(model(batch['img']), batch)[0].sum()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at image {k}')
        loss.backward()
        for name, a in acts.items():
            if a.grad is None:
                raise RuntimeError(f'No gradient reached {name}')
            sums[name] += (a.detach().double() * a.grad.double()).sum((2, 3)).abs()[0]
        losses.append(float(loss))
        if (k + 1) % 500 == 0 or k + 1 == len(indices):
            print(f'  Taylor {k + 1}/{len(indices)} images, {(k + 1)/(time.perf_counter() - begin):.1f} img/s',
                  flush=True)
    for h in handles:
        h.remove()
    model.model[-1].training = False
    return {name: (s / len(indices)).float().cpu() for name, s in sums.items()}, \
        {'images': len(indices), 'loss_mean': float(np.mean(losses)), 'seconds': time.perf_counter() - begin}


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def compare_reference(scores):
    """Rank correlation with the 2026-10-14 Taylor-1 table (same checkpoint, fused convolutions)."""
    if not REFERENCE.is_file():
        return None
    ref = {}
    with REFERENCE.open(encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if row['taylor1'] not in ('', 'None'):
                ref[(row['layer'], int(row['channel']))] = float(row['taylor1'])
    mine, theirs = [], []
    for name, s in scores.items():
        for c, v in enumerate(s.tolist()):
            if (f'{name}.conv', c) in ref:
                mine.append(v)
                theirs.append(ref[(f'{name}.conv', c)])
    mine, theirs = np.array(mine), np.array(theirs)
    return {'reference': str(REFERENCE), 'channels_matched': len(mine), 'spearman_global': spearman(mine, theirs),
            'max_relative_difference': float(np.max(np.abs(mine - theirs) / np.maximum(np.abs(theirs), 1e-12)))}


def validate(model, data_yaml, batch):
    m = copy.deepcopy(model).eval()       # the validator fuses BN in place; keep the pruned model unfused
    stats = DetectionValidator(args=dict(data=str(data_yaml), imgsz=640, batch=batch, half=False, plots=False,
                                         save_json=False, verbose=False, workers=4))(model=m)
    return {'mAP50-95': stats['metrics/mAP50-95(B)'], 'mAP50': stats['metrics/mAP50(B)'],
            'precision': stats['metrics/precision(B)'], 'recall': stats['metrics/recall(B)']}


def conv_widths(model):
    return {name: m.out_channels for name, m in model.named_modules() if isinstance(m, nn.Conv2d)}


def to_c2f_v2(model, scores, example):
    """C2f -> C2f_v2 in place, carrying the cv1 scores over to the two halves.
    Returns ({nn.Conv2d: (name, original name, channel offset, scores)}, max output difference)."""
    with torch.no_grad():
        before = model(example)[0]
    split = replace_c2f(model)
    with torch.no_grad():
        diff = float((model(example)[0] - before).abs().max() / before.abs().max())   # relative (TF32 on GPU)
    if diff > 1e-3:
        raise RuntimeError(f'C2f_v2 differs from C2f by {diff} (relative)')
    score_of = {}
    for name, m in model.named_modules():
        if isinstance(m, Conv) and hasattr(m, 'bn'):
            original, rows = split.get(name, (name, slice(None)))
            score_of[m.conv] = (name, original, (rows.start or 0), scores[original][rows].clone())
    return score_of, diff


def dependency_groups(model, example):
    """Torch-Pruning groups; Detect output convolutions and the DFL projection are never pruned."""
    detect = model.model[-1]
    ignored = [seq[-1] for seq in (*detect.cv2, *detect.cv3)] + [detect.dfl.conv]
    model.eval()
    graph = tp.DependencyGraph().build_dependency(model, example_inputs=example)
    return graph, list(graph.get_all_groups(ignored_layers=ignored, root_module_types=[nn.Conv2d])), ignored


def group_scores(graph, groups, score_of, names_of):
    """Per group: sum of member convolution out-channel Taylor scores, mapped onto the group root's channels."""
    rows = []
    for g, group in enumerate(groups):
        root = group[0].dep.target.module
        imp = torch.zeros(graph.get_out_channels(root), dtype=torch.float64)
        members = []
        for item in group:
            layer = item.dep.target.module
            if graph.is_out_channel_pruning_fn(item.dep.handler) and layer in score_of:
                imp.index_add_(0, torch.tensor(item.root_idxs), score_of[layer][3][item.idxs].double())
                members.append(names_of[layer])
        if not members:
            raise RuntimeError(f'Group rooted at {names_of[root]} has no scored convolution')
        rows.append({'group': g, 'root': names_of[root], 'members': ' '.join(members), 'channels': len(imp),
                     'imp': imp})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/baseline/8n_float32.pt')
    parser.add_argument('--data', type=Path, default=ROOT/'PTQ/datasets/kitti')
    parser.add_argument('--output', type=Path, default=ROOT/'models/pruning/taylor_global_25pct')
    parser.add_argument('--ratio', type=float, default=0.25, help='fraction of all group channels to remove')
    parser.add_argument('--min-channels', type=int, default=1, help='channels kept at least in every group')
    parser.add_argument('--images', type=int, default=None, help='limit Taylor images (default: all train)')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch', type=int, default=32, help='validation batch size')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')
    data_yaml = Path(tempfile.gettempdir())/'kitti_prune.yaml'
    data_yaml.write_text(yaml.safe_dump({'path': str(args.data), 'train': 'images/train', 'val': 'images/val',
                                         'names': KITTI, 'nc': len(KITTI)}), encoding='utf-8')

    model = load_unfused(args.weights, device)
    example = torch.zeros(1, 3, 640, 640, device=device)
    print(f'Taylor-1 scores on KITTI train ({args.images or "all"} images)...', flush=True)
    scores, taylor_info = taylor_scores(model, args.data, device, args.workers, args.images)
    reference = compare_reference(scores)
    print(f'Check against 2026-10-14 scores: {reference}', flush=True)
    base_ops, base_params = tp.utils.count_ops_and_params(model, example)
    base_val = validate(model, data_yaml, args.batch)
    print(f'Unpruned: {base_params/1e6:.3f} M params, {2*base_ops/1e9:.2f} GFLOPs, val {base_val}', flush=True)

    score_of, diff = to_c2f_v2(model, scores, example)
    widths_before = conv_widths(model)
    names_of = {m: n for n, m in model.named_modules()}
    graph, groups, ignored = dependency_groups(model, example)
    group_rows = group_scores(graph, groups, score_of, names_of)
    units = [(float(v), r['group'], c) for r in group_rows for c, v in enumerate(r['imp'].tolist())]
    total = len(units)
    target = int(round(total * args.ratio))
    chosen, n_chosen = {g: [] for g in range(len(groups))}, 0
    for score, g, c in sorted(units):                       # ascending: lowest Taylor first
        if n_chosen == target:
            break
        if group_rows[g]['channels'] - len(chosen[g]) > args.min_channels:
            chosen[g].append(c)
            n_chosen += 1
    threshold = max(group_rows[g]['imp'][c].item() for g, cs in chosen.items() for c in cs)
    print(f'{len(groups)} groups, {total} channels; pruning {target} ({args.ratio:.0%}), '
          f'global Taylor threshold {threshold:.4g}', flush=True)

    # Prune each group (fresh pruning group from the root, so earlier prunes cannot shift its indices).
    for g, group in enumerate(groups):
        if chosen[g]:
            root = group[0].dep.target.module
            pruning_group = graph.get_pruning_group(root, group[0].dep.handler, sorted(chosen[g]))
            if not graph.check_pruning_group(pruning_group):
                raise RuntimeError(f'Torch-Pruning rejected group {group_rows[g]["root"]}')
            pruning_group.prune()
    with torch.no_grad():
        out_shape = tuple(model(example)[0].shape)
    ops, params = tp.utils.count_ops_and_params(model, example)
    pruned_val = validate(model, data_yaml, args.batch)
    print(f'Pruned: {params/1e6:.3f} M params ({params/base_params:.1%}), {2*ops/1e9:.2f} GFLOPs '
          f'({ops/base_ops:.1%}), output {out_shape}, val {pruned_val}', flush=True)

    # Reports.
    pruned_channels = {}
    for g, cs in chosen.items():
        group = groups[g]
        for item in group:
            layer = item.dep.target.module
            if graph.is_out_channel_pruning_fn(item.dep.handler) and layer in score_of:
                root_to_local = dict(zip(item.root_idxs, item.idxs))
                pruned_channels.setdefault(score_of[layer][0], set()).update(root_to_local[c] for c in cs
                                                                             if c in root_to_local)
    rows = []
    for layer, (name, original, offset, s) in score_of.items():
        for c, v in enumerate(s.tolist()):
            rows.append({'layer': name, 'original_layer': original, 'channel': c, 'original_channel': offset + c,
                         'taylor1': v, 'pruned': c in pruned_channels.get(name, set())})
    for rank, row in enumerate(sorted(rows, key=lambda r: -r['taylor1']), 1):
        row['rank_global_taylor1'] = rank
    write_csv(args.output/'taylor_channels.csv', rows)
    write_csv(args.output/'groups.csv', [{'group': r['group'], 'root': r['root'], 'members': r['members'],
                                          'channels': r['channels'], 'pruned': len(chosen[r['group']]),
                                          'kept': r['channels'] - len(chosen[r['group']]),
                                          'score_sum': float(r['imp'].sum()),
                                          'pruned_score_sum': float(r['imp'][chosen[r['group']]].sum())}
                                         for r in group_rows])
    widths_after = conv_widths(model)
    write_csv(args.output/'layers.csv', [{'layer': n, 'out_before': widths_before[n], 'out_after': widths_after[n],
                                          'pruned': widths_before[n] - widths_after[n]} for n in widths_before])

    stem = args.weights.stem
    model_path = args.output/f'{stem}_taylor{int(args.ratio * 100)}_pruned.pt'
    model.zero_grad(set_to_none=True)
    torch.save({'model': copy.deepcopy(model).cpu(), 'ema': None, 'date': datetime.now().isoformat(),
                'version': ultralytics.__version__, 'train_args': {}}, model_path)
    manifest = {'source': str(args.weights), 'source_sha256': sha256(args.weights), 'torch': torch.__version__,
                'torch_pruning': tp.__version__, 'ultralytics': ultralytics.__version__,
                'taylor1': 'mean_images(abs(sum_hw(A * dL/dA))), A = pre-activation Conv+BN output, BN eval',
                'taylor_images': taylor_info, 'reference_check': reference,
                'group_importance': 'sum of member conv out-channel Taylor-1, no normalisation (global ranking)',
                'ratio': args.ratio, 'groups': len(groups), 'group_channels_total': total,
                'group_channels_pruned': target, 'min_channels_per_group': args.min_channels,
                'global_threshold': threshold, 'c2f_split_max_abs_diff': diff,
                'ignored_layers': [names_of[m] for m in ignored], 'fine_tuned': False,
                'unpruned': {'params': base_params, 'gflops': 2 * base_ops / 1e9, 'val': base_val},
                'pruned': {'params': params, 'gflops': 2 * ops / 1e9, 'val': pruned_val, 'output_shape': out_shape,
                           'file': model_path.name, 'conv_out_channels': sum(widths_after.values()),
                           'conv_out_channels_before': sum(widths_before.values())}}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Done: {args.output}')


if __name__ == '__main__':
    main()
