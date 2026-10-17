"""Global L1-norm structured channel pruning of YOLOv8n (8n_float32.pt), ONNX export, trtexec build.

The L1 norm of an output channel is sum(|W[o, :, :, :]|) of the unfused Conv2d weight. All
prunable output channels of the network are ranked together, and the globally lowest-L1 channels
are removed until --ratio of the total has gone. There is no per-layer quota; a layer only stops
losing channels when it reaches --min-channels (otherwise whole layers would be removed: raw L1
grows with fan-in, so low-fan-in layers dominate the bottom of the ranking).

Structural constraints:
  * Residual C2f Bottlenecks (x + cv2(cv1(x))) tie the second half of C2f.cv1 to every
    Bottleneck.cv2 in that block. A tied position removes one channel in each member conv;
    its score is the mean L1 of those channels.
  * C2f.cv1 output is split by chunk(2), so both halves must lose the same number of channels:
    the k-th lowest position of each half is removed together (score = mean L1 of the pair).
Not pruned (fixed output size): Detect cv2[i][2] (4*reg_max), cv3[i][2] (nc) and the DFL conv.
Removed channels take their BatchNorm entries and the matching consumer input channels with them.
No fine-tuning is done.

Outputs (in --output):
  <stem>.pt               pruned model, Ultralytics checkpoint layout (FP32, BN kept)
  <stem>.onnx             BN-fused export, input images (1,3,640,640), output output0
  <stem>.engine           trtexec FP32 engine, --profilingVerbosity=detailed
  <stem>.layers.json      trtexec --exportLayerInfo
  <stem>.build.log        trtexec log
  <stem>_channel_l1.csv   L1 of every output channel, global removal order, kept/pruned
  <stem>_layers.csv       channels before/after per convolution
  <stem>_manifest.json    settings, parameter counts and output checks
"""
from __future__ import annotations

import argparse
import copy
import csv
import heapq
import json
import os
import subprocess
from pathlib import Path

import torch
from torch import nn
from ultralytics.nn.modules import C2f, SPPF, Concat, Conv, Detect

ROOT = Path(__file__).resolve().parents[1]
TRT = Path(r'C:\Users\dxu1\AppData\Local\NVIDIA\TensorRT-11.3.0.99')


class Tracer:
    """Symbolic forward pass: every tensor is a list of channel labels, (conv_name, index).

    Records each Conv2d's input labels, the prunable output segments, the residual ties
    between segments and the C2f.cv1 half pairs that must stay equal.
    """

    def __init__(self, model):
        self.names = {m: n for n, m in model.named_modules()}
        self.inputs = {}         # conv2d name -> list of input labels
        self.segments = []       # (conv_name, start, length)
        self.ties = []           # (segment, segment): same indices removed
        self.pairs = []          # (segment, segment): same number removed

    def conv(self, module, x):
        conv2d = module.conv if isinstance(module, Conv) else module
        name = self.names[conv2d]
        if conv2d.groups != 1:
            raise RuntimeError(f'Grouped convolution not supported: {name}')
        if len(x) != conv2d.in_channels:
            raise RuntimeError(f'{name}: traced {len(x)} input channels, expected {conv2d.in_channels}')
        self.inputs[name] = x
        c = conv2d.out_channels
        if not isinstance(module, Conv):     # plain Conv2d = Detect output layer, fixed width
            return [('fixed', name, i) for i in range(c)]
        return [(name, i) for i in range(c)]

    def c2f(self, m, x):
        y = self.conv(m.cv1, x)
        name, c = self.names[m.cv1.conv], len(y) // 2
        head, tail = (name, 0, c), (name, c, c)
        self.segments += [head, tail]
        self.pairs.append((head, tail))
        ys = [y[:c], y[c:]]
        for b in m.m:
            h = self.conv(b.cv1, ys[-1])
            self.segments.append((self.names[b.cv1.conv], 0, len(h)))
            out = self.conv(b.cv2, h)
            seg = (self.names[b.cv2.conv], 0, len(out))
            self.segments.append(seg)
            if b.add:
                self.ties.append((tail, seg))
                out = ys[-1]                 # x + f(x) carries x's (tied) channels
            else:
                tail = seg
            ys.append(out)
        return self.plain(m.cv2, sum(ys, []))

    def plain(self, module, x):
        y = self.conv(module, x)
        self.segments.append((self.names[module.conv], 0, len(y)))
        return y

    def run(self, model, channels=3):
        y, x = [], [('fixed', 'images', i) for i in range(channels)]
        for m in model.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if type(m) is Conv:
                x = self.plain(m, x)
            elif type(m) is C2f:
                x = self.c2f(m, x)
            elif type(m) is SPPF:
                if getattr(m, 'add', False):
                    raise RuntimeError('SPPF shortcut not supported')
                h = self.plain(m.cv1, x)
                x = self.plain(m.cv2, h * (getattr(m, 'n', 3) + 1))
            elif type(m) is Concat:
                if m.d != 1:
                    raise RuntimeError('Concat on non-channel axis')
                x = sum(x, [])
            elif type(m) is nn.Upsample:
                pass
            elif type(m) is Detect:
                if getattr(m, 'one2one_cv2', None) is not None:
                    raise RuntimeError('End-to-end Detect head not supported')
                for i, xi in enumerate(x):
                    for branch in (m.cv2[i], m.cv3[i]):
                        h = xi
                        for layer in branch:
                            h = self.plain(layer, h) if isinstance(layer, Conv) else self.conv(layer, h)
                x = None
            else:
                raise RuntimeError(f'Unsupported module {type(m).__name__} at layer {m.i}')
            y.append(x)


def union_groups(segments, ties):
    parent = {s: s for s in segments}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    for a, b in ties:
        if a[2] != b[2]:
            raise RuntimeError(f'Tied segments differ in length: {a} {b}')
        parent[find(a)] = find(b)
    groups = {}
    for s in segments:
        groups.setdefault(find(s), []).append(s)
    return {root: members for root, members in groups.items()}


def plan_global(model, ratio, min_channels):
    """Returns the tracer, the set of removed (conv, channel) labels and per-channel rows."""
    tracer = Tracer(model)
    tracer.run(model)
    modules = dict(model.named_modules())
    l1 = {name: modules[name].weight.detach().abs().sum(dim=(1, 2, 3)).double() for name in tracer.inputs}

    groups = union_groups(tracer.segments, tracer.ties)
    root_of = {s: root for root, members in groups.items() for s in members}

    def positions(root):
        """Positions of a group, ascending by mean L1: [(score, [(conv, ch), ...])]."""
        members, length = groups[root], groups[root][0][2]
        items = []
        for pos in range(length):
            channels = [(name, start + pos) for name, start, _ in members]
            items.append((sum(float(l1[n][c]) for n, c in channels) / len(channels), pos, channels))
        return sorted(items)

    # A block is one group, or two groups (C2f.cv1 halves) whose k-th lowest positions go together.
    paired = {}
    for a, b in tracer.pairs:
        ra, rb = root_of[a], root_of[b]
        if ra == rb or ra in paired or rb in paired:
            raise RuntimeError(f'Unexpected C2f pairing {a} {b}')
        paired[ra], paired[rb] = rb, ra
    blocks, seen = [], set()
    for root in groups:
        if root in seen:
            continue
        roots = [root] + ([paired[root]] if root in paired else [])
        seen.update(roots)
        lists = [positions(r) for r in roots]
        length = len(lists[0])
        if any(len(lst) != length for lst in lists):
            raise RuntimeError('Paired C2f halves differ in length')
        units = []
        for k in range(length):
            channels = sum((lst[k][2] for lst in lists), [])
            score = sum(float(l1[n][c]) for n, c in channels) / len(channels)
            units.append((score, channels))
        blocks.append({'roots': roots, 'units': units, 'removable': max(0, length - min_channels)})

    total = sum(length for _, _, length in tracer.segments)
    target = int(round(ratio * total))
    heap = [(b['units'][0][0], i, 0) for i, b in enumerate(blocks) if b['removable'] > 0]
    heapq.heapify(heap)
    removed, order, count = set(), {}, 0
    while count < target and heap:
        score, i, k = heapq.heappop(heap)
        for label in blocks[i]['units'][k][1]:
            removed.add(label)
            order[label] = len(order)
        count += len(blocks[i]['units'][k][1])
        if k + 1 < blocks[i]['removable']:
            heapq.heappush(heap, (blocks[i]['units'][k + 1][0], i, k + 1))
    if count < target:
        raise RuntimeError(f'Only {count}/{target} channels removable with min_channels={min_channels}')
    threshold = max((u[0] for b in blocks for u in b['units'] if u[1][0] in removed), default=None)

    rows = []
    for name, start, length in tracer.segments:
        for ch in range(start, start + length):
            rows.append({'conv': name, 'channel': ch, 'l1': float(l1[name][ch]),
                         'tied_group': ';'.join(f'{n}[{s}:{s + q}]' for n, s, q in groups[root_of[(name, start, length)]]),
                         'removal_order': order.get((name, ch), ''),
                         'status': 'pruned' if (name, ch) in removed else 'kept'})
    stats = {'prunable_channels': total, 'target_removed': target, 'removed': count,
             'last_removed_unit_score': threshold}
    return tracer, removed, rows, stats


def apply_pruning(model, tracer, removed):
    modules = dict(model.named_modules())
    rows = []
    for name, in_labels in tracer.inputs.items():
        conv2d = modules[name]
        keep_in = torch.tensor([i for i, label in enumerate(in_labels) if label not in removed], dtype=torch.long)
        keep_out = torch.tensor([i for i in range(conv2d.out_channels) if (name, i) not in removed], dtype=torch.long)
        rows.append({'conv': name, 'in_before': conv2d.in_channels, 'in_after': len(keep_in),
                     'out_before': conv2d.out_channels, 'out_after': len(keep_out),
                     'out_pruned': conv2d.out_channels - len(keep_out),
                     'out_pruned_pct': round(100 * (1 - len(keep_out) / conv2d.out_channels), 2)})
        conv2d.weight = nn.Parameter(conv2d.weight.data[keep_out][:, keep_in].clone())
        if conv2d.bias is not None:
            conv2d.bias = nn.Parameter(conv2d.bias.data[keep_out].clone())
        conv2d.in_channels, conv2d.out_channels = len(keep_in), len(keep_out)
        parent = modules.get(name.rsplit('.', 1)[0])
        if isinstance(parent, Conv) and parent.conv is conv2d:
            bn = parent.bn
            bn.weight = nn.Parameter(bn.weight.data[keep_out].clone())
            bn.bias = nn.Parameter(bn.bias.data[keep_out].clone())
            bn.running_mean = bn.running_mean[keep_out].clone()
            bn.running_var = bn.running_var[keep_out].clone()
            bn.num_features = len(keep_out)
    for m in model.modules():
        if type(m) is C2f:
            if m.cv1.conv.out_channels % 2:
                raise RuntimeError('C2f.cv1 halves became unequal')
            m.c = m.cv1.conv.out_channels // 2
    return rows


class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        for m in model.modules():
            if isinstance(m, Detect):
                m.export, m.format, m.dynamic, m.shape = True, 'onnx', False, None

    def forward(self, x):
        y = self.model(x)
        return y[0] if isinstance(y, (list, tuple)) else y


def export_onnx(model, path, imgsz):
    wrapper = ExportWrapper(copy.deepcopy(model).float().eval().fuse(verbose=False)).eval()
    with torch.no_grad():
        torch.onnx.export(wrapper, torch.zeros(1, 3, imgsz, imgsz), str(path), opset_version=17, dynamo=False,
                          input_names=['images'], output_names=['output0'], do_constant_folding=True)
    return wrapper


def build_engine(onnx_path, engine_path, layers_path, log_path):
    env = dict(os.environ, PATH=f"{TRT / 'bin'};{TRT / 'lib'};{os.environ['PATH']}")
    command = [str(TRT / 'bin/trtexec.exe'), f'--onnx={onnx_path}', f'--saveEngine={engine_path}',
               '--profilingVerbosity=detailed', '--dumpLayerInfo', f'--exportLayerInfo={layers_path}']
    with open(log_path, 'w', encoding='utf-8') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    if code != 0 or not engine_path.is_file():
        raise RuntimeError(f'trtexec failed ({code}); see {log_path}')
    return command


def engine_vs_torch(engine_path, wrapper, imgsz, seed):
    """Runs the engine and the BN-fused PyTorch export model on the same random input."""
    import tensorrt as trt
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inp = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    out = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    device = torch.device('cuda:0')
    x = torch.rand(1, 3, imgsz, imgsz, generator=torch.Generator().manual_seed(seed)).to(device)
    result = torch.empty(tuple(engine.get_tensor_shape(out)), device=device)
    context.set_tensor_address(inp, x.data_ptr())
    context.set_tensor_address(out, result.data_ptr())
    stream = torch.cuda.Stream()
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError('TensorRT inference failed')
    stream.synchronize()
    with torch.no_grad():
        reference = wrapper.to(device)(x)
    diff = (reference - result).abs()
    return {'output_shape': list(result.shape), 'max_abs_error': float(diff.max()), 'mean_abs_error': float(diff.mean())}


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT / 'models/baseline/8n_float32.pt')
    parser.add_argument('--ratio', type=float, default=0.2, help='fraction of all prunable channels removed')
    parser.add_argument('--min-channels', type=int, default=8, help='channels every layer (segment) keeps at least')
    parser.add_argument('--output', type=Path, default=ROOT / 'models/pruning/l1_global20')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f'{args.weights.stem}_l1_global{round(args.ratio * 100)}'

    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
    model = (ckpt.get('ema') or ckpt['model']).float().eval()
    original = copy.deepcopy(model)
    params_before = sum(p.numel() for p in model.parameters())

    tracer, removed, l1_rows, stats = plan_global(model, args.ratio, args.min_channels)
    layer_rows = apply_pruning(model, tracer, removed)
    params_after = sum(p.numel() for p in model.parameters())

    x = torch.rand(1, 3, args.imgsz, args.imgsz, generator=torch.Generator().manual_seed(args.seed))
    with torch.no_grad():
        y_old, y_new = original(x)[0], model(x)[0]
    if y_old.shape != y_new.shape:
        raise RuntimeError(f'Output shape changed: {tuple(y_old.shape)} -> {tuple(y_new.shape)}')

    pt_path = args.output / f'{stem}.pt'
    torch.save({'date': ckpt.get('date'), 'version': ckpt.get('version'), 'license': ckpt.get('license'),
                'docs': ckpt.get('docs'), 'epoch': -1, 'best_fitness': None, 'model': model, 'ema': None,
                'updates': None, 'optimizer': None, 'train_args': ckpt.get('train_args'),
                'pruning': {'method': 'global L1 output-channel norm', 'ratio': args.ratio,
                            'min_channels': args.min_channels, 'source': str(args.weights), 'fine_tuned': False}},
               pt_path)
    reloaded = torch.load(pt_path, map_location='cpu', weights_only=False)['model'].eval()
    with torch.no_grad():
        if not torch.equal(reloaded(x)[0], y_new):
            raise RuntimeError('Reloaded .pt differs from the pruned model')

    onnx_path = args.output / f'{stem}.onnx'
    wrapper = export_onnx(model, onnx_path, args.imgsz)
    import onnx
    onnx.checker.check_model(str(onnx_path))
    print(f'Saved {pt_path.name} and {onnx_path.name}', flush=True)

    engine_path = args.output / f'{stem}.engine'
    print('Building TensorRT engine (profilingVerbosity=detailed)...', flush=True)
    command = build_engine(onnx_path, engine_path, args.output / f'{stem}.layers.json', args.output / f'{stem}.build.log')
    check = engine_vs_torch(engine_path, wrapper, args.imgsz, args.seed)

    write_csv(args.output / f'{stem}_channel_l1.csv', l1_rows)
    write_csv(args.output / f'{stem}_layers.csv', layer_rows)
    manifest = {
        'source': str(args.weights), 'ratio': args.ratio, 'min_channels': args.min_channels,
        'importance': 'sum |W| over (in, kh, kw) of the unfused Conv2d weight, ranked globally; '
                      'tied/paired positions scored by mean L1', 'fine_tuned': False, **stats,
        'params_before': params_before, 'params_after': params_after,
        'param_reduction': 1 - params_after / params_before,
        'layers_at_min_channels': [r['conv'] for r in layer_rows if r['out_pruned'] and r['out_after'] <= args.min_channels],
        'layers_untouched': [r['conv'] for r in layer_rows if not r['out_pruned']],
        'torch_output_mean_abs_change_vs_original': float((y_old - y_new).abs().mean()),
        'onnx': {'file': onnx_path.name, 'opset': 17, 'bn_fused': True, 'input': [1, 3, args.imgsz, args.imgsz]},
        'trtexec': command, 'engine': {'file': engine_path.name, 'size_mb': engine_path.stat().st_size / 1e6},
        'engine_vs_fused_torch_random_input': check,
    }
    (args.output / f'{stem}_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('prunable_channels', 'target_removed', 'removed', 'params_before',
                                               'params_after', 'param_reduction', 'layers_at_min_channels',
                                               'engine', 'engine_vs_fused_torch_random_input')}, indent=2))


if __name__ == '__main__':
    main()
