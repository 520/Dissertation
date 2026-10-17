"""L1-norm structured channel pruning of YOLOv8n (8n_float32.pt), then ONNX export and trtexec build.

For every convolution the L1 norm of each output channel is sum(|W[o, :, :, :]|) of the
(unfused) Conv2d weight. Every prunable layer then loses round(ratio * C_out) of its output
channels, the ones with the lowest L1. The matching BatchNorm entries and the input channels of
every consumer convolution are removed with them.

Coupled channels (they must be removed with the same indices):
  * C2f residual Bottlenecks: x + cv2(cv1(x)) ties the second half of C2f.cv1 to every
    Bottleneck.cv2 in that block; their L1 values are summed per channel and ranked together.
  * C2f.cv1 is split with chunk(2): each half is pruned separately by the same ratio, so the
    halves stay equal.
Not pruned (fixed output size): Detect cv2[i][2] (4*reg_max box bins), cv3[i][2] (nc classes)
and the DFL projection. No fine-tuning is done.

Outputs (in --output):
  <stem>.pt               pruned model, Ultralytics checkpoint layout (FP32, BN kept)
  <stem>.onnx             BN-fused export, input images (1,3,640,640), output output0
  <stem>.engine           trtexec FP32 engine, --profilingVerbosity=detailed
  <stem>.layers.json      trtexec --exportLayerInfo
  <stem>.build.log        trtexec log
  <stem>_channel_l1.csv   L1 of every output channel, rank in its group and kept/pruned
  <stem>_layers.csv       channels before/after per convolution
  <stem>_manifest.json    settings, parameter counts and output checks
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn
from ultralytics.nn.modules import C2f, SPPF, Concat, Conv, Detect

ROOT = Path(__file__).resolve().parents[1]
TRT = Path(r'C:\Users\dxu1\AppData\Local\NVIDIA\TensorRT-11.3.0.99')


class Tracer:
    """Symbolic forward pass: every tensor is a list of channel labels.

    A label is (conv_name, out_index) for a prunable channel or ('fixed', ...) otherwise.
    Records the input labels of every Conv2d and the coupling caused by residual adds.
    """

    def __init__(self, model):
        self.names = {m: n for n, m in model.named_modules()}
        self.inputs = {}         # conv2d name -> list of input labels
        self.segments = []       # [(conv_name, start, length)] prunable output ranges
        self.ties = []           # [(segment_a, segment_b)] aligned residual coupling

    def conv(self, module, x, segments=1):
        conv2d = module.conv if isinstance(module, Conv) else module
        name = self.names[conv2d]
        if conv2d.groups != 1:
            raise RuntimeError(f'Grouped convolution not supported: {name}')
        if len(x) != conv2d.in_channels:
            raise RuntimeError(f'{name}: traced {len(x)} input channels, expected {conv2d.in_channels}')
        self.inputs[name] = x
        c = conv2d.out_channels
        if not isinstance(module, Conv):     # plain Conv2d: detect output layers
            return [('fixed', name, i) for i in range(c)]
        step = c // segments
        for s in range(segments):
            self.segments.append((name, s * step, step))
        return [(name, i) for i in range(c)]

    def seq(self, module, x):
        for m in module:
            x = self.conv(m, x)
        return x

    def c2f(self, m, x):
        y = self.conv(m.cv1, x, segments=2)
        c = len(y) // 2
        ys = [y[:c], y[c:]]
        tail = (self.names[m.cv1.conv], c, c)
        for b in m.m:
            out = self.conv(b.cv2, self.conv(b.cv1, ys[-1]))
            if b.add:
                self.ties.append((tail, (self.names[b.cv2.conv], 0, len(out))))
                out = ys[-1]                 # x + f(x): same (tied) channels as x
            else:
                tail = (self.names[b.cv2.conv], 0, len(out))
            ys.append(out)
        return self.conv(m.cv2, sum(ys, []))

    def sppf(self, m, x):
        if getattr(m, 'add', False):
            raise RuntimeError('SPPF shortcut not supported')
        y = self.conv(m.cv1, x)
        return self.conv(m.cv2, y * (getattr(m, 'n', 3) + 1))

    def detect(self, m, xs):
        if getattr(m, 'one2one_cv2', None) is not None:
            raise RuntimeError('End-to-end Detect head not supported')
        for i, x in enumerate(xs):
            self.seq(m.cv2[i], x)
            self.seq(m.cv3[i], x)
        return None

    def run(self, model, channels=3):
        y, x = [], [('fixed', 'images', i) for i in range(channels)]
        for m in model.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if isinstance(m, Conv):
                x = self.conv(m, x)
            elif isinstance(m, C2f) and type(m) is C2f:
                x = self.c2f(m, x)
            elif isinstance(m, SPPF):
                x = self.sppf(m, x)
            elif isinstance(m, Concat):
                if m.d != 1:
                    raise RuntimeError('Concat on non-channel axis')
                x = sum(x, [])
            elif isinstance(m, nn.Upsample):
                pass
            elif isinstance(m, Detect) and type(m) is Detect:
                x = self.detect(m, x)
            else:
                raise RuntimeError(f'Unsupported module {type(m).__name__} at layer {m.i}')
            y.append(x)


def group_segments(segments, ties):
    """Union-find over segments; returns lists of aligned segments that share pruning indices."""
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
    return list(groups.values())


def channel_l1(conv2d):
    return conv2d.weight.detach().abs().sum(dim=(1, 2, 3)).double()


def plan_pruning(model, ratio):
    tracer = Tracer(model)
    tracer.run(model)
    convs = dict(model.named_modules())
    l1 = {name: channel_l1(convs[name]) for name in tracer.inputs}
    removed, rows = set(), []
    for gid, group in enumerate(sorted(group_segments(tracer.segments, tracer.ties))):
        length = group[0][2]
        score = sum(l1[name][start:start + length] for name, start, _ in group)
        k = int(round(ratio * length))
        order = torch.argsort(score, stable=True).tolist()     # ascending L1
        rank = {pos: r for r, pos in enumerate(order)}
        drop = set(order[:k])
        members = ';'.join(f'{n}[{s}:{s + length}]' for n, s, _ in group)
        for name, start, _ in group:
            for pos in range(length):
                ch = start + pos
                if pos in drop:
                    removed.add((name, ch))
                rows.append({'group': gid, 'group_members': members, 'conv': name, 'channel': ch,
                             'l1': float(l1[name][ch]), 'group_l1': float(score[pos]),
                             'rank_in_group_ascending': rank[pos], 'group_size': length,
                             'pruned_in_group': k, 'status': 'pruned' if pos in drop else 'kept'})
    return tracer, removed, rows


def keep_index(labels, removed):
    return torch.tensor([i for i, label in enumerate(labels) if label not in removed], dtype=torch.long)


def apply_pruning(model, tracer, removed):
    modules = dict(model.named_modules())
    layer_rows = []
    for name, in_labels in tracer.inputs.items():
        conv2d = modules[name]
        out_labels = [(name, i) for i in range(conv2d.out_channels)]
        keep_in, keep_out = keep_index(in_labels, removed), keep_index(out_labels, removed)
        layer_rows.append({'conv': name, 'in_before': conv2d.in_channels, 'in_after': len(keep_in),
                           'out_before': conv2d.out_channels, 'out_after': len(keep_out),
                           'out_pruned': conv2d.out_channels - len(keep_out)})
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
            m.c = m.cv1.conv.out_channels // 2
            if m.cv1.conv.out_channels % 2:
                raise RuntimeError('C2f.cv1 halves became unequal')
    return layer_rows


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
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        torch.onnx.export(wrapper, dummy, str(path), opset_version=17, dynamo=False,
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
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inp = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    out = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    device = torch.device('cuda:0')
    x = torch.rand(1, 3, imgsz, imgsz, generator=torch.Generator().manual_seed(seed)).to(device)
    result = torch.empty(tuple(engine.get_tensor_shape(out)), device=device)
    context.set_tensor_address(inp, x.data_ptr())
    context.set_tensor_address(out, result.data_ptr())
    if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
        raise RuntimeError('TensorRT inference failed')
    torch.cuda.synchronize()
    with torch.no_grad():
        reference = wrapper.to(device)(x)
    diff = (reference - result).abs()
    return {'output_shape': list(result.shape), 'max_abs_error': float(diff.max()),
            'mean_abs_error': float(diff.mean())}


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT / 'models/baseline/8n_float32.pt')
    parser.add_argument('--ratio', type=float, default=0.2, help='fraction of output channels removed per layer')
    parser.add_argument('--output', type=Path, default=ROOT / 'models/pruning/l1_20')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f'{args.weights.stem}_l1_prune{round(args.ratio * 100)}'

    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
    model = (ckpt.get('ema') or ckpt['model']).float().eval()
    original = copy.deepcopy(model)
    params_before = sum(p.numel() for p in model.parameters())

    tracer, removed, l1_rows = plan_pruning(model, args.ratio)
    layer_rows = apply_pruning(model, tracer, removed)
    params_after = sum(p.numel() for p in model.parameters())

    # The pruned graph must run end to end; outputs differ from the original (no fine-tuning).
    x = torch.rand(1, 3, args.imgsz, args.imgsz, generator=torch.Generator().manual_seed(args.seed))
    with torch.no_grad():
        y_old, y_new = original(x)[0], model(x)[0]
    if y_old.shape != y_new.shape:
        raise RuntimeError(f'Output shape changed: {tuple(y_old.shape)} -> {tuple(y_new.shape)}')

    pt_path = args.output / f'{stem}.pt'
    torch.save({'date': ckpt.get('date'), 'version': ckpt.get('version'), 'license': ckpt.get('license'),
                'docs': ckpt.get('docs'), 'epoch': -1, 'best_fitness': None, 'model': model, 'ema': None,
                'updates': None, 'optimizer': None, 'train_args': ckpt.get('train_args'),
                'pruning': {'method': 'L1 output-channel norm, per layer', 'ratio': args.ratio,
                            'source': str(args.weights), 'fine_tuned': False}}, pt_path)
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
    command = build_engine(onnx_path, engine_path, args.output / f'{stem}.layers.json',
                           args.output / f'{stem}.build.log')
    check = engine_vs_torch(engine_path, wrapper, args.imgsz, args.seed)

    write_csv(args.output / f'{stem}_channel_l1.csv', l1_rows)
    write_csv(args.output / f'{stem}_layers.csv', layer_rows)
    pruned_layers = [r for r in layer_rows if r['out_pruned']]
    manifest = {
        'source': str(args.weights), 'ratio': args.ratio, 'rounding': 'round(ratio * channels) per group',
        'importance': 'sum |W| over (in, kh, kw) of the unfused Conv2d weight; coupled groups summed',
        'fine_tuned': False,
        'params_before': params_before, 'params_after': params_after,
        'param_reduction': 1 - params_after / params_before,
        'convs_total': len(layer_rows), 'convs_output_pruned': len(pruned_layers),
        'convs_output_fixed': [r['conv'] for r in layer_rows if not r['out_pruned']],
        'channels_before': sum(r['out_before'] for r in pruned_layers),
        'channels_after': sum(r['out_after'] for r in pruned_layers),
        'residual_coupled_groups': [sorted({s[0] for s in g}) for g in group_segments(tracer.segments, tracer.ties) if len(g) > 1],
        'torch_output_mean_abs_change_vs_original': float((y_old - y_new).abs().mean()),
        'onnx': {'file': onnx_path.name, 'opset': 17, 'bn_fused': True, 'input': [1, 3, args.imgsz, args.imgsz]},
        'trtexec': command, 'engine': {'file': engine_path.name, 'size_mb': engine_path.stat().st_size / 1e6},
        'engine_vs_fused_torch_random_input': check,
    }
    (args.output / f'{stem}_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('params_before', 'params_after', 'param_reduction',
                                               'convs_output_pruned', 'channels_before', 'channels_after',
                                               'engine', 'engine_vs_fused_torch_random_input')}, indent=2))


if __name__ == '__main__':
    main()
