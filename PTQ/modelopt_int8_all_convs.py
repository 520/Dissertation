"""INT8 PTQ of YOLOv8n-KITTI convolutions with ModelOpt, Q/DQ ONNX export, trtexec build.

Every Conv2d (backbone, neck, detection outputs, DFL projection) gets an INT8 per-channel
weight quantizer and an INT8 per-tensor input quantizer, except the convolutions named with
--exclude, which keep no quantizer and stay FP32. No other module is quantized.
Calibration: max over KITTI training images (no augmentation).

Outputs (in --output):
  <stem>.onnx              explicit Q/DQ ONNX
  <stem>.engine            TensorRT engine, profilingVerbosity=detailed
  <stem>.layers.json       trtexec --exportLayerInfo
  <stem>.build.log         trtexec log
  <stem>.precision.csv     per-convolution weight/input/output datatypes in the engine
  manifest.json            settings, checks and per-layer quantizer amax
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

import modelopt
import modelopt.torch.quantization as mtq
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.modules.head import Detect

ROOT = Path(__file__).resolve().parents[1]
TRT = Path(r'C:\Users\dxu1\AppData\Local\NVIDIA\TensorRT-11.3.0.99')
KITTI = {0: 'car', 1: 'van', 2: 'truck', 3: 'pedestrian', 4: 'person_sitting', 5: 'cyclist', 6: 'tram', 7: 'misc'}

# Only the explicitly listed quantizers are enabled; everything else stays off.
INT8_CONV_CFG = {
    'quant_cfg': {
        '*': {'enable': False},
        '*weight_quantizer': {'num_bits': 8, 'axis': 0},      # per output channel
        '*input_quantizer': {'num_bits': 8, 'axis': None},    # per tensor
    },
    'algorithm': 'max',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_fused(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = payload.get('ema') or payload['model'] if isinstance(payload, dict) else payload
    model = model.float().eval()
    model.fuse(verbose=False)          # fold BN into the convolutions, as deployed
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class ExportWrapper(nn.Module):
    """Returns the single decoded output tensor (1, 4+nc, 8400), like Ultralytics ONNX export."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        for m in model.modules():
            if isinstance(m, Detect):
                m.export, m.format, m.dynamic = True, 'onnx', False

    def forward(self, x):
        y = self.model(x)
        return y[0] if isinstance(y, (list, tuple)) else y


def calibration_images(data_dir, count, seed):
    data = {'path': str(data_dir), 'train': 'images/train', 'val': 'images/val', 'names': KITTI, 'nc': 8}
    dataset = YOLODataset(img_path=str(data_dir/'images/train'), data=data, imgsz=640, augment=False,
                          rect=False, cache=False, stride=32, batch_size=1, hyp=get_cfg(), prefix='Calibration: ')
    order = np.random.default_rng(seed).permutation(len(dataset))[:count]
    return dataset, [int(i) for i in order]


def quantize(model, dataset, indices, device, exclude):
    def forward_loop(m):
        with torch.no_grad():
            for k, i in enumerate(indices):
                m(dataset[i]['img'][None].to(device).float() / 255)
                if (k + 1) % 100 == 0:
                    print(f'  calibrated {k + 1}/{len(indices)}', flush=True)
    model = mtq.quantize(model, INT8_CONV_CFG, forward_loop)
    # Keep quantizers on Conv2d only (ModelOpt also wraps pooling layers).
    convs, others = [], []
    for name, module in model.named_modules():
        if not hasattr(module, 'input_quantizer'):
            continue
        if isinstance(module, nn.Conv2d):
            convs.append((name, module))
        else:
            module.input_quantizer.disable()
            if hasattr(module, 'output_quantizer'):
                module.output_quantizer.disable()
            others.append(name)
    unknown = set(exclude) - {name for name, _ in convs}
    if unknown:
        raise ValueError(f'--exclude names not found among convolutions: {sorted(unknown)}')
    for name, module in convs:
        if name in exclude:
            module.weight_quantizer.disable()
            module.input_quantizer.disable()
        elif not (module.weight_quantizer.is_enabled and module.input_quantizer.is_enabled):
            raise RuntimeError(f'Quantizer disabled on {name}')
    return model, [(n, m) for n, m in convs if n not in exclude], others


def check_onnx(path, expected, exclude):
    graph = onnx.load(path).graph
    producer = {out: node for node in graph.node for out in node.output}
    rows = []
    for node in graph.node:
        if node.op_type != 'Conv':
            continue
        x, w = node.input[0], node.input[1]
        x_dq = producer.get(x) is not None and producer[x].op_type == 'DequantizeLinear'
        w_dq = producer.get(w) is not None and producer[w].op_type == 'DequantizeLinear'
        # module 'model.22.dfl.conv' -> ONNX node '.../model.22/dfl/conv/Conv'
        excluded = any(node.name.endswith('/' + '.'.join(name.split('.')[:2]) + '/'
                                          + '/'.join(name.split('.')[2:]) + '/Conv') for name in exclude)
        rows.append({'node': node.name, 'input_qdq': x_dq, 'weight_qdq': w_dq, 'excluded': excluded})
    ops = Counter(n.op_type for n in graph.node)
    missing = [r['node'] for r in rows if not r['excluded'] and not (r['input_qdq'] and r['weight_qdq'])]
    leaked = [r['node'] for r in rows if r['excluded'] and (r['input_qdq'] or r['weight_qdq'])]
    if len(rows) != expected or missing or leaked or sum(r['excluded'] for r in rows) != len(exclude):
        raise RuntimeError(f'ONNX check failed: {len(rows)} convs, not Q/DQ: {missing}, '
                           f'excluded but Q/DQ: {leaked}')
    return {'convs': len(rows), 'convs_with_input_and_weight_qdq': sum(r['input_qdq'] and r['weight_qdq'] for r in rows),
            'convs_without_qdq': [r['node'] for r in rows if r['excluded']],
            'QuantizeLinear': ops['QuantizeLinear'], 'DequantizeLinear': ops['DequantizeLinear']}


def build_engine(onnx_path, engine_path, layers_path, log_path):
    env = dict(os.environ, PATH=f"{TRT/'bin'};{TRT/'lib'};{os.environ['PATH']}")
    command = [str(TRT/'bin/trtexec.exe'), f'--onnx={onnx_path}', f'--saveEngine={engine_path}',
               '--profilingVerbosity=detailed', '--dumpLayerInfo', f'--exportLayerInfo={layers_path}']
    with open(log_path, 'w', encoding='utf-8') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    if code != 0 or not engine_path.is_file():
        raise RuntimeError(f'trtexec failed ({code}); see {log_path}')
    return command


def engine_precision(layers_path, csv_path):
    layers = json.loads(Path(layers_path).read_text())['Layers']
    rows = []
    for layer in layers:
        if 'Weights' not in layer:
            continue
        names = re.findall(r'/(model\.[\w./]+?)/Conv', layer['Name'])
        rows.append({'conv': (names[-1] if names else layer['Name']).replace('/', '.'),
                     'weights': layer['Weights']['Type'], 'weight_count': layer['Weights']['Count'],
                     'input': layer['Inputs'][0]['Datatype'], 'output': layer['Outputs'][0]['Datatype'],
                     'tactic': layer.get('TacticName', '')})
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    size = {'Int8': 1, 'Half': 2, 'Float': 4}
    weight_bytes = sum(l[k]['Count']*size[l[k]['Type']] for l in layers for k in ('Weights', 'Bias')
                       if isinstance(l.get(k), dict) and l[k].get('Count'))
    return {'conv_layers': len(rows), 'weight_types': dict(Counter(r['weights'] for r in rows)),
            'weight/input/output': {'/'.join(k): v for k, v in Counter((r['weights'], r['input'], r['output']) for r in rows).items()},
            'engine_weight_bytes': weight_bytes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/baseline/8n_float32.pt')
    parser.add_argument('--data', type=Path, default=ROOT/'PTQ/datasets/kitti')
    parser.add_argument('--output', type=Path, default=ROOT/'models/quantization/int8/modelopt_all64')
    parser.add_argument('--calib', type=int, default=512, help='number of KITTI training images')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exclude', nargs='*', default=[],
                        help='conv module names to keep unquantized (FP32), e.g. model.22.dfl.conv')
    parser.add_argument('--tag', default=None, help='file-name tag; default int8_<n>conv')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')
    torch.manual_seed(args.seed)

    model = load_fused(args.weights).to(device)
    n_conv = sum(isinstance(m, nn.Conv2d) for m in model.modules())
    print(f'Fused model: {n_conv} Conv2d layers; excluded: {args.exclude}', flush=True)
    stem = f'{args.weights.stem}_{args.tag or f"int8_{n_conv - len(args.exclude)}conv"}'
    onnx_path, engine_path = args.output/f'{stem}.onnx', args.output/f'{stem}.engine'
    dataset, indices = calibration_images(args.data, args.calib, args.seed)
    print(f'Calibrating on {len(indices)} KITTI training images...', flush=True)
    begin = time.perf_counter()
    model, convs, others = quantize(model, dataset, indices, device, args.exclude)
    print(f'Quantized convs: {len(convs)}; quantizers disabled on {len(others)} non-conv modules', flush=True)
    if len(convs) != n_conv - len(args.exclude):
        raise RuntimeError(f'Only {len(convs)}/{n_conv - len(args.exclude)} convolutions were quantized')

    # Export on CPU: the TorchScript exporter crashes (access violation) with CUDA fake-quant here.
    wrapper = ExportWrapper(model.cpu()).eval()
    for m in wrapper.modules():
        if isinstance(m, Detect):
            m.shape = None      # rebuild cached anchors/strides on the export device
    dummy = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        out_shape = tuple(wrapper(dummy).shape)
        torch.onnx.export(wrapper, dummy, str(onnx_path), opset_version=17, dynamo=False,
                          input_names=['images'], output_names=['output0'], do_constant_folding=True)
    onnx_check = check_onnx(onnx_path, n_conv, args.exclude)
    print(f'ONNX {onnx_path.name}: output {out_shape}, {onnx_check}', flush=True)

    print('Building TensorRT engine (detailed)...', flush=True)
    command = build_engine(onnx_path, engine_path, args.output/f'{stem}.layers.json', args.output/f'{stem}.build.log')
    precision = engine_precision(args.output/f'{stem}.layers.json', args.output/f'{stem}.precision.csv')
    print(f'Engine {engine_path.name}: {engine_path.stat().st_size/1e6:.2f} MB, {precision}', flush=True)

    amax = {name: {'input_amax': float(m.input_quantizer.amax), 'weight_amax_min': float(m.weight_quantizer.amax.min()),
                   'weight_amax_max': float(m.weight_quantizer.amax.max())} for name, m in convs}
    manifest = {'source': str(args.weights), 'source_sha256': sha256(args.weights), 'modelopt': modelopt.__version__,
                'torch': torch.__version__, 'config': INT8_CONV_CFG, 'bn_fused_before_quantization': True,
                'quantized_convs': len(convs), 'unquantized_convs_fp32': args.exclude,
                'calibration': {'images': len(indices), 'split': 'KITTI train', 'seed': args.seed, 'imgsz': 640,
                                'augment': False, 'algorithm': 'max'},
                'calibration_seconds': time.perf_counter() - begin, 'non_conv_quantizers_disabled': others,
                'onnx': {'file': onnx_path.name, 'opset': 17, 'output_shape': out_shape, **onnx_check},
                'trtexec': command, 'engine': {'file': engine_path.name, 'size_mb': engine_path.stat().st_size/1e6,
                                               **precision},
                'quantizer_amax': amax}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Done: {args.output}')


if __name__ == '__main__':
    main()
