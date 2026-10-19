"""INT8 QAT of YOLOv8n-KITTI with ModelOpt on all 64 convolutions, Q/DQ ONNX export, trtexec build.

Same quantization as PTQ/modelopt_int8_all_convs.py (BN folded first, INT8 per-channel weights and
per-tensor inputs on every Conv2d including the DFL projection, max calibration on 512 KITTI training
images), followed by Ultralytics fine-tuning through the fake quantizers. Quantizer ranges stay fixed
during training; only the (fused) weights and biases are updated. DFL stays frozen, as in Ultralytics.

Outputs (in --output):
  <stem>.modelopt.pt       ModelOpt checkpoint of the best epoch (restore onto the fused model)
  <stem>.onnx              explicit Q/DQ ONNX
  <stem>.engine            TensorRT engine, profilingVerbosity=detailed
  <stem>.layers.json       trtexec --exportLayerInfo
  <stem>.build.log         trtexec log
  <stem>.precision.csv     per-convolution weight/input/output datatypes in the engine
  manifest.json            settings, training results, checks and per-layer quantizer amax

Example:
    .venv/Scripts/python.exe -m QAT.modelopt_qat_all64
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

import modelopt
import modelopt.torch.opt as mto
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.head import Detect

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import PTQ.modelopt_int8_all_convs as ptq
from PTQ.modelopt_int8_all_convs import (INT8_CONV_CFG, KITTI, ExportWrapper, build_engine, calibration_images,
                                         check_onnx, engine_precision, load_fused, quantize, sha256)

ROOT = Path(__file__).resolve().parents[1]


class ModelOptTrainer(DetectionTrainer):
    """Trains a pre-quantized model and also keeps a full-precision ModelOpt checkpoint of the best epoch."""

    def save_model(self):
        super().save_model()   # Ultralytics checkpoint (FP16 weights + quantizer state), for reference
        if self.best_fitness == self.fitness:
            mto.save(self.ema.ema, self.wdir/'best.modelopt.pt')

    def final_eval(self):
        return                 # per-epoch validation already ran on the EMA model


def myelin_precision(layers_path, csv_path):
    """Per-convolution datatypes for engines compiled by Myelin (e.g. sm120), where convolutions appear as
    'correlation' layers with 'Constants' instead of CaskConvolution layers with 'Weights'. One kernel may
    implement several ONNX convolutions; its ONNX layers are listed in the layer Metadata."""
    layers = json.loads(Path(layers_path).read_text())['Layers']
    if any('Weights' in layer for layer in layers):
        return engine_precision(layers_path, csv_path)
    size = {'Int8': 1, 'Half': 2, 'Float': 4}
    rows, weight_bytes = [], 0
    for layer in layers:
        if layer['LayerType'] != 'correlation':
            continue
        consts = layer.get('Constants', [])
        weight_bytes += sum(int(np.prod(c['Dimensions']))*size.get(c['Datatype'], 4) for c in consts)
        for name in re.findall(r'ONNX Layer: /model/([^\x1f\]]*)/Conv\]', layer.get('Metadata', '')):
            rows.append({'conv': name.replace('/', '.'), 'kernel': layer['Name'],
                         'weights': consts[0]['Datatype'] if consts else '', 'input': layer['Inputs'][0]['Datatype'],
                         'output': layer['Outputs'][0]['Datatype'], 'tactic': layer.get('TacticName', '')})
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {'conv_layers': len(rows), 'conv_kernels': len({r['kernel'] for r in rows}),
            'weight_types': dict(Counter(r['weights'] for r in rows)),
            'weight/input/output': {'/'.join(k): v for k, v in Counter((r['weights'], r['input'], r['output'])
                                                                       for r in rows).items()},
            'engine_constant_bytes': weight_bytes}


def export(model, onnx_path):
    wrapper = ExportWrapper(model.float().cpu()).eval()
    for m in wrapper.modules():
        if isinstance(m, Detect):
            m.shape = None
    dummy = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        out_shape = tuple(wrapper(dummy).shape)
        torch.onnx.export(wrapper, dummy, str(onnx_path), opset_version=17, dynamo=False,
                          input_names=['images'], output_names=['output0'], do_constant_folding=True)
    return out_shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/baseline/8n_float32.pt')
    parser.add_argument('--data', type=Path, default=ROOT/'PTQ/datasets/kitti')
    parser.add_argument('--output', type=Path, default=ROOT/'models/quantization/int8/qat/modelopt_all64')
    parser.add_argument('--calib', type=int, default=512, help='number of KITTI training images')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--cache', default='ram', help="Ultralytics image cache: 'ram', 'disk' or 'false'")
    parser.add_argument('--lr0', type=float, default=1e-4)
    parser.add_argument('--name', default='kitti_modelopt_qat_all64')
    parser.add_argument('--trt', type=Path, default=Path(r'C:\Users\dxu1\AppData\Local\NVIDIA\TensorRT-11.3-tools'),
                        help='TensorRT folder containing bin/trtexec.exe')
    args = parser.parse_args()
    if not (args.trt/'bin/trtexec.exe').is_file():
        parser.error(f'trtexec.exe not found below {args.trt}')
    ptq.TRT = args.trt
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')
    torch.manual_seed(args.seed)

    model = load_fused(args.weights).to(device)
    n_conv = sum(isinstance(m, nn.Conv2d) for m in model.modules())
    stem = f'{args.weights.stem}_qat_int8_all{n_conv}'
    dataset, indices = calibration_images(args.data, args.calib, args.seed)
    print(f'Fused model: {n_conv} Conv2d layers. Calibrating on {len(indices)} KITTI training images...', flush=True)
    model, convs, others = quantize(model, dataset, indices, device, [])
    if len(convs) != n_conv:
        raise RuntimeError(f'Only {len(convs)}/{n_conv} convolutions were quantized')
    calib_amax = {name: float(m.input_quantizer.amax) for name, m in convs}
    for p in model.parameters():
        p.requires_grad_(True)

    data_yaml = args.data/'kitti.yaml'
    data_yaml.write_text(json.dumps({'path': str(args.data.resolve()), 'train': 'images/train',
                                     'val': 'images/val', 'nc': len(KITTI), 'names': KITTI}), encoding='utf-8')
    trainer = ModelOptTrainer(overrides={
        'model': str(args.weights), 'data': str(data_yaml), 'epochs': args.epochs, 'imgsz': 640,
        'batch': args.batch, 'device': '0', 'workers': args.workers, 'seed': args.seed,
        'cache': False if args.cache == 'false' else args.cache,
        'project': str(ROOT/'QAT/runs'), 'name': args.name, 'exist_ok': True,
        'optimizer': 'AdamW', 'lr0': args.lr0, 'lrf': 0.1, 'weight_decay': 5e-4, 'warmup_epochs': 1.0,
        'cos_lr': True, 'close_mosaic': min(10, args.epochs), 'amp': False, 'plots': False,
        'val': True, 'save': True})
    trainer.model = model
    begin = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - begin

    best = Path(trainer.wdir)/'best.modelopt.pt'
    checkpoint = args.output/f'{stem}.modelopt.pt'
    shutil.copy2(best, checkpoint)
    model = mto.restore(load_fused(args.weights), str(checkpoint), map_location='cpu').float().eval()
    convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    if not all(m.input_quantizer.is_enabled and m.weight_quantizer.is_enabled for _, m in convs):
        raise RuntimeError('Restored model lost a conv quantizer')
    if any(abs(float(m.input_quantizer.amax) - calib_amax[n]) > 1e-6 for n, m in convs):
        raise RuntimeError('Input quantizer ranges changed during QAT')

    onnx_path, engine_path = args.output/f'{stem}.onnx', args.output/f'{stem}.engine'
    out_shape = export(model, onnx_path)
    onnx_check = check_onnx(onnx_path, n_conv, [])
    print(f'ONNX {onnx_path.name}: output {out_shape}, {onnx_check}', flush=True)

    print('Building TensorRT engine (detailed)...', flush=True)
    command = build_engine(onnx_path, engine_path, args.output/f'{stem}.layers.json', args.output/f'{stem}.build.log')
    precision = myelin_precision(args.output/f'{stem}.layers.json', args.output/f'{stem}.precision.csv')
    print(f'Engine {engine_path.name}: {engine_path.stat().st_size/1e6:.2f} MB, {precision}', flush=True)

    results = trainer.read_results_csv()
    # Same selection as the saved checkpoint: Ultralytics fitness is mAP50-95.
    best_epoch = max(range(len(results['epoch'])), key=lambda i: results['metrics/mAP50-95(B)'][i])
    amax = {name: {'input_amax': float(m.input_quantizer.amax), 'weight_amax_min': float(m.weight_quantizer.amax.min()),
                   'weight_amax_max': float(m.weight_quantizer.amax.max())} for name, m in convs}
    manifest = {'source': str(args.weights), 'source_sha256': sha256(args.weights), 'modelopt': modelopt.__version__,
                'torch': torch.__version__, 'config': INT8_CONV_CFG, 'bn_fused_before_quantization': True,
                'quantized_convs': len(convs), 'non_conv_quantizers_disabled': others,
                'calibration': {'images': len(indices), 'split': 'KITTI train', 'seed': args.seed, 'imgsz': 640,
                                'augment': False, 'algorithm': 'max'},
                'qat': {'epochs': args.epochs, 'batch': args.batch, 'optimizer': 'AdamW', 'lr0': args.lr0, 'lrf': 0.1,
                        'cos_lr': True, 'amp': False, 'quantizer_ranges': 'fixed after calibration',
                        'frozen': ['model.22.dfl'], 'run_dir': str(trainer.save_dir), 'train_seconds': train_seconds,
                        'best_epoch': int(results['epoch'][best_epoch]),
                        'best_val': {k: results[k][best_epoch] for k in results if k.startswith('metrics/')}},
                'checkpoint': checkpoint.name,
                'onnx': {'file': onnx_path.name, 'opset': 17, 'output_shape': out_shape, **onnx_check},
                'trtexec': command, 'engine': {'file': engine_path.name, 'size_mb': engine_path.stat().st_size/1e6,
                                               **precision},
                'quantizer_amax': amax}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Done: {args.output}')


if __name__ == '__main__':
    main()
