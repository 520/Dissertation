"""INT8 QAT of YOLOv8n-KITTI with ModelOpt: 63 convolutions quantized, DFL kept FP32, then Q/DQ ONNX + trtexec.

Same quantization as modelopt_int8_all_convs.py --exclude model.22.dfl.conv (BN folded first, INT8
per-channel weights, INT8 per-tensor inputs, max calibration on KITTI train), followed by fine-tuning
with the fake quantizers in the graph (straight-through estimator, amax fixed from calibration).
Training: Ultralytics v8DetectionLoss and train-time augmentation, SGD with cosine LR, EMA weights,
mosaic closed for the last --close-mosaic epochs. The EMA checkpoint with the best val mAP50-95 is exported.

Outputs (in --output):
  qat_last.pth / qat_best.pth   ModelOpt checkpoints (mto.save) of the EMA model
  train_log.csv                 per-epoch losses, lr and val metrics
  <stem>.onnx                   explicit Q/DQ ONNX of qat_best
  <stem>.engine                 TensorRT engine, profilingVerbosity=detailed
  <stem>.layers.json            trtexec --exportLayerInfo
  <stem>.build.log              trtexec log
  <stem>.precision.csv          per-convolution weight/input/output datatypes in the engine
  manifest.json                 settings, checks, training history and per-layer quantizer amax
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
import time
from pathlib import Path

import torch
import yaml
from torch import nn

import modelopt
import modelopt.torch.opt as mto
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import ModelEMA

import modelopt_int8_all_convs as ptq
from modelopt_int8_all_convs import (INT8_CONV_CFG, KITTI, ROOT, ExportWrapper, build_engine, calibration_images,
                                     check_onnx, engine_precision, load_fused, quantize, sha256)

DFL = 'model.22.dfl.conv'


def write_data_yaml(data_dir):
    path = Path(tempfile.gettempdir())/'kitti_qat.yaml'
    path.write_text(yaml.safe_dump({'path': str(data_dir), 'train': 'images/train', 'val': 'images/val',
                                    'names': KITTI, 'nc': len(KITTI)}), encoding='utf-8')
    return path


def validate(model, data_yaml, batch):
    """mAP on KITTI val of a (fake-quantized) FP32 module, without changing its training state."""
    was_training = model.training
    validator = DetectionValidator(args=dict(data=str(data_yaml), imgsz=640, batch=batch, half=False, plots=False,
                                             save_json=False, verbose=False, workers=4))
    stats = validator(model=model)
    model.train(was_training)
    return {'mAP50-95': stats['metrics/mAP50-95(B)'], 'mAP50': stats['metrics/mAP50(B)'],
            'precision': stats['metrics/precision(B)'], 'recall': stats['metrics/recall(B)']}


def param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.ndim > 1 else no_decay).append(p)       # no decay on biases (BN already folded)
    return [{'params': decay, 'weight_decay': weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/baseline/8n_float32.pt')
    parser.add_argument('--data', type=Path, default=ROOT/'PTQ/datasets/kitti')
    parser.add_argument('--output', type=Path, default=ROOT/'models/quantization/int8/modelopt_qat_63conv_keep_dfl')
    parser.add_argument('--calib', type=int, default=512, help='number of KITTI training images')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4, help='initial LR (about 1%% of the FP32 lr0)')
    parser.add_argument('--lrf', type=float, default=0.01, help='final LR = lr * lrf (cosine)')
    parser.add_argument('--warmup-iters', type=int, default=300)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--close-mosaic', type=int, default=10)
    parser.add_argument('--val-interval', type=int, default=5, help='validate every n epochs (and the last)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trt', type=Path, default=ptq.TRT, help='TensorRT root containing bin/trtexec.exe')
    parser.add_argument('--export-only', action='store_true',
                        help='skip calibration/training; export qat_best.pth from --output and build the engine')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ptq.TRT = args.trt
    torch.manual_seed(args.seed)
    n_conv = sum(isinstance(m, nn.Conv2d) for m in load_fused(args.weights).modules())
    stem = f'{args.weights.stem}_int8_qat_{n_conv - 1}conv'
    manifest_path = args.output/'manifest.json'
    if args.export_only:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    else:
        manifest = train(args, n_conv)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    # 3. Export the best EMA model exactly like the PTQ pipeline.
    model = mto.restore(load_fused(args.weights), args.output/'qat_best.pth').eval()
    onnx_path, engine_path = args.output/f'{stem}.onnx', args.output/f'{stem}.engine'
    wrapper = ExportWrapper(model.cpu()).eval()     # CPU: the TorchScript exporter crashes with CUDA fake-quant
    for m in wrapper.modules():
        if isinstance(m, Detect):
            m.shape = None
    dummy = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        out_shape = tuple(wrapper(dummy).shape)
        torch.onnx.export(wrapper, dummy, str(onnx_path), opset_version=17, dynamo=False,
                          input_names=['images'], output_names=['output0'], do_constant_folding=True)
    onnx_check = check_onnx(onnx_path, n_conv, [DFL])
    print(f'ONNX {onnx_path.name}: output {out_shape}, {onnx_check}', flush=True)
    quantized = [(n, m) for n, m in model.named_modules()
                 if isinstance(m, nn.Conv2d) and hasattr(m, 'input_quantizer') and m.input_quantizer.is_enabled]
    manifest.update({
        'quantized_convs': len(quantized),
        'onnx': {'file': onnx_path.name, 'opset': 17, 'output_shape': out_shape, **onnx_check},
        'quantizer_amax': {name: {'input_amax': float(m.input_quantizer.amax),
                                  'weight_amax_min': float(m.weight_quantizer.amax.min()),
                                  'weight_amax_max': float(m.weight_quantizer.amax.max())} for name, m in quantized}})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    if not (args.trt/'bin/trtexec.exe').is_file():
        print(f'trtexec not found under {args.trt}; engine skipped. Build it later with --export-only --trt <dir>')
        return
    print('Building TensorRT engine (detailed)...', flush=True)
    command = build_engine(onnx_path, engine_path, args.output/f'{stem}.layers.json', args.output/f'{stem}.build.log')
    precision = engine_precision(args.output/f'{stem}.layers.json', args.output/f'{stem}.precision.csv')
    print(f'Engine {engine_path.name}: {engine_path.stat().st_size/1e6:.2f} MB, {precision}', flush=True)
    manifest.update({'trtexec': command,
                     'engine': {'file': engine_path.name, 'size_mb': engine_path.stat().st_size/1e6, **precision}})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Done: {args.output}')


def train(args, n_conv):
    """PTQ initialisation then QAT; saves qat_best.pth / qat_last.pth and returns the manifest so far."""
    device = torch.device('cuda:0')
    data_yaml = write_data_yaml(args.data)
    # 1. PTQ initialisation: identical to the 63-conv PTQ model.
    model = load_fused(args.weights).to(device)
    calib_set, indices = calibration_images(args.data, args.calib, args.seed)
    print(f'Calibrating on {len(indices)} KITTI training images...', flush=True)
    model, convs, others = quantize(model, calib_set, indices, device, [DFL])
    if len(convs) != n_conv - 1:
        raise RuntimeError(f'Only {len(convs)}/{n_conv - 1} convolutions were quantized')
    ptq_metrics = validate(model, data_yaml, args.batch * 2)
    print(f'PTQ start point: {ptq_metrics}', flush=True)

    # 2. QAT fine-tuning. The DFL projection is a fixed constant in Ultralytics and stays frozen.
    for name, p in model.named_parameters():
        p.requires_grad_(not name.startswith(DFL))
    hyp = get_cfg(overrides={'imgsz': 640, 'batch': args.batch, 'close_mosaic': args.close_mosaic})
    model.args = hyp                                           # v8DetectionLoss reads box/cls/dfl gains here
    criterion = v8DetectionLoss(model)
    data = {'path': str(args.data), 'train': 'images/train', 'val': 'images/val', 'names': KITTI, 'nc': len(KITTI),
            'channels': 3}
    train_set = build_yolo_dataset(hyp, str(args.data/'images/train'), args.batch, data, mode='train',
                                   stride=int(model.stride.max()))
    loader = build_dataloader(train_set, args.batch, args.workers, shuffle=True, rank=-1)
    optimizer = torch.optim.SGD(param_groups(model, args.weight_decay), lr=args.lr, momentum=0.937, nesterov=True)
    total_iters = args.epochs * len(loader)
    lr_at = lambda it: (min(1.0, (it + 1) / args.warmup_iters) *
                        args.lr * (args.lrf + (1 - args.lrf) * 0.5 * (1 + math.cos(math.pi * it / total_iters))))
    ema = ModelEMA(model)
    history, best, it, begin = [], (-1.0, None), 0, time.perf_counter()
    for epoch in range(args.epochs):
        if epoch == args.epochs - args.close_mosaic:
            print('Closing mosaic', flush=True)
            train_set.close_mosaic(hyp=hyp)
            loader.reset()
        model.train()
        sums, seen = torch.zeros(3, device=device), 0
        for batch in loader:
            for g in optimizer.param_groups:
                g['lr'] = lr_at(it)
            batch['img'] = batch['img'].to(device, non_blocking=True).float() / 255
            loss, items = criterion(model(batch['img']), batch)
            optimizer.zero_grad(set_to_none=True)
            loss.sum().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            ema.update(model)
            sums += torch.stack(list(items.values())) if isinstance(items, dict) else items.detach()
            seen += 1
            it += 1
        row = {'epoch': epoch + 1, 'lr': lr_at(it - 1), **dict(zip(('box', 'cls', 'dfl'), (sums / seen).tolist()))}
        if (epoch + 1) % args.val_interval == 0 or epoch + 1 == args.epochs:
            row.update(validate(ema.ema, data_yaml, args.batch * 2))
            mto.save(ema.ema, args.output/'qat_last.pth')
            if row['mAP50-95'] > best[0]:
                best = (row['mAP50-95'], epoch + 1)
                mto.save(ema.ema, args.output/'qat_best.pth')
        history.append(row)
        print(f"epoch {epoch + 1}/{args.epochs} " + ' '.join(f'{k}={v:.4g}' for k, v in row.items() if k != 'epoch'),
              flush=True)
        with open(args.output/'train_log.csv', 'w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['epoch', 'lr', 'box', 'cls', 'dfl', 'mAP50-95', 'mAP50',
                                                        'precision', 'recall'])
            writer.writeheader()
            writer.writerows(history)
    train_seconds = time.perf_counter() - begin
    print(f'Best val mAP50-95 {best[0]:.4f} at epoch {best[1]}', flush=True)
    return {'source': str(args.weights), 'source_sha256': sha256(args.weights), 'modelopt': modelopt.__version__,
            'torch': torch.__version__, 'config': INT8_CONV_CFG, 'bn_fused_before_quantization': True,
            'unquantized_convs_fp32': [DFL], 'non_conv_quantizers_disabled': others,
            'calibration': {'images': len(indices), 'split': 'KITTI train', 'seed': args.seed, 'imgsz': 640,
                            'augment': False, 'algorithm': 'max'},
            'ptq_start_val': ptq_metrics,
            'qat': {'epochs': args.epochs, 'batch': args.batch, 'optimizer': 'SGD nesterov momentum 0.937',
                    'lr': args.lr, 'lrf': args.lrf, 'schedule': 'cosine', 'warmup_iters': args.warmup_iters,
                    'weight_decay': args.weight_decay, 'close_mosaic': args.close_mosaic, 'ema': True,
                    'amax': 'fixed from calibration', 'best_epoch': best[1], 'best_val_mAP50-95': best[0],
                    'train_seconds': train_seconds, 'history': history}}


if __name__ == '__main__':
    main()
