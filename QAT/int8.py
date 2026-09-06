#!/usr/bin/env python3
"""Convert custom QAT checkpoints to CPU quantized Conv2d islands.

Weights use the trained symmetric QAT ranges. Conv input/output ranges are
calibrated on real images because the original observers are AFTER BN/SiLU.
BN, activations, residuals and detection decoding remain floating point.
Run from the project root: python -m QAT.int8 --help
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import torch
from torch import nn
from ultralytics.data.augment import LetterBox

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from QAT.torch_qat import MinMaxFakeQuant, QATConv2d


class Int8Conv2d(nn.Module):
    """Float boundary around a real packed-weight quantized::conv2d CPU op."""

    def __init__(self, conv, input_params, output_params):
        super().__init__()
        self.input_scale, self.input_zero_point = input_params
        self.conv = torch.ao.nn.quantized.Conv2d(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            conv.stride, conv.padding, conv.dilation, conv.groups,
            conv.bias is not None, conv.padding_mode,
        )
        fq = conv.weight_fake_quant
        if not fq.initialized:
            raise ValueError('Uninitialized QAT weight observer')
        scale = max(float(torch.maximum(fq.min_val.abs(), fq.max_val.abs())) / 127,
                    torch.finfo(torch.float32).eps)
        # Preserve QAT's [-127, 127] clamp (qint8 itself also allows -128).
        weight = torch.quantize_per_tensor(
            conv.weight.detach().clamp(-127 * scale, 127 * scale),
            scale, 0, torch.qint8,
        )
        self.conv.set_weight_bias(weight, conv.bias.detach() if conv.bias is not None else None)
        self.conv.scale, self.conv.zero_point = output_params

    def __getstate__(self):
        state = super().__getstate__()
        state['_modules'] = dict(state['_modules'])
        conv = state['_modules'].pop('conv')
        state['_packed_conv_config'] = (
            conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
            conv.padding, conv.dilation, conv.groups, conv.bias() is not None,
            conv.padding_mode,
        )
        state['_int8_engine'] = torch.backends.quantized.engine
        state['_packed_conv_state'] = conv.state_dict()
        return state

    def __setstate__(self, state):
        backend = state.pop('_int8_engine', 'qnnpack')
        if backend not in torch.backends.quantized.supported_engines:
            raise RuntimeError(f'Unavailable INT8 backend: {backend}')
        torch.backends.quantized.engine = backend
        config = state.pop('_packed_conv_config')
        weights = state.pop('_packed_conv_state')
        super().__setstate__(state)
        self.conv = torch.ao.nn.quantized.Conv2d(*config)
        self.conv.load_state_dict(weights)

    def forward(self, x):
        if x.device.type != 'cpu':
            raise ValueError('This INT8 model requires device=cpu')
        x = torch.quantize_per_tensor(x, self.input_scale, self.input_zero_point, torch.quint8)
        return self.conv(x).dequantize()


def qparams(bounds):
    low, high = min(bounds[0], 0.), max(bounds[1], 0.)
    scale = max((high - low) / 255, torch.finfo(torch.float32).eps)
    return scale, max(0, min(255, round(-low / scale)))


def image_tensor(path, size):
    im = cv2.imread(str(path))
    if im is None:
        raise ValueError(f'Cannot read image: {path}')
    im = LetterBox(new_shape=(size, size), auto=False)(image=im)
    return torch.from_numpy(im[:, :, ::-1].transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255


def convert(model, batches):
    model = model.cpu().float().eval()
    for m in model.modules():
        if isinstance(m, MinMaxFakeQuant):
            m.observer_enabled = False
            if not m.initialized:
                raise ValueError('Checkpoint contains uninitialized QAT ranges')
            m.fake_quant_enabled = True
    convs = {name: m for name, m in model.named_modules() if isinstance(m, QATConv2d)}
    if not convs:
        raise ValueError('No custom QAT convolutions found')
    ranges, handles = {}, []
    def hook(name):
        def observe(module, inputs, output):
            bounds = ranges.setdefault(name, [[float('inf'), -float('inf')] for _ in range(2)])
            for limits, tensor in zip(bounds, (inputs[0], output)):
                if not torch.isfinite(tensor).all():
                    raise ValueError(f'Non-finite calibration values: {name}')
                limits[0] = min(limits[0], float(tensor.min()))
                limits[1] = max(limits[1], float(tensor.max()))
        return observe
    for name, conv in convs.items():
        handles.append(conv.register_forward_hook(hook(name)))
    count = 0
    try:
        with torch.inference_mode():
            for batch in batches:
                model(batch)
                count += len(batch)
    finally:
        for handle in handles:
            handle.remove()
    if count == 0 or set(ranges) != set(convs):
        raise ValueError('Calibration must exercise every QAT convolution')
    for name, conv in convs.items():
        parent_name, _, attr = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, Int8Conv2d(conv, qparams(ranges[name][0]), qparams(ranges[name][1])))
    # Remove the remaining simulated activation quantization for deployment.
    for name, module in list(model.named_modules()):
        if isinstance(module, MinMaxFakeQuant):
            parent_name, _, attr = name.rpartition('.')
            setattr(model.get_submodule(parent_name), attr, nn.Identity())
    model.int8_backend = torch.backends.quantized.engine
    model.int8_conv_count = len(convs)
    return model, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'QAT/runs/kitti_qat_10e-3/weights/best.pt')
    parser.add_argument('--images', type=Path, default=ROOT / 'datasets/kitti/images/train')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--backend', default='qnnpack')
    args = parser.parse_args()
    if args.samples < 1 or args.imgsz < 32 or args.imgsz % 32:
        parser.error('samples must be positive; imgsz must be a positive multiple of 32')
    if args.backend not in torch.backends.quantized.supported_engines or args.backend == 'none':
        parser.error(f'Unsupported backend: {args.backend}')
    torch.backends.quantized.engine = args.backend
    torch.set_num_threads(4)
    output = args.output or args.model.with_name('best_real_int8.pt')
    if output.resolve() == args.model.resolve() or output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    paths = sorted(p for p in args.images.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if not paths:
        raise ValueError(f'No calibration images in {args.images}')
    # Deterministic sample spread across the training image list.
    indices = torch.linspace(0, len(paths)-1, min(args.samples, len(paths))).long().tolist()
    paths = [paths[i] for i in indices]
    model = torch.load(args.model, map_location='cpu', weights_only=False)
    if isinstance(model, dict):
        model = model.get('ema') if model.get('ema') is not None else model.get('model')
    model, count = convert(model, (image_tensor(p, args.imgsz) for p in paths))
    sample = image_tensor(paths[0], args.imgsz)
    with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        prediction = model(sample)[0]
    events = {e.key: e.count for e in prof.key_averages() if 'quantized::conv2d' in e.key}
    if not events or not torch.isfinite(prediction).all():
        raise RuntimeError('INT8 operator / finite output verification failed')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.pt")
    torch.save(model, temporary)
    reloaded = torch.load(temporary, map_location='cpu', weights_only=False)
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(sample)[0], prediction, rtol=0, atol=0)
    temporary.replace(output)
    report = dict(model=str(output), backend=args.backend, calibration_images=count,
                  calibration_source=str(args.images), imgsz=args.imgsz,
                  quantized_convs=model.int8_conv_count, runtime_operators=events,
                  precision='int8_convs_float_remainder', reload_verified=True)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    # Keep serialized custom classes importable when launched from an IDE.
    from QAT.int8 import main as canonical_main
    canonical_main()
