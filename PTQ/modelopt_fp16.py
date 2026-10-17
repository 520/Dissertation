"""FP16 YOLOv8n-KITTI with ModelOpt AutoCast: FP32 ONNX export, FP16 conversion, trtexec build.

The FP32 ONNX is exported exactly like the INT8 pipeline (BN folded, same wrapper, opset 17), then
modelopt.onnx.autocast converts it to FP16. Nodes whose activations exceed --data-max on the
calibration images stay FP32 (AutoCast's overflow rule); graph inputs/outputs stay FP32.
TensorRT 11 is strongly typed, so the engine runs every tensor in the ONNX datatype.
Requires the nvidia-modelopt[onnx] dependencies.

Outputs (in --output):
  <stem>_fp32.onnx         FP32 ONNX (AutoCast input)
  <stem>.onnx              FP16 ONNX
  <stem>.engine            TensorRT engine, profilingVerbosity=detailed
  <stem>.layers.json       trtexec --exportLayerInfo
  <stem>.build.log         trtexec log
  <stem>.precision.csv     per-convolution weight/input/output datatypes in the engine
  manifest.json            settings and checks
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
from ultralytics.nn.modules.head import Detect

import modelopt
from modelopt.onnx.autocast import convert_to_mixed_precision

import modelopt_int8_all_convs as ptq
from modelopt_int8_all_convs import ROOT, ExportWrapper, build_engine, calibration_images, engine_precision, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'models/baseline/8n_float32.pt')
    parser.add_argument('--data', type=Path, default=ROOT/'PTQ/datasets/kitti')
    parser.add_argument('--output', type=Path, default=ROOT/'models/quantization/fp16/modelopt')
    parser.add_argument('--calib', type=int, default=32, help='KITTI training images for the AutoCast range check')
    parser.add_argument('--data-max', type=float, default=512, help='AutoCast: keep nodes with larger outputs FP32')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trt', type=Path, default=ptq.TRT, help='TensorRT root containing bin/trtexec.exe')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ptq.TRT = args.trt
    stem = f'{args.weights.stem}_fp16'
    fp32_path, onnx_path = args.output/f'{stem}_fp32.onnx', args.output/f'{stem}.onnx'

    # 1. FP32 ONNX, exported like the INT8 pipeline.
    wrapper = ExportWrapper(ptq.load_fused(args.weights)).eval()
    for m in wrapper.modules():
        if isinstance(m, Detect):
            m.shape = None
    dummy = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        out_shape = tuple(wrapper(dummy).shape)
        torch.onnx.export(wrapper, dummy, str(fp32_path), opset_version=17, dynamo=False,
                          input_names=['images'], output_names=['output0'], do_constant_folding=True)

    # 2. FP16 with AutoCast, range-checked on real KITTI images.
    dataset, indices = calibration_images(args.data, args.calib, args.seed)
    calib_path = args.output/f'{stem}_calib.npz'
    np.savez(calib_path, images=np.stack([dataset[i]['img'].float().numpy() / 255 for i in indices]))
    model = convert_to_mixed_precision(str(fp32_path), low_precision_type='fp16', keep_io_types=True,
                                       data_max=args.data_max, calibration_data=str(calib_path))
    calib_path.unlink()
    onnx.save(model, str(onnx_path))
    onnx.checker.check_model(str(onnx_path))
    graph = model.graph
    dtype = {t.name: t.type.tensor_type.elem_type for t in graph.value_info}
    dtype.update({t.name: t.data_type for t in graph.initializer})
    conv_types = Counter(onnx.TensorProto.DataType.Name(dtype.get(n.input[1], 0)) for n in graph.node if n.op_type == 'Conv')
    kept_fp32 = [n.name for n in graph.node if n.op_type != 'Cast' and n.output
                 and dtype.get(n.output[0]) == onnx.TensorProto.FLOAT]
    onnx_check = {'opset': 17, 'output_shape': out_shape, 'conv_weight_types': dict(conv_types),
                  'casts': sum(n.op_type == 'Cast' for n in graph.node), 'nodes_kept_fp32': kept_fp32}
    print(f'ONNX {onnx_path.name}: {onnx_check}', flush=True)
    manifest = {'source': str(args.weights), 'source_sha256': sha256(args.weights), 'modelopt': modelopt.__version__,
                'torch': torch.__version__, 'bn_fused': True,
                'autocast': {'low_precision_type': 'fp16', 'keep_io_types': True, 'data_max': args.data_max,
                             'calibration': {'images': args.calib, 'split': 'KITTI train', 'seed': args.seed}},
                'onnx': {'fp32_file': fp32_path.name, 'file': onnx_path.name, **onnx_check}}
    manifest_path = args.output/'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    # 3. Engine.
    print('Building TensorRT engine (detailed)...', flush=True)
    engine_path = args.output/f'{stem}.engine'
    command = build_engine(onnx_path, engine_path, args.output/f'{stem}.layers.json', args.output/f'{stem}.build.log')
    precision = engine_precision(args.output/f'{stem}.layers.json', args.output/f'{stem}.precision.csv')
    print(f'Engine {engine_path.name}: {engine_path.stat().st_size/1e6:.2f} MB, {precision}', flush=True)
    manifest.update({'trtexec': command,
                     'engine': {'file': engine_path.name, 'size_mb': engine_path.stat().st_size/1e6, **precision}})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Done: {args.output}')


if __name__ == '__main__':
    main()
