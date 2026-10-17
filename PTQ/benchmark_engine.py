"""Benchmark any KITTI TensorRT engine with the exact procedure of tensorrt_baseline.ipynb.

The notebook's measurement cell is executed unchanged; this wrapper only adapts inputs:
- a trtexec engine without the Ultralytics metadata header gets a prefixed copy for val/E2E;
- a checkpoint saved as a bare DetectionModel is wrapped as {'model': ...} for YOLO().
Weights (MB) is measured from the engine: sum over layers of weight and bias element counts
times bytes per element, read from the engine inspector (profilingVerbosity=detailed) or from a
trtexec --exportLayerInfo file. Engine (MB) is the original engine file size.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
HEADERS = ['Precision', 'mAP50-95 (%)', 'mAP50 (%)', 'mAP75 (%)', 'Detection Precision (%)', 'Recall (%)',
           'Engine median / P75 / P95 (ms)', 'E2E median / P75 / P95 (ms)', 'E2E FPS', 'Params (M)', 'GFLOPs',
           'Weights (MB)', 'Engine (MB)', 'Approx. GPU load delta (MiB)', 'E2E GPU power (W)',
           'E2E GPU energy (J/image)']


def has_metadata(path):
    with path.open('rb') as f:
        n = int.from_bytes(f.read(4), 'little')
        try:
            return 0 < n < path.stat().st_size - 4 and isinstance(json.loads(f.read(n)), dict)
        except (ValueError, UnicodeDecodeError):
            return False


BYTES = {'Int8': 1, 'FP8': 1, 'Half': 2, 'BFloat16': 2, 'Float': 4, 'Int32': 4, 'Int64': 8}


def engine_weights(engine, layer_info=None):
    """Weight + bias bytes stored in the engine, per data type."""
    if layer_info is not None:
        layers, source = json.loads(Path(layer_info).read_text())['Layers'], str(layer_info)
    else:
        import tensorrt as trt
        raw = engine.read_bytes()
        n = int.from_bytes(raw[:4], 'little')
        offset = 4 + n if has_metadata(engine) else 0
        plan = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(raw[offset:])
        info = plan.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
        layers, source = json.loads(info)['Layers'], 'engine inspector'
    counts, total = {}, 0
    for layer in layers:
        for key in ('Weights', 'Bias'):
            entry = layer.get(key) if isinstance(layer, dict) else None
            if isinstance(entry, dict) and entry.get('Count'):
                counts[f"{key}/{entry['Type']}"] = counts.get(f"{key}/{entry['Type']}", 0) + entry['Count']
                total += entry['Count'] * BYTES[entry['Type']]
    if not total:
        raise RuntimeError(f'{engine.name}: no weight counts in the engine (built without '
                           'profilingVerbosity=detailed). Pass --layer-info from a detailed trtexec build.')
    return {'bytes': total, 'counts': counts, 'source': source}


def table_row(label, r):
    e = r['engine_only']['latency']; end = r['end_to_end']; l = end['latency']; t = end['telemetry']; a = r['accuracy']
    return [label, f"{a['map50_95']*100:.2f}", f"{a['map50']*100:.2f}", f"{a['map75']*100:.2f}",
            f"{a['precision']*100:.2f}", f"{a['recall']*100:.2f}",
            f"{e['median_ms']:.3f} / {e['p75_ms']:.3f} / {e['p95_ms']:.3f}",
            f"{l['median_ms']:.3f} / {l['p75_ms']:.3f} / {l['p95_ms']:.3f}", f"{l['fps_observed_wall']:.1f}",
            f"{r['parameters']/1e6:.3f}", f"{r['gflops']:.2f}", f"{r['weights_size_mb']:.2f}",
            f"{r['engine_size_mb']:.2f}",
            f"{end['memory_model_loaded']['gpu_total_used_mib']-end['memory_before_load']['gpu_total_used_mib']:.2f}",
            f"{t['mean_power_w_time_weighted']:.2f}", f"{t['gpu_energy_j_per_image']:.4f}"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--label', default='FP32')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layer-info', type=Path, default=None,
                        help='trtexec --exportLayerInfo JSON, if the engine lacks detailed layer info')
    args = parser.parse_args()
    engine, weights = args.engine.resolve(), args.weights.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    weight_info = engine_weights(engine, args.layer_info)
    print(f"Engine weights: {weight_info['bytes']/1e6:.2f} MB {weight_info['counts']}", flush=True)

    run_engine = engine
    if not has_metadata(engine):
        reference = ROOT/'original/yolov8n_kitti/8n_float32_fp32.engine'
        with reference.open('rb') as f:
            meta = json.loads(f.read(int.from_bytes(f.read(4), 'little')))
        model = torch.load(weights, map_location='cpu', weights_only=False)
        model = model.get('model', model) if isinstance(model, dict) else model
        meta.update({'description': f'trtexec engine {engine.name} (metadata added for Ultralytics)',
                     'stride': int(model.stride.max()), 'names': {str(k): v for k, v in model.names.items()}})
        blob = json.dumps(meta).encode()
        run_engine = output/f'{engine.stem}.with_metadata.engine'
        run_engine.write_bytes(len(blob).to_bytes(4, 'little') + blob + engine.read_bytes())

    run_weights = weights
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        run_weights = output/f'{weights.stem}.wrapped.pt'
        torch.save({'model': checkpoint}, run_weights)

    nb = json.loads((ROOT/'tensorrt_baseline.ipynb').read_text(encoding='utf-8'))
    source = next(''.join(c['source']) for c in nb['cells']
                  if c['cell_type'] == 'code' and 'def benchmark_model' in ''.join(c['source']))
    namespace = {'__name__': 'benchmark'}
    exec(compile(source, 'tensorrt_baseline.ipynb', 'exec'), namespace)
    namespace['OUTPUT'] = output
    config = {'engine': run_engine, 'weights': run_weights, 'data': 'kitti.yaml',
              'images': namespace['DATASETS']/'kitti/images/val'}
    result = namespace['benchmark_model'](config)
    result.update({'model': str(engine.relative_to(ROOT)), 'weights': str(weights.relative_to(ROOT)),
                   'weights_size_mb': weight_info['bytes']/1e6, 'weights_size_source': weight_info['source'],
                   'engine_weight_counts': weight_info['counts'],
                   'checkpoint_file_mb': weights.stat().st_size/1e6, 'engine_size_mb': engine.stat().st_size/1e6,
                   'benchmark_engine_file': str(run_engine.relative_to(ROOT)),
                   'benchmark_weights_file': str(run_weights.relative_to(ROOT))})
    environment = {**namespace['ENVIRONMENT'], 'timestamp_utc': datetime.now(timezone.utc).isoformat()}
    (output/'results.json').write_text(json.dumps({'environment': environment, 'models': [result]}, indent=2),
                                       encoding='utf-8')
    table = ('| ' + ' | '.join(HEADERS) + ' |\n| ' + ' | '.join(['---']*len(HEADERS)) + ' |\n| '
             + ' | '.join(table_row(args.label, result)) + ' |')
    (output/'metrics.md').write_text(table + '\n', encoding='utf-8')
    print(table)


if __name__ == '__main__':
    main()
