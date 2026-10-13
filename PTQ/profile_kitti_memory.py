"""Inventory, capture target, and strict Nsight Compute CSV aggregation.

Inventory/aggregate use Python's standard library. Infer requires the engine's
matching TensorRT runtime, numpy and CUDA-enabled torch. See the generated report.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'original/yolov8n_kitti/8n_float32_fp32.engine'
OLD = ROOT / 'PTQ/profiling_results'
OUT = OLD / 'kitti_memory'
METRICS = {
    'duration_ns': 'gpu__time_duration.sum',
    'L1_global_read_sectors': 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
    'L1_global_write_sectors': 'l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum',
    'L1_all_sectors': 'l1tex__t_sectors.sum',
    'L1_hit_pct': 'l1tex__t_sector_hit_rate.pct',
    'L2_read_sectors': 'lts__t_sectors_op_read.sum',
    'L2_write_sectors': 'lts__t_sectors_op_write.sum',
    'L2_all_sectors': 'lts__t_sectors.sum',
    'L2_hit_pct': 'lts__t_sector_hit_rate.pct',
    'DRAM_read_bytes': 'dram__bytes_read.sum',
    'DRAM_write_bytes': 'dram__bytes_write.sum',
}


def inventory():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((OLD / 'trtexec/kitti_input_metadata.json').read_text())
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != meta['engine_sha256']:
        raise RuntimeError('Engine differs from the existing profiling source.')
    inspector = json.loads((OLD / 'kitti_engine_inspector.json').read_text())
    names = json.loads(inspector['layers'])['Layers']
    if len(set(names)) != len(names):
        raise RuntimeError('Duplicate layer names require separate launch attribution.')
    profiles = [json.loads((OLD / f'trtexec/kitti_run{i}_profile.json').read_text()) for i in (1, 2, 3)]
    timings = [{r['name']: r['averageMs'] for r in p if 'name' in r} for p in profiles]
    rows = []
    for index, name in enumerate(names):
        rows.append(dict(layer_index=index, layer=name,
                         previous_instrumented_mean_ms=sum(t[name] for t in timings) / 3,
                         measurement_status='NOT_MEASURED', **{key: '' for key in METRICS}))
    write_csv(OUT / 'per_layer_memory.csv', rows)
    (OUT / 'metrics.txt').write_text(','.join(METRICS.values()), encoding='utf-8')
    (OUT / 'inventory.json').write_text(json.dumps(dict(
        engine=str(SOURCE), engine_sha256=digest, layers=len(names),
        profiling_verbosity=inspector['profiling_verbosity'],
        metrics=METRICS, status='NOT_MEASURED'), indent=2), encoding='utf-8')
    return rows


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def infer():
    import numpy as np
    import torch
    import tensorrt as trt
    inventory()
    data = SOURCE.read_bytes()
    size = int.from_bytes(data[:4], 'little')
    json.loads(data[4:4 + size])  # Validate Ultralytics metadata prefix.
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(data[4 + size:])
    if engine is None:
        raise RuntimeError('Cannot load engine; use the matching TensorRT/CUDA runtime.')
    context = engine.create_execution_context()
    context.nvtx_verbosity = trt.ProfilingVerbosity.LAYER_NAMES_ONLY
    stream = torch.cuda.Stream()
    tensors = {}
    meta = json.loads((OLD / 'trtexec/kitti_input_metadata.json').read_text())
    input_file = OLD / 'trtexec/kitti_images_fp32.bin'
    if hashlib.sha256(input_file.read_bytes()).hexdigest() != meta['input_sha256']:
        raise RuntimeError('Fixed input hash mismatch.')
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(context.get_tensor_shape(name))
        if any(d < 0 for d in shape) or engine.get_tensor_dtype(name) != trt.float32:
            raise RuntimeError(f'Unexpected shape/dtype: {name}')
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            tensors[name] = torch.from_numpy(np.fromfile(input_file, dtype=np.float32).reshape(shape)).cuda()
        else:
            tensors[name] = torch.empty(shape, dtype=torch.float32, device='cuda')
        context.set_tensor_address(name, tensors[name].data_ptr())
    torch.cuda.synchronize()
    for _ in range(30):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('Warmup enqueue failed.')
    stream.synchronize()
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    try:
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('Profile enqueue failed.')
        stream.synchronize()
    finally:
        cudart.cudaProfilerStop()
    print('Captured exactly one inference after 30 warmups; no CUDA Graph; input transfers excluded.')


def aggregate(path):
    # Export raw with --print-nvtx-rename kernel; require exact engine layer names.
    raw = path.read_text(encoding='utf-8-sig')
    lines = raw.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID",'))
    records = list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))))
    rows = inventory()
    by_name = {r['layer']: r for r in rows}
    launches = {}
    unknown = []
    for r in records:
        if r['Metric Name'] not in METRICS.values():
            continue
        if r['Kernel Name'] not in by_name:
            unknown.append(r)
            continue
        key = (r.get('Process ID', ''), r['ID'])
        item = launches.setdefault(key, dict(layer=r['Kernel Name'], values={}))
        value = r['Metric Value'].replace(',', '')
        item['values'][r['Metric Name']] = float(value)
    if unknown:
        write_csv(OUT / 'unmapped_kernels.csv', unknown)
        raise RuntimeError('NVTX names do not match engine layers; see unmapped_kernels.csv. No guessed mapping.')
    if not launches:
        raise RuntimeError('No attributed kernels found.')
    per_layer = {name: [] for name in by_name}
    for launch in launches.values():
        missing = set(METRICS.values()) - set(launch['values'])
        if missing:
            raise RuntimeError(f'Missing metrics for {launch["layer"]}: {missing}')
        per_layer[launch['layer']].append(launch['values'])
    for row in rows:
        kernels = per_layer[row['layer']]
        row['kernel_count'] = len(kernels)
        row['measurement_status'] = 'MEASURED' if kernels else 'NO_ATTRIBUTED_KERNEL'
        if not kernels:
            continue
        for label, metric in METRICS.items():
            if label.endswith('hit_pct'):
                weight = METRICS[label.replace('hit_pct', 'all_sectors')]
                total = sum(k[weight] for k in kernels)
                row[label] = sum(k[metric] * k[weight] for k in kernels) / total if total else ''
            else:
                row[label] = sum(k[metric] for k in kernels)
        row['L1_global_read_bytes'] = 32 * row['L1_global_read_sectors']
        row['L1_global_write_bytes'] = 32 * row['L1_global_write_sectors']
        row['L2_read_bytes'] = 32 * row['L2_read_sectors']
        row['L2_write_bytes'] = 32 * row['L2_write_sectors']
    # Keep the same schema for layers with no attributed GPU launch.
    fields = list(dict.fromkeys(key for row in rows for key in row))
    rows = [{key: row.get(key, '') for key in fields} for row in rows]
    write_csv(OUT / 'per_layer_memory_measured.csv', rows)
    print(f'{len(launches)} kernels attributed; {len(rows)} layer rows exported.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['inventory', 'infer', 'aggregate'])
    parser.add_argument('--csv', type=Path)
    args = parser.parse_args()
    if args.mode == 'inventory':
        print(f'{len(inventory())} layers inventoried; hardware metrics not measured.')
    elif args.mode == 'infer':
        infer()
    else:
        if args.csv is None:
            parser.error('aggregate requires --csv')
        aggregate(args.csv)
