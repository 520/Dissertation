"""Compare baseline YOLOv8n vs LRF+pruned model: per-stage FLOPs, conv count, PyTorch and TensorRT FP32 latency."""
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

ROOT = Path(r'C:\Users\dxu1\Desktop\Dissertation')
sys.path[:0] = [str(ROOT), str(ROOT / 'PTQ'), str(ROOT / 'Decomposition'), str(ROOT / 'Pruning')]
from ultralytics import YOLO  # noqa: E402

OUT = Path(__file__).resolve().parent / 'latency'
MODELS = {
    'baseline': ROOT / 'models/baseline/8n_float32.pt',
    'lrf_ch8': ROOT / 'models/lrf_both/budget1.7/ch8_b0.2/8n_float32_lrf_ft50_ch8_b0.2_ft10.pt',
}
X = torch.zeros(1, 3, 640, 640, device='cuda')


def load(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return (ck.get('ema') or ck['model']).float().eval().cuda()


def stage_stats(m):
    """Per top-level stage: conv count, GMACs, output tensor MB written by convs."""
    stats = defaultdict(lambda: dict(convs=0, gmacs=0.0, conv_out_mb=0.0))
    hooks = []
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            stage = int(name.split('.')[1])
            def hook(mod, inp, out, stage=stage):
                s = stats[stage]
                s['convs'] += 1
                k = mod.kernel_size[0] * mod.kernel_size[1]
                s['gmacs'] += out.numel() * mod.in_channels // mod.groups * k / 1e9
                s['conv_out_mb'] += out.numel() * 4 / 1e6
            hooks.append(mod.register_forward_hook(hook))
    with torch.inference_mode():
        m(X)
    for h in hooks:
        h.remove()
    return dict(stats)


@torch.inference_mode()
def torch_latency(m, iters=300):
    for _ in range(50):
        m(X)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); m(X); b.record(); torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return float(np.median(times))


@torch.inference_mode()
def torch_stage_time(m, iters=200):
    """Instrumented: CUDA events around each top-level stage (adds sync overhead, use for shares only)."""
    acc = defaultdict(list)
    ev = {}
    hooks = []
    for i, layer in enumerate(m.model):
        def pre(mod, inp, i=i):
            e = torch.cuda.Event(enable_timing=True); e.record(); ev[i] = [e]
        def post(mod, inp, out, i=i):
            e = torch.cuda.Event(enable_timing=True); e.record(); ev[i].append(e)
        hooks += [layer.register_forward_pre_hook(pre), layer.register_forward_hook(post)]
    for it in range(iters + 30):
        m(X); torch.cuda.synchronize()
        if it >= 30:
            for i, (a, b) in ev.items():
                acc[i].append(a.elapsed_time(b))
    for h in hooks:
        h.remove()
    return {i: float(np.median(v)) for i, v in acc.items()}


class Profiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.cur = defaultdict(float)

    def report_layer_time(self, name, ms):
        self.cur[name] += float(ms)


def build_engine(key, path):
    work = OUT / key
    work.mkdir(parents=True, exist_ok=True)
    pt = work / path.name
    if not pt.exists():
        shutil.copy(path, pt)
    eng = pt.with_suffix('.engine')
    if not eng.exists():
        YOLO(str(pt), task='detect').export(format='engine', imgsz=640, batch=1, half=False, device=0,
                                            simplify=True, workspace=4, verbose=False)
    return eng


def trt_profile(eng_path, iters=300):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(eng_path, 'rb') as f:
        meta_len = int.from_bytes(f.read(4), 'little')
        try:
            json.loads(f.read(meta_len).decode('utf-8'))
        except Exception:
            f.seek(0)
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    bufs = {}
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(n))
        bufs[n] = torch.zeros(shape, dtype=torch.float32, device='cuda')
        ctx.set_tensor_address(n, bufs[n].data_ptr())
    stream = torch.cuda.Stream()
    s = stream.cuda_stream
    with torch.cuda.stream(stream):
        for _ in range(100):
            ctx.execute_async_v3(s)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(stream); ctx.execute_async_v3(s); b.record(stream); torch.cuda.synchronize()
            times.append(a.elapsed_time(b))
    latency = float(np.median(times))
    prof = Profiler()
    ctx.profiler = prof
    rows = []
    for _ in range(100):
        prof.cur.clear()
        ctx.execute_async_v3(s); torch.cuda.synchronize()
        rows.append(dict(prof.cur))
    names = {n for r in rows for n in r}
    layers = {n: float(np.median([r.get(n, 0.0) for r in rows])) for n in names}
    insp = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    return latency, layers, insp


def stage_of(layer_name):
    hits = re.findall(r'model\.(\d+)', layer_name)
    return int(hits[0]) if hits else -1


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for key, path in MODELS.items():
        m = load(path)
        r = dict(params=sum(p.numel() for p in m.parameters()), stages=stage_stats(m),
                 torch_ms=torch_latency(m), torch_stage_ms=torch_stage_time(m))
        del m
        torch.cuda.empty_cache()
        eng = build_engine(key, path)
        lat, layers, insp = trt_profile(eng)
        by_stage = defaultdict(lambda: dict(ms=0.0, kernels=0))
        for n, ms in layers.items():
            st = stage_of(n)
            by_stage[st]['ms'] += ms
            by_stage[st]['kernels'] += 1
        r.update(engine=str(eng), trt_ms=lat, trt_layer_count=len(layers), trt_layer_sum_ms=sum(layers.values()),
                 trt_stage=dict(by_stage), trt_layers=dict(sorted(layers.items(), key=lambda x: -x[1])),
                 trt_inspector_layers=len(insp.get('Layers', [])))
        report[key] = r
        print(f"{key}: params {r['params']/1e6:.3f}M torch {r['torch_ms']:.3f} ms  TRT {lat:.3f} ms  "
              f"TRT layers {len(layers)} sum {r['trt_layer_sum_ms']:.3f} ms", flush=True)
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
