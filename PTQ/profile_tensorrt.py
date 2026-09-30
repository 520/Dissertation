"""Profile the existing KITTI/VOC engines. Run from TENSORRT_PROFILING.ipynb or CLI."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import tensorrt as trt
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import nms, ops

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'PTQ/profiling_results'
MODELS = {
    'kitti': (ROOT / 'original/yolov8n_kitti/8n_float32_fp32.engine', ROOT / 'PTQ/datasets/kitti/images/val'),
    'voc': (ROOT / 'original/yolov8n_voc/voc_8n_best_fp32.engine', ROOT / 'PTQ/datasets/VOC/images/test2007'),
}


def summary(values):
    return dict(mean_ms=float(np.mean(values)), median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)), samples=len(values))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


class ProfilePredictor(DetectionPredictor):
    """Same preprocessing as installed Ultralytics 8.4.163, with explicit NVTX ranges.

    sync_stages is only enabled in the diagnostic decomposition pass. Natural
    timeline collection keeps asynchronous CUDA execution and the backend graph.
    """
    sync_stages = False
    record = False

    @contextmanager
    def stage(self, name, gpu=False):
        torch.cuda.nvtx.range_push(name)
        begin = time.perf_counter()
        try:
            yield
            if self.sync_stages and gpu:
                torch.cuda.synchronize()
        finally:
            elapsed = (time.perf_counter() - begin) * 1000
            torch.cuda.nvtx.range_pop()
            if self.record:
                self.current[name] = self.current.get(name, 0.0) + elapsed

    def preprocess(self, im):
        if isinstance(im, torch.Tensor):
            raise ValueError('This profiling protocol requires preloaded BGR numpy images.')
        with self.stage('01_CPU_letterbox'):
            im = self.pre_transform(im)
        with self.stage('02_CPU_tensor_view'):
            im = torch.from_numpy(im[0]).unsqueeze(0) if len(im) == 1 else torch.from_numpy(np.stack(im))
        with self.stage('03_H2D_uint8', gpu=True):
            im = im.to(self.device)
        with self.stage('04_GPU_layout_cast_normalize', gpu=True):
            im = im.permute(0, 3, 1, 2)
            if im.shape[1] == 3:
                im = im.flip(1)
            im = im.contiguous()
            im = (im.half() if self.model.fp16 else im.float()).div_(255)
        return im

    def inference(self, im, *args, **kwargs):
        # Backend inference includes the D2D copy into the graph's input buffer.
        with self.stage('05_TensorRT_backend_D2D_and_graph', gpu=True):
            return super().inference(im, *args, **kwargs)

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        if getattr(self, '_feats', None) is not None:
            raise ValueError('Feature extraction is outside this detection profiling protocol.')
        with self.stage('06_NMS', gpu=True):
            preds = nms.non_max_suppression(
                preds, self.args.conf, kwargs.pop('iou', self.args.iou),
                self.args.classes, self.args.agnostic_nms, max_det=self.args.max_det,
                nc=0 if self.args.task == 'detect' else len(self.model.names),
                end2end=getattr(self.model, 'end2end', False),
                rotated=self.args.task == 'obb', return_idxs=False,
            )
        with self.stage('07_scale_boxes_and_Results', gpu=True):
            return self.construct_results(preds, img, orig_imgs, **kwargs)


class LayerProfiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.current = defaultdict(float)

    def report_layer_time(self, layer_name, ms):
        self.current[layer_name] += float(ms)


def layer_profile(model, predictor, image, iterations, key):
    backend = predictor.model.backend
    engine = backend.model
    inspector = engine.create_engine_inspector()
    write_json(OUT / f'{key}_engine_inspector.json', {
        'profiling_verbosity': str(engine.profiling_verbosity),
        'layers': inspector.get_engine_information(trt.LayerInformationFormat.JSON),
    })
    context = engine.create_execution_context()
    buffers = {}
    tensor = predictor.preprocess([image])
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            context.set_input_shape(name, tuple(tensor.shape))
            buffers[name] = tensor
        else:
            shape = tuple(context.get_tensor_shape(name))
            dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(engine.get_tensor_dtype(name)))).dtype
            buffers[name] = torch.empty(shape, dtype=dtype, device='cuda:0')
        context.set_tensor_address(name, buffers[name].data_ptr())
    profile_stream = torch.cuda.Stream()
    profile_stream.wait_stream(torch.cuda.current_stream())
    stream = profile_stream.cuda_stream
    for _ in range(20):
        if not context.execute_async_v3(stream):
            raise RuntimeError('TensorRT warmup failed')
    torch.cuda.synchronize()
    profiler = LayerProfiler()
    context.profiler = profiler
    context.enqueue_emits_profile = False
    # Capture with the profiler already attached so layer CUDA events are replayed.
    # This keeps the deployment's CUDA Graph execution pattern.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=profile_stream):
        if not context.execute_async_v3(stream):
            raise RuntimeError('TensorRT graph capture failed')
    torch.cuda.synchronize()
    rows = []
    for _ in range(iterations):
        profiler.current.clear()
        with torch.cuda.stream(profile_stream):
            graph.replay()
        torch.cuda.synchronize()
        if not context.report_to_profiler():
            raise RuntimeError('TensorRT did not report layer timing')
        rows.append(dict(profiler.current))
    names = sorted({name for row in rows for name in row})
    total = float(np.mean([sum(row.values()) for row in rows]))
    layers = [dict(layer=name, **summary([row.get(name, 0.0) for row in rows])) for name in names]
    for row in layers:
        row['share_pct'] = 100 * row['mean_ms'] / total if total else 0
    layers.sort(key=lambda row: row['mean_ms'], reverse=True)
    payload = dict(model=key, method='IProfiler, separate context, CUDA Graph replay with per-layer CUDA events',
                   note='Fused TensorRT layers. Instrumented timing; not baseline latency. One fixed image/input.',
                   profiling_verbosity=str(engine.profiling_verbosity), layer_sum_mean_ms=total,
                   iterations=iterations, layers=layers, raw_iterations=rows)
    write_json(OUT / f'{key}_layers.json', payload)
    with (OUT / f'{key}_layers.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=['layer','mean_ms','median_ms','p95_ms','samples','share_pct'])
        writer.writeheader()
        writer.writerows(layers)
    print(f'{key}: {len(layers)} fused layers; layer sum {total:.3f} ms')
    for row in layers[:10]:
        print(f"  {row['mean_ms']:.4f} ms ({row['share_pct']:.1f}%) {row['layer']}")


@torch.inference_mode()
def run(key, mode, iterations=100, warmup=30, image_count=100):
    OUT.mkdir(parents=True, exist_ok=True)
    engine, folder = MODELS[key]
    paths = sorted(path for path in folder.iterdir() if path.suffix.lower() in {'.png','.jpg','.jpeg'})[:image_count]
    images = [cv2.imread(str(path)) for path in paths]
    if not images or any(image is None for image in images):
        raise RuntimeError(f'Failed to preload images from {folder}')
    model = YOLO(str(engine), task='detect')
    options = dict(imgsz=640, device=0, rect=False, conf=0.25, iou=0.7,
                   max_det=300, verbose=False, save=False, predictor=ProfilePredictor)
    model.predict(images[0], **options)
    predictor = model.predictor
    # Check the instrumented preprocessing against this installed version.
    candidate = predictor.preprocess([images[0]])
    reference = DetectionPredictor.preprocess(predictor, [images[0]])
    if not torch.equal(candidate, reference):
        raise RuntimeError('Preprocessing differs from installed Ultralytics; update ProfilePredictor.')
    for i in range(warmup):
        model.predict(images[i % len(images)], **options)
    torch.cuda.synchronize()
    if mode == 'layers':
        layer_profile(model, predictor, images[0], iterations, key)
        return
    predictor.sync_stages = mode == 'stages'
    predictor.record = True
    rows = []
    original_profile_time = ops.Profile.time
    if mode == 'timeline':
        def annotated_profile_time(profile):
            # Annotate existing framework waits without adding synchronization.
            torch.cuda.nvtx.range_push('Framework_GPU_sync')
            try:
                return original_profile_time(profile)
            finally:
                torch.cuda.nvtx.range_pop()
        ops.Profile.time = annotated_profile_time
        status = torch.cuda.cudart().cudaProfilerStart()
        if int(status) != 0:
            raise RuntimeError(f'cudaProfilerStart failed: {status}')
    try:
        for i in range(iterations):
            predictor.current = {}
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(f'E2E/{key}/image_{i:04d}')
            start = time.perf_counter()
            model.predict(images[i % len(images)], **options)
            with predictor.stage('08_final_GPU_wait', gpu=True):
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            torch.cuda.nvtx.range_pop()
            rows.append(dict(image=str(paths[i % len(images)]), e2e_ms=elapsed, **predictor.current))
    finally:
        if mode == 'timeline':
            torch.cuda.cudart().cudaProfilerStop()
            ops.Profile.time = original_profile_time
    stages = {name: summary([row[name] for row in rows]) for name in predictor.current}
    residual = [row['e2e_ms'] - sum(row[name] for name in stages) for row in rows]
    stages['09_framework_and_instrumentation_overhead'] = summary(residual)
    result = dict(model=key, engine=str(engine), engine_sha256=hashlib.sha256(engine.read_bytes()).hexdigest(),
                  mode=mode, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  ultralytics=ultralytics.__version__, tensorrt=trt.__version__,
                  cuda_graph_enabled=predictor.model.backend.graph is not None,
                  iterations=iterations, warmup=warmup, imgsz=640, batch=1, conf=0.25, iou=0.7,
                  image_count=len(images), e2e=summary([row['e2e_ms'] for row in rows]), stages=stages,
                  notes=('Each GPU stage synchronizes: additive diagnostic decomposition, altered scheduling.'
                         if mode == 'stages' else
                         'Natural async path: stage values are CPU wall ranges/launch time, NOT GPU stage durations. '
                         'Inspect CUDA GPU rows in Nsight. Final wait closes each E2E range.'),
                  excludes='disk decode/read, model load, initialization, visualization, D2H export of detections',
                  samples=rows)
    write_json(OUT / f'{key}_{mode}.json', result)
    print(f"{key} {mode}: E2E {result['e2e']}")
    if mode == 'stages':
        for name, values in stages.items():
            print(f"  {name}: mean={values['mean_ms']:.3f}, P95={values['p95_ms']:.3f} ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODELS), required=True)
    parser.add_argument('--mode', choices=['stages','layers','timeline'], required=True)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=30)
    args = parser.parse_args()
    run(args.model, args.mode, args.iterations, args.warmup)
