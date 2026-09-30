from pathlib import Path
import gc, json, os, platform, threading, time
from datetime import datetime, timezone
import cv2
import numpy as np
from IPython.display import display, HTML
import html
import torch
import tensorrt as trt
import pynvml as nv
import ultralytics
from ultralytics import YOLO, settings
import ultralytics.data.utils as data_utils
from ultralytics.utils.torch_utils import get_num_params, get_flops

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'original').is_dir())
OUTPUT = ROOT / 'PTQ/tensorrt_baseline_results'
OUTPUT.mkdir(parents=True, exist_ok=True)
DEVICE = 0
IMGSZ = 640
LATENCY_IMAGES = 100
WARMUP = 20
MIN_IMAGES = 200
RUN_SECONDS = 15.0
IDLE_SECONDS = 5.0
SAMPLE_SECONDS = 0.1
CONF, IOU = 0.25, 0.70
DATASETS = ROOT / 'PTQ/datasets'
settings.update({'datasets_dir': str(DATASETS)})
data_utils.DATASETS_DIR = DATASETS
assert torch.cuda.is_available(), 'CUDA-enabled PyTorch is required'
torch.cuda.set_device(DEVICE)
nv.nvmlInit()
gpu_uuid = torch.cuda.get_device_properties(DEVICE).uuid
GPU = nv.nvmlDeviceGetHandleByUUID(str(gpu_uuid))

def safe(fn, *args):
    try:
        value = fn(*args)
        return value if value < 2**63 else None
    except nv.NVMLError:
        return None

def memory_snapshot():
    info = nv.nvmlDeviceGetMemoryInfo(GPU)
    process = None
    try:
        matches = [p.usedGpuMemory for p in nv.nvmlDeviceGetComputeRunningProcesses(GPU)
                   if p.pid == os.getpid() and p.usedGpuMemory is not None and p.usedGpuMemory < 2**63]
        if matches:
            process = sum(matches) / 2**20
    except nv.NVMLError:
        pass
    return {'gpu_total_used_mib': info.used / 2**20, 'process_used_mib': process,
            'torch_allocated_mib': torch.cuda.memory_allocated(DEVICE) / 2**20,
            'torch_reserved_mib': torch.cuda.memory_reserved(DEVICE) / 2**20}

def sample_gpu():
    try:
        util = nv.nvmlDeviceGetUtilizationRates(GPU)
        gpu_util, memory_util = util.gpu, util.memory
    except nv.NVMLError:
        gpu_util = memory_util = None
    return {'t': time.perf_counter(), 'power_w': (p / 1000 if (p := safe(nv.nvmlDeviceGetPowerUsage, GPU)) is not None else None),
            'temperature_c': safe(nv.nvmlDeviceGetTemperature, GPU, nv.NVML_TEMPERATURE_GPU),
            'sm_clock_mhz': safe(nv.nvmlDeviceGetClockInfo, GPU, nv.NVML_CLOCK_SM),
            'memory_clock_mhz': safe(nv.nvmlDeviceGetClockInfo, GPU, nv.NVML_CLOCK_MEM),
            'gpu_util_percent': gpu_util, 'memory_util_percent': memory_util,
            **memory_snapshot()}

class Sampler:
    def __init__(self):
        self.rows, self.stop_event = [], threading.Event()
    def start(self):
        self.rows.append(sample_gpu())
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self
    def loop(self):
        while not self.stop_event.wait(SAMPLE_SECONDS):
            self.rows.append(sample_gpu())
    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.rows.append(sample_gpu())
        return self.rows

def stats(values):
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if not x.size:
        return {'mean': None, 'median': None, 'min': None, 'max': None}
    return {k: float(f(x)) for k, f in [('mean', np.mean), ('median', np.median), ('min', np.min), ('max', np.max)]}

def summarize_telemetry(rows, start, end, images, idle_power):
    power = [r for r in rows if r['power_w'] is not None]
    energy = None
    if len(power) >= 2:
        t = np.array([r['t'] for r in power])
        p = np.array([r['power_w'] for r in power])
        inner = t[(t > start) & (t < end)]
        knots = np.r_[start, inner, end]
        energy = float(np.trapezoid(np.interp(knots, t, p), knots))
    inside = [r for r in rows if start <= r['t'] <= end] or rows
    fields = ['power_w', 'temperature_c', 'sm_clock_mhz', 'memory_clock_mhz',
              'gpu_util_percent', 'memory_util_percent', 'gpu_total_used_mib', 'process_used_mib']
    result = {key: stats([r[key] for r in inside]) for key in fields}
    duration = end - start
    result.update({'sampling_interval_s': SAMPLE_SECONDS, 'samples': len(inside),
                   'duration_s': duration, 'images': images,
                   'mean_power_w_time_weighted': energy / duration if energy is not None else None,
                   'gpu_energy_j': energy,
                   'gpu_energy_j_per_image': energy / images if energy is not None and images else None,
                   'incremental_gpu_energy_j_per_image': (energy - idle_power * duration) / images
                      if energy is not None and idle_power is not None and images else None})
    return result

def idle_measure():
    torch.cuda.synchronize()
    sampler = Sampler().start()
    start = time.perf_counter()
    time.sleep(IDLE_SECONDS)
    end = time.perf_counter()
    rows = sampler.stop()
    summary = summarize_telemetry(rows, start, end, 0, None)
    return summary['mean_power_w_time_weighted'], summary, rows

def latency_summary(ms, wall_seconds):
    x = np.asarray(ms)
    return {'images': len(ms), 'mean_ms': float(x.mean()), 'median_ms': float(np.median(x)),
            'p95_ms': float(np.percentile(x, 95)), 'fps_inverse_mean_latency': float(1000 / x.mean()),
            'wall_duration_s': wall_seconds, 'fps_observed_wall': len(ms) / wall_seconds}

def load_engine(path):
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    with path.open('rb') as f:
        n = int.from_bytes(f.read(4), 'little')
        metadata = {}
        offset = 0
        if 0 < n < path.stat().st_size - 4:
            try:
                metadata = json.loads(f.read(n))
                offset = n + 4
            except (ValueError, UnicodeDecodeError):
                pass
        f.seek(offset)
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f'Failed to deserialize {path}')
    context = engine.create_execution_context()
    buffers = {}
    dtypes = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
              trt.int64: torch.int64, trt.int8: torch.int8, trt.bool: torch.bool}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        if any(d < 0 for d in shape):
            raise RuntimeError('This baseline requires static batch-1 engines')
        buffers[name] = torch.empty(shape, device=f'cuda:{DEVICE}', dtype=dtypes[engine.get_tensor_dtype(name)])
        context.set_tensor_address(name, buffers[name].data_ptr())
    return runtime, engine, context, buffers, metadata

def clean_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def engine_only(path, image):
    clean_cuda()
    before = memory_snapshot()
    runtime, engine, context, buffers, metadata = load_engine(path)
    from ultralytics.data.augment import LetterBox
    prepared = LetterBox((IMGSZ, IMGSZ), auto=False)(image=image)
    array = np.ascontiguousarray(prepared[:, :, ::-1].transpose(2, 0, 1)[None])
    input_names = [name for name in buffers if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
    if len(input_names) != 1:
        raise RuntimeError('Expected a single image input')
    buffers[input_names[0]].copy_(torch.from_numpy(array).to(DEVICE).to(buffers[input_names[0]].dtype) / 255)
    stream = torch.cuda.current_stream(DEVICE)
    for _ in range(WARMUP):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT enqueue failed')
    torch.cuda.synchronize()
    loaded = memory_snapshot()
    idle_power, idle, idle_rows = idle_measure()
    begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []
    sampler = Sampler().start()
    start = time.perf_counter()
    try:
        while len(timings) < MIN_IMAGES or time.perf_counter() - start < RUN_SECONDS:
            begin.record(stream)
            ok = context.execute_async_v3(stream.cuda_stream)
            finish.record(stream)
            finish.synchronize()
            if not ok:
                raise RuntimeError('TensorRT enqueue failed')
            timings.append(begin.elapsed_time(finish))
        end = time.perf_counter()
    finally:
        rows = sampler.stop()
    result = {'latency': latency_summary(timings, end-start), 'memory_before_load': before,
              'memory_model_loaded': loaded, 'memory_after_inference': memory_snapshot(),
              'idle_power_w': idle_power, 'idle_telemetry': idle,
              'telemetry': summarize_telemetry(rows, start, end, len(timings), idle_power),
              'engine_metadata': metadata,
              'io_types': {name: str(engine.get_tensor_dtype(name)) for name in buffers}}
    del context, buffers, engine, runtime
    clean_cuda()
    return result, {'idle': idle_rows, 'running': rows, 'latency_ms': timings}

def end_to_end(path, images):
    clean_cuda()
    before = memory_snapshot()
    model = YOLO(str(path), task='detect')
    kwargs = dict(imgsz=IMGSZ, device=DEVICE, conf=CONF, iou=IOU, rect=False,
                  save=False, verbose=False)
    for i in range(WARMUP):
        model.predict(images[i % len(images)], **kwargs)
    torch.cuda.synchronize()
    loaded = memory_snapshot()
    idle_power, idle, idle_rows = idle_measure()
    timings = []
    sampler = Sampler().start()
    start = time.perf_counter()
    try:
        while len(timings) < MIN_IMAGES or time.perf_counter() - start < RUN_SECONDS:
            torch.cuda.synchronize()
            begin = time.perf_counter()
            prediction = model.predict(images[len(timings) % len(images)], **kwargs)
            torch.cuda.synchronize()
            timings.append((time.perf_counter()-begin)*1000)
            del prediction
        end = time.perf_counter()
    finally:
        rows = sampler.stop()
    result = {'latency': latency_summary(timings, end-start), 'memory_before_load': before,
              'memory_model_loaded': loaded, 'memory_after_inference': memory_snapshot(),
              'idle_power_w': idle_power, 'idle_telemetry': idle,
              'telemetry': summarize_telemetry(rows, start, end, len(timings), idle_power)}
    del model
    clean_cuda()
    return result, {'idle': idle_rows, 'running': rows, 'latency_ms': timings}

def discover():
    engines = sorted(p for p in (ROOT/'original').rglob('*.engine') if p.name.lower().endswith('_fp32.engine'))
    configs = []
    for engine in engines:
        weights = engine.with_name(engine.name[:-len('_fp32.engine')] + '.pt')
        folder = engine.parent.name.lower()
        if 'voc' in folder:
            data, images = 'VOC.yaml', DATASETS/'VOC/images/test2007'
        elif 'kitti' in folder:
            data, images = 'kitti.yaml', DATASETS/'kitti/images/val'
        else:
            raise ValueError(f'Cannot infer dataset for {engine}')
        if not weights.is_file() or not images.is_dir():
            raise FileNotFoundError(f'Missing weights or dataset: {weights}, {images}')
        configs.append({'engine': engine, 'weights': weights, 'data': data, 'images': images})
    if not configs:
        raise FileNotFoundError('No *_fp32.engine models in original')
    return configs

def benchmark_model(config):
    engine, weights = config['engine'], config['weights']
    print(f'\nBENCHMARK: {engine.name}', flush=True)
    paths = sorted(p for p in config['images'].iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})[:LATENCY_IMAGES]
    images = [cv2.imread(str(p)) for p in paths]
    if not images or any(im is None for im in images):
        raise RuntimeError('Cannot preload validation images')
    reference = YOLO(str(weights), task='detect')
    params = int(get_num_params(reference.model))
    gflops = float(get_flops(reference.model, IMGSZ))
    if not gflops:
        raise RuntimeError('THOP FLOPs profiling failed')
    del reference
    clean_cuda()
    print('Running full validation split...', flush=True)
    detector = YOLO(str(engine), task='detect')
    metrics = detector.val(data=config['data'], split='val', imgsz=IMGSZ, batch=1, device=DEVICE,
                           rect=False, workers=0, plots=False, save_json=False, verbose=False,
                           project=str(OUTPUT/'validation'), name=engine.stem, exist_ok=True)
    accuracy = {'map50_95': float(metrics.box.map), 'map50': float(metrics.box.map50),
                'recall': float(metrics.box.mr),
                'per_class_recall': np.asarray(metrics.box.r).tolist(),
                'recall_definition': 'Ultralytics mean class recall at its validation-selected max-mean-F1 operating point'}
    del detector, metrics
    clean_cuda()
    print('Measuring engine-only latency and GPU telemetry...', flush=True)
    engine_result, engine_samples = engine_only(engine, images[0])
    print('Measuring end-to-end latency and GPU telemetry...', flush=True)
    e2e_result, e2e_samples = end_to_end(engine, images)
    result = {'model': str(engine.relative_to(ROOT)), 'weights': str(weights.relative_to(ROOT)),
              'dataset': config['data'], 'accuracy': accuracy, 'parameters': params,
              'gflops': gflops, 'gmacs': gflops/2, 'weights_size_mb': weights.stat().st_size/1e6,
              'engine_size_mb': engine.stat().st_size/1e6, 'engine_only': engine_result,
              'end_to_end': e2e_result, 'latency_image_count': len(images)}
    (OUTPUT/f'{engine.stem}_samples.json').write_text(json.dumps({'engine_only':engine_samples,'end_to_end':e2e_samples}, indent=2), encoding='utf-8')
    (OUTPUT/f'{engine.stem}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'model': engine.name, **accuracy,
                      'engine_median_ms':engine_result['latency']['median_ms'],
                      'e2e_median_ms':e2e_result['latency']['median_ms']}, indent=2), flush=True)
    return result

CONFIGS = discover()
ENVIRONMENT = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'gpu': torch.cuda.get_device_name(DEVICE),
               'gpu_uuid': str(gpu_uuid), 'python': platform.python_version(), 'torch':torch.__version__,
               'tensorrt':trt.__version__, 'ultralytics':ultralytics.__version__,
               'driver':nv.nvmlSystemGetDriverVersion(), 'imgsz':IMGSZ, 'batch':1,
               'run_seconds_min':RUN_SECONDS, 'idle_seconds':IDLE_SECONDS, 'warmup':WARMUP,
               'sampling_interval_s':SAMPLE_SECONDS, 'pid':os.getpid()}
print(json.dumps(ENVIRONMENT, indent=2))
print('FP32 engines:', [str(c['engine'].relative_to(ROOT)) for c in CONFIGS])
