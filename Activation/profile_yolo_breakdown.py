"""Profile original KITTI YOLO forward operators without double counting."""
import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from Activation.profile_yolo_activation import load_model, DEFAULT_GIT_MODEL


def category(name):
    if 'conv' in name:
        return 'Convolution'
    if 'batch_norm' in name:
        return 'BatchNorm'
    if 'silu' in name:
        return 'SiLU'
    if 'pool' in name:
        return 'Pooling'
    if 'upsample' in name:
        return 'Upsampling'
    if name == 'aten::cat':
        return 'Concatenation'
    if any(s in name for s in ('copy_', 'contiguous', 'clone')):
        return 'Copy / layout'
    if name == 'model_forward':
        return 'Python / dispatch gaps'
    return 'Other operators'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--threads', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--results', type=Path)
    args = parser.parse_args()
    if args.threads < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error('threads/repeats must be positive and warmup nonnegative')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS is not available in this PyTorch environment')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    model = load_model(None, DEFAULT_GIT_MODEL).to(args.device)
    x = torch.randn(1, 3, 640, 640).to(args.device)
    def synchronize():
        if args.device == 'mps':
            torch.mps.synchronize()
    repeats = 30
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x)
        synchronize()
        samples = []
        for _ in range(args.repeats):
            synchronize()
            start = time.perf_counter()
            model(x)
            synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        if args.device == 'mps':
            # CPU profiler events measure host dispatch, not Metal kernel time.
            result = dict(model=DEFAULT_GIT_MODEL, device='cpu',
                          torch_version=torch.__version__, threads=args.threads,
                          input_shape=list(x.shape), repeats=args.repeats,
                          scope='forward only; excludes preprocessing and NMS',
                          unprofiled_median_ms=statistics.median(samples),
                          timing='wall clock with MPS synchronization',
                          note='GPU operator breakdown requires Apple Instruments Metal tracing; CPU operator percentages are not GPU percentages.')
            dest = args.results or Path(__file__).with_name('original_kitti_operator_profile_mps.json')
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(result, indent=2))
            print(f'Saved: {dest.resolve()}')
            return
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            for _ in range(repeats):
                with record_function('model_forward'):
                    model(x)
    events = prof.key_averages()
    total = next(e.cpu_time_total for e in events if e.key == 'model_forward')
    groups = defaultdict(float)
    rows = []
    for e in events:
        groups[category(e.key)] += e.self_cpu_time_total
        rows.append(dict(operator=e.key, calls_per_forward=e.count/repeats,
                         self_ms_per_forward=e.self_cpu_time_total/1000/repeats,
                         percent=100*e.self_cpu_time_total/total))
    result = dict(model=DEFAULT_GIT_MODEL, torch_version=torch.__version__,
                  device='cpu', threads=args.threads, input_shape=list(x.shape),
                  input='seeded random normal tensor', fusion='checkpoint as loaded',
                  scope='forward only; excludes preprocessing and NMS',
                  unprofiled_median_ms=statistics.median(samples),
                  profiled_mean_ms=total/1000/repeats,
                  profiled_forwards=repeats,
                  categories=[dict(category=k, ms_per_forward=v/1000/repeats,
                                   percent=100*v/total)
                              for k,v in sorted(groups.items(), key=lambda p:-p[1])],
                  operators=sorted(rows, key=lambda r:-r['self_ms_per_forward']))
    dest = args.results or Path(__file__).with_name('original_kitti_operator_profile.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
