"""trtexec latency + per-layer profile for the baseline and LRF+pruned FP32 engines.

Ultralytics .engine files carry a 4-byte length + JSON metadata header; it is stripped to a raw .plan first.
Runs alternate between models to spread thermal drift. Results are merged into report.json under 'trtexec'.
Run: .venv/Scripts/python.exe figures/lrf_latency/trtexec_compare.py [--repeat 3] [--duration 15]
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'trtexec'
TRT = Path(os.environ.get('TRT_DIR', Path(os.environ['LOCALAPPDATA']) / 'NVIDIA/TensorRT-11.3-tools'))
TRTEXEC = TRT / 'bin/trtexec.exe'


def plan_for(engine):
    data = Path(engine).read_bytes()
    n = int.from_bytes(data[:4], 'little')
    plan = OUT / (Path(engine).stem + '.plan')
    plan.write_bytes(data[4 + n:])
    return plan


def stat(text, label):
    m = re.search(rf'{label}: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms', text)
    return dict(min=float(m[1]), max=float(m[2]), mean=float(m[3]), median=float(m[4])) if m else None


def gpu_temp():
    out = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


def wait_cool(limit, timeout=600):
    """The RTX A2000 throttles under sustained load; start every run from the same thermal state."""
    start = time.time()
    while gpu_temp() > limit and time.time() - start < timeout:
        time.sleep(5)
    return gpu_temp()


def run(key, plan, rep, duration, mode):
    prefix = OUT / f'{key}_{mode}_run{rep}'
    cmd = [str(TRTEXEC), f'--loadEngine={plan}', '--noDataTransfers', '--warmUp=500', f'--duration={duration}',
           '--iterations=1000', '--percentile=50,95', '--dumpProfile', '--separateProfileRun',
           f'--exportProfile={prefix}_profile.json', f'--exportTimes={prefix}_times.json']
    if mode == 'graph':
        cmd.append('--useCudaGraph')
    env = dict(os.environ, PATH=f"{TRT / 'bin'};{TRT / 'lib'};{os.environ['PATH']}")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    Path(f'{prefix}.log').write_text(proc.stdout + proc.stderr, encoding='utf-8')
    if '&&&& PASSED' not in proc.stdout:
        raise RuntimeError(f'trtexec failed for {key}, see {prefix}.log')
    profile = json.loads(Path(f'{prefix}_profile.json').read_text())
    layers = {row['name']: row['medianMs'] for row in profile if 'name' in row}
    return dict(gpu_compute=stat(proc.stdout, 'GPU Compute Time'), latency=stat(proc.stdout, 'Latency'),
                enqueue=stat(proc.stdout, 'Enqueue Time'),
                throughput=float(re.search(r'Throughput: ([\d.]+) qps', proc.stdout)[1]), layers=layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeat', type=int, default=4)
    ap.add_argument('--duration', type=int, default=5)
    ap.add_argument('--cool', type=int, default=60, help='start each run at or below this GPU temperature (°C)')
    ap.add_argument('--modes', default='graph,nograph')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    report = json.loads((HERE / 'report.json').read_text(encoding='utf-8'))
    plans = {k: plan_for(v['engine']) for k, v in report.items()}
    keys = list(report)
    for mode in args.modes.split(','):
        runs = {k: [] for k in keys}
        for rep in range(args.repeat):
            # ABBA order cancels linear drift.
            for key in (keys if rep % 2 == 0 else keys[::-1]):
                t0 = wait_cool(args.cool)
                r = run(key, plans[key], rep, args.duration, mode)
                r.update(temp_start_c=t0, temp_end_c=gpu_temp())
                runs[key].append(r)
                print(f"{mode} {key} run{rep}: {t0}->{r['temp_end_c']} °C, GPU compute median "
                      f"{r['gpu_compute']['median']:.4f} ms, enqueue median {r['enqueue']['median']:.4f} ms, "
                      f"{r['throughput']:.1f} qps, layer sum {sum(r['layers'].values()):.4f} ms", flush=True)
        for key, rs in runs.items():
            names = set().union(*(r['layers'] for r in rs))
            med = [r['gpu_compute']['median'] for r in rs]
            report[key].setdefault('trtexec', {})[mode] = dict(
                method=f"trtexec 11.3.0.99, {'--useCudaGraph ' if mode == 'graph' else ''}--noDataTransfers, "
                       f"{args.duration}s per run, ABBA order, each run started at <= {args.cool} °C, "
                       'separate profiling run, random input',
                runs=len(rs), gpu_compute_median_ms=med, gpu_compute_mean_ms=[r['gpu_compute']['mean'] for r in rs],
                enqueue_median_ms=[r['enqueue']['median'] for r in rs], throughput_qps=[r['throughput'] for r in rs],
                temp_c=[[r['temp_start_c'], r['temp_end_c']] for r in rs],
                median_ms=float(np.mean(med)), median_sd_ms=float(np.std(med, ddof=1)) if len(rs) > 1 else 0.0,
                layers={n: float(np.mean([r['layers'].get(n, 0.0) for r in rs])) for n in names})
    (HERE / 'report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
