"""Single-layer microbenchmark: a 3x3 conv vs its Tucker-2 factorisations, across resolution and rank.

Original layer: Conv3x3(C->C) + SiLU, with C set to the YOLOv8n width at each resolution so every original has the
same MACs (H*W*9*C^2 = 236 M). Factorised layers (no activation inside the chain, as in apply_lrf.py):
  tucker2-in   : 1x1 C->r, 3x3 r->C                    (2 convs)
  tucker2-both : 1x1 C->r, 3x3 r->r, 1x1 r->C          (3 convs)
Each variant is built as its own TensorRT FP32 engine (default TF32 allowed, same as deployment) and timed per
inference with CUDA events, with and without CUDA Graph, over shuffled rounds.

Run: .venv/Scripts/python.exe figures/lrf_latency/microbench.py [--plot-only]
"""
import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / 'microbench.json'
CONFIGS = [(160, 32), (80, 64), (40, 128), (20, 256)]  # (H=W, C)
RANKS = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
KINDS = {'original': None, 'tucker2-in': [(1, 'C', 'r'), (3, 'r', 'C')],
         'tucker2-both': [(1, 'C', 'r'), (3, 'r', 'r'), (1, 'r', 'C')]}
LOGGER = trt.Logger(trt.Logger.ERROR)


def layers_of(kind, c, r):
    if kind == 'original':
        return [(3, c, c)]
    return [(k, c if a == 'C' else r, c if b == 'C' else r) for k, a, b in KINDS[kind]]


def macs(h, convs):
    return sum(h * h * k * k * cin * cout for k, cin, cout in convs)


def build(h, convs):
    builder = trt.Builder(LOGGER)
    net = builder.create_network(0)
    x = net.add_input('x', trt.float32, (1, convs[0][1], h, h))
    rng = np.random.default_rng(0)
    keep = []
    for i, (k, cin, cout) in enumerate(convs):
        w = (rng.standard_normal((cout, cin, k, k)) * (2 / (cin * k * k)) ** 0.5).astype(np.float32)
        # Only the last conv of a chain carries a bias (the fused BN); inner convs have none, as in apply_lrf.py.
        b = rng.standard_normal(cout).astype(np.float32) * 0.01 if i == len(convs) - 1 else None
        keep += [w, b]
        conv = net.add_convolution_nd(x, cout, (k, k), trt.Weights(w), trt.Weights(b) if b is not None else trt.Weights())
        conv.padding_nd = (k // 2, k // 2)
        x = conv.get_output(0)
    sig = net.add_activation(x, trt.ActivationType.SIGMOID)
    y = net.add_elementwise(x, sig.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)
    net.mark_output(y)
    plan = builder.build_serialized_network(net, builder.create_builder_config())
    return trt.Runtime(LOGGER).deserialize_cuda_engine(plan)


class Bench:
    def __init__(self, engine):
        self.engine = engine
        self.ctx = engine.create_execution_context()
        self.bufs = []
        for i in range(engine.num_io_tensors):
            n = engine.get_tensor_name(i)
            t = torch.randn(tuple(engine.get_tensor_shape(n)), device='cuda')
            self.bufs.append(t)
            self.ctx.set_tensor_address(n, t.data_ptr())
        self.stream = torch.cuda.Stream()
        with torch.cuda.stream(self.stream):
            for _ in range(50):
                self.ctx.execute_async_v3(self.stream.cuda_stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        torch.cuda.synchronize()

    def time(self, mode, iters=200):
        """Median per-inference time (µs), synchronising after every inference like the Ultralytics backend."""
        out = []
        with torch.cuda.stream(self.stream):
            for _ in range(20):
                self._one(mode)
            torch.cuda.synchronize()
            for _ in range(iters):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(self.stream)
                self._one(mode)
                b.record(self.stream)
                b.synchronize()
                out.append(a.elapsed_time(b) * 1e3)
        return float(np.median(out))

    def _one(self, mode):
        if mode == 'graph':
            self.graph.replay()
        else:
            self.ctx.execute_async_v3(self.stream.cuda_stream)


def gpu_temp():
    return int(subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
                              capture_output=True, text=True).stdout.split()[0])


def measure(rounds):
    variants = []
    for h, c in CONFIGS:
        variants.append(dict(h=h, c=c, kind='original', r=c))
        for kind in ('tucker2-in', 'tucker2-both'):
            variants += [dict(h=h, c=c, kind=kind, r=r) for r in RANKS if r <= c]
    for v in variants:
        convs = layers_of(v['kind'], v['c'], v['r'])
        v.update(convs=len(convs), macs=macs(v['h'], convs))
        v['bench'] = Bench(build(v['h'], convs))
    print(f'built {len(variants)} engines', flush=True)
    times = {i: {'nograph': [], 'graph': []} for i in range(len(variants))}
    for rnd in range(rounds):
        while gpu_temp() > 65:
            time.sleep(5)
        order = list(range(len(variants)))
        random.Random(rnd).shuffle(order)
        for i in order:
            for mode in ('nograph', 'graph'):
                times[i][mode].append(variants[i]['bench'].time(mode))
        print(f'round {rnd} done, GPU {gpu_temp()} °C', flush=True)
    rows = []
    for i, v in enumerate(variants):
        v.pop('bench')
        rows.append(dict(v, **{f'{m}_us': float(np.median(t)) for m, t in times[i].items()},
                         **{f'{m}_rounds_us': t for m, t in times[i].items()}))
    OUT.write_text(json.dumps(dict(
        method='TensorRT 11.3 FP32 (TF32 allowed), batch 1, per-inference CUDA events with sync, 200 iters, '
               f'median of {rounds} shuffled rounds; GPU <= 65 °C at each round start',
        gpu=torch.cuda.get_device_name(0), rows=rows), indent=2), encoding='utf-8')


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from make_figures import COLOR, INK2, save  # shared style (rcParams applied on import)

    data = json.loads(OUT.read_text(encoding='utf-8'))['rows']
    style = {'tucker2-in': dict(marker='^', ls='-', label='Tucker-2 (in): 2 convs'),
             'tucker2-both': dict(marker='s', ls='-', label='Tucker-2 (both): 3 convs')}
    shade = {'tucker2-in': COLOR['lrf_ch8'], 'tucker2-both': '#6250d6'}
    for mode, title in [('nograph', 'without CUDA Graph'), ('graph', 'with CUDA Graph')]:
        fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.6), sharey=True)
        for ax, (h, c) in zip(axes, CONFIGS):
            base = next(r for r in data if r['h'] == h and r['kind'] == 'original')
            for kind in ('tucker2-in', 'tucker2-both'):
                rows = sorted((r for r in data if r['h'] == h and r['kind'] == kind), key=lambda r: r['r'])
                xs = [r['r'] / c for r in rows]
                ax.plot(xs, [r[f'{mode}_us'] / base[f'{mode}_us'] for r in rows], color=shade[kind],
                        markersize=4, linewidth=1.6, **style[kind])
                ax.plot(xs, [r['macs'] / base['macs'] for r in rows], color=shade[kind], linewidth=1, ls=':',
                        marker=None)
            ax.axhline(1, color=INK2, linewidth=0.8, ls='--')
            ax.set_title(f'{h}×{h}, C={c}\noriginal {base[f"{mode}_us"]:.0f} µs', fontsize=7.5)
            ax.set_xlim(0, 1.05)
            ax.set_xticks([0, 0.5, 1])
            ax.set_ylim(0, None)
        axes[0].set_ylabel('Relative to original 3×3 conv')
        fig.supxlabel('Rank / C', fontsize=8.5, y=0.02)
        handles, labels = axes[0].get_legend_handles_labels()
        from matplotlib.lines import Line2D
        handles += [Line2D([], [], color=INK2, ls=':', lw=1), Line2D([], [], color=INK2, ls='--', lw=0.8)]
        labels += ['MACs ratio (dotted)', 'Original conv = 1']
        fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=7, bbox_to_anchor=(0.5, 1.1))
        fig.text(0.0, -0.08, f'Solid: latency ratio ({title}); dotted: MACs ratio. All originals have equal MACs '
                 '(236 M). TensorRT 11.3 FP32, RTX A2000, batch 1.', fontsize=6.5, color=INK2)
        fig.tight_layout()
        save(fig, f'fig_microbench_{mode}')

    lines = ['| Size | C | Original µs (no graph / graph) | Decomposition | Fastest rank: latency ratio (no graph / graph) | '
             'MACs ratio at that rank |', '|---|---|---|---|---|---|']
    for h, c in CONFIGS:
        base = next(r for r in data if r['h'] == h and r['kind'] == 'original')
        for kind in ('tucker2-in', 'tucker2-both'):
            rows = [r for r in data if r['h'] == h and r['kind'] == kind]
            best = min(rows, key=lambda r: r['nograph_us'])
            lines.append(f"| {h}×{h} | {c} | {base['nograph_us']:.1f} / {base['graph_us']:.1f} | {kind} | "
                         f"r={best['r']}: {best['nograph_us']/base['nograph_us']:.2f} / {best['graph_us']/base['graph_us']:.2f} | "
                         f"{best['macs']/base['macs']:.2f} |")
    (HERE / 'microbench_table.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot-only', action='store_true')
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--wait-for', type=Path, help='start only after this file exists (GPU busy until then)')
    args = ap.parse_args()
    if not args.plot_only:
        if args.wait_for:
            while not args.wait_for.exists():
                time.sleep(10)
            time.sleep(20)
        measure(args.rounds)
    plot()


if __name__ == '__main__':
    main()
