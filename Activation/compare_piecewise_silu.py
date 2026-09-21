"""Compare native SiLU with a searched 21-knot piecewise-linear LUT.

Run directly in an IDE, or use --device cpu/mps. --refit reruns the seeded
minimax search. This benchmarks PyTorch eager implementations, not fused C++.
The search is restricted to symmetric interpolation and is not a proof of
global optimality. Full precision coefficients live beside this script.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn


DIRECTORY = Path(__file__).resolve().parent


class PiecewiseSiLU(nn.Module):
    """Coefficient LUT + ten positive linear segments, reflected to negatives."""
    def __init__(self, fit):
        super().__init__()
        self.register_buffer('boundaries', torch.tensor(fit['positive_knots'][1:], dtype=torch.float32))
        # Final entry implements the identity tail for positive inputs.
        self.register_buffer('slopes', torch.tensor(fit['slopes']+[1.], dtype=torch.float32))
        self.register_buffer('intercepts', torch.tensor(fit['intercepts']+[0.], dtype=torch.float32))

    def forward(self, x):
        u = x.abs()
        index = torch.bucketize(u, self.boundaries, right=True)
        y = self.slopes[index]*u + self.intercepts[index]
        return torch.where(x < 0, y+x, y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--threads', type=int, default=10)
    parser.add_argument('--elements', type=int, default=1_000_000)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--minimum', type=float, default=-12.)
    parser.add_argument('--maximum', type=float, default=12.)
    parser.add_argument('--refit', action='store_true')
    args = parser.parse_args()
    if min(args.threads,args.elements,args.repeats) < 1 or args.warmup < 0 or args.maximum <= args.minimum:
        parser.error('Invalid counts or input range')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS unavailable')
    fit_path = DIRECTORY/'piecewise_silu_fit.json'
    if args.refit or not fit_path.exists():
        subprocess.run([sys.executable, str(DIRECTORY/'fit_piecewise_silu.py')], check=True)
    fit = json.loads(fit_path.read_text())
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    native = nn.SiLU().to(args.device)
    approx = PiecewiseSiLU(fit).to(args.device).eval()

    def sync():
        if args.device == 'mps':
            torch.mps.synchronize()

    with torch.inference_mode():
        # Include exact knots and their FP32 neighbors in the accuracy check.
        grid = torch.linspace(args.minimum,args.maximum,200001)
        knots = torch.tensor(fit['positive_knots'], dtype=torch.float32)
        knots = torch.cat([knots,-knots])
        boundary_samples = torch.cat([knots,torch.nextafter(knots,torch.full_like(knots,float('inf'))),
                                      torch.nextafter(knots,torch.full_like(knots,-float('inf')))])
        boundary_samples = boundary_samples[(boundary_samples >= args.minimum)&(boundary_samples <= args.maximum)]
        check = torch.cat([grid,boundary_samples]).to(args.device)
        actual = approx(check).cpu().double()
        reference = torch.nn.functional.silu(check.cpu().double())
        errors = (actual-reference).abs()
        if not torch.isfinite(actual).all():
            raise RuntimeError('Nonfinite approximation output')
        # Benchmark identical uniformly distributed FP32 inputs; no transfers timed.
        x = (torch.rand(args.elements)*(args.maximum-args.minimum)+args.minimum).to(args.device)
        methods = {'native_silu': native, 'piecewise_lut': approx}
        for _ in range(args.warmup):
            for fn in methods.values():
                fn(x)
        sync()
        samples = {name: [] for name in methods}
        for i in range(args.repeats):
            names = list(methods) if i%2 == 0 else list(reversed(methods))
            for name in names:
                sync()
                start = time.perf_counter()
                output = methods[name](x)
                sync()
                samples[name].append((time.perf_counter()-start)*1000)
                del output
        timing = {name: {'median_ms':statistics.median(v),'p10_ms':float(np.percentile(v,10)),
                         'p90_ms':float(np.percentile(v,90))} for name,v in samples.items()}
        result = dict(device=args.device, torch_version=torch.__version__, threads=args.threads,
                      elements=args.elements, repeats=args.repeats, warmup=args.warmup,
                      dtype='float32', input_distribution='uniform', input_range=[args.minimum,args.maximum],
                      implementation='PyTorch eager; MPS synchronized; allocation included',
                      approximation='21 symmetric nonuniform knots; coefficient LUT; 20 linear segments and 0/x tails',
                      qualification=fit['qualification'],
                      grid_mae=float(errors[:len(grid)].mean()),
                      grid_rmse=float(errors[:len(grid)].square().mean().sqrt()),
                      max_error_including_boundaries=float(errors.max()),
                      worst_input=float(check.cpu()[errors.argmax()]),
                      timing=timing,
                      speedup_native_over_lut=timing['native_silu']['median_ms']/timing['piecewise_lut']['median_ms'])
        plot_x=grid[::100].numpy()
        plot_y=actual[:len(grid):100].numpy()
        plot_ref=reference[:len(grid):100].numpy()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,11),layout='constrained')
    axes[0].plot(plot_x,plot_ref,label='Native SiLU',color='black')
    axes[0].plot(plot_x,plot_y,'--',label='21-knot piecewise LUT')
    axes[0].set(title='SiLU vs nonuniform piecewise-linear approximation',ylabel='Output')
    axes[0].legend()
    axes[1].plot(plot_x,np.abs(plot_y-plot_ref))
    axes[1].set(title='Absolute error',xlabel='Input',ylabel='Absolute error')
    labels=list(timing)
    bars=axes[2].bar(labels,[timing[k]['median_ms'] for k in labels])
    axes[2].bar_label(bars,fmt='%.3f ms')
    axes[2].set(title=f'PyTorch eager / {args.device} / {args.elements:,} FP32 elements',ylabel='Median latency (ms)')
    for ax in axes[:2]:
        ax.grid(alpha=.25)
    stem=DIRECTORY/f'piecewise_silu_comparison_{args.device}'
    fig.savefig(stem.with_suffix('.png'),dpi=160)
    plt.close(fig)
    stem.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    print(f'Plot: {stem.with_suffix(".png")}')


if __name__ == '__main__':
    main()
