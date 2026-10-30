"""Tables and figures: why LRF + channel pruning cuts FLOPs but not TensorRT latency.

Input: report.json from latency_compare.py (TRT 11.3 FP32, 640x640, batch 1, RTX A2000).
Run:   .venv/Scripts/python.exe figures/lrf_latency/make_figures.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'PTQ'), str(ROOT / 'Decomposition'), str(ROOT / 'Pruning')]

MODELS = {
    'baseline': ROOT / 'models/baseline/8n_float32.pt',
    'lrf_ch8': ROOT / 'models/lrf_both/budget1.7/ch8_b0.2/8n_float32_lrf_ft50_ch8_b0.2_ft10.pt',
}
LABEL = {'baseline': 'Baseline YOLOv8n', 'lrf_ch8': 'LRF + channel pruning'}
COLOR = {'baseline': '#2a78d6', 'lrf_ch8': '#eb6834'}
MARKER = {'baseline': 'o', 'lrf_ch8': '^'}
HATCH = {'baseline': '', 'lrf_ch8': '////'}
# KITTI val mAP50-95, batch 1, conf 0.001, iou 0.7 (PyTorch checkpoints).
MAP = {'baseline': 67.53, 'lrf_ch8': 65.33}
RES = {0: '320', 1: '160', 2: '160', 3: '80', 4: '80', 5: '40', 6: '40', 7: '20', 8: '20', 9: '20', 10: '40', 11: '40',
       12: '40', 13: '80', 14: '80', 15: '80', 16: '40', 17: '40', 18: '40', 19: '20', 20: '20', 21: '20', 22: 'head'}
COMPRESSED = [4, 5, 6, 7, 8, 12, 16, 18, 19, 21, 22]
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 8, 'axes.edgecolor': INK2,
    'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'axes.axisbelow': True, 'legend.frameon': False, 'pdf.fonttype': 42,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'hatch.linewidth': 0.6,
})


def conv_macs(path):
    """MACs of every Conv2d, keyed by module name (BN fused away at export, names unchanged)."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = (ck.get('ema') or ck['model']).float().eval()
    macs, hooks = {}, []
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            def hook(mod, inp, out, name=name):
                macs[name] = out.numel() * mod.in_channels // mod.groups * mod.kernel_size[0] * mod.kernel_size[1]
            hooks.append(mod.register_forward_hook(hook))
    with torch.inference_mode():
        m(torch.zeros(1, 3, 640, 640))
    return macs


def trt_to_module(layer):
    """'/model.21/m.0/cv2/conv/conv.0/Conv' -> 'model.21.m.0.cv2.conv.0'."""
    parts = []
    for seg in layer.strip().strip('/').split('/')[:-1]:
        if parts and seg.startswith(parts[-1] + '.'):
            parts[-1] = seg
        else:
            parts.append(seg)
    return '.'.join(parts)


def kernel_points(report):
    """(MMACs, µs) for every TensorRT convolution kernel; horizontally fused kernels sum their convs."""
    points = {}
    for key, path in MODELS.items():
        macs = conv_macs(path)
        pts, missed = [], 0
        for name, ms in report[key]['trt_layers'].items():
            if 'Reformat' in name or not name.rstrip().endswith('Conv') or ms <= 0:
                continue
            mods = [trt_to_module(p) for p in name.split('||')]
            if not all(m in macs for m in mods):
                missed += 1
                continue
            pts.append((sum(macs[m] for m in mods) / 1e6, ms * 1e3, name))
        points[key] = pts
        print(f'{key}: {len(pts)} conv kernels mapped, {missed} unmapped')
    return points


def stage(report, key, s, field):
    return report[key]['stages'].get(str(s), {}).get(field, 0.0)


def trt_ms(report, key, s):
    return report[key]['trt_stage'].get(str(s), {}).get('ms', 0.0)


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(HERE / f'{name}.{ext}')
    plt.close(fig)


def fig_stages(report):
    stages = [s for s in range(23) if any(stage(report, k, s, 'gmacs') or trt_ms(report, k, s) for k in MODELS)]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 4.4), sharey=True)
    h = 0.38
    for ax, (field, title, scale) in zip(axes, [('gmacs', 'Compute per stage (MMACs)', 1e3),
                                                ('trt', 'TensorRT time per stage (µs)', 1e3)]):
        for j, key in enumerate(MODELS):
            vals = [(trt_ms(report, key, s) if field == 'trt' else stage(report, key, s, 'gmacs')) * scale for s in stages]
            ys = [i + (j - 0.5) * h for i in range(len(stages))]
            ax.barh(ys, vals, height=h - 0.04, color=COLOR[key], hatch=HATCH[key], edgecolor='white',
                    linewidth=0.4, label=LABEL[key])
        ax.set_title(title, loc='left')
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(range(len(stages)))
    def tick(s):
        nb, nl = (int(stage(report, k, s, 'convs')) for k in MODELS)
        convs = f'{nb:>2}→{nl:<2}' if nb or nl else 'upsmp'
        return f"{'*' if s in COMPRESSED else ' '}{s:>2} {RES[s]:>4}  {convs}"
    axes[0].set_yticklabels([tick(s) for s in stages], family='monospace')
    axes[0].invert_yaxis()
    axes[0].set_ylabel('Stage · feature-map size · conv layers (baseline → LRF)')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=2, bbox_to_anchor=(0.55, 1.04))
    fig.text(0.0, -0.04, '* stage modified by low-rank factorisation and/or channel pruning. TensorRT times are '
             'IProfiler per-layer medians (instrumented)\n  of the layers named after each stage; SiLU kernels '
             'generated by Myelin carry no stage name and are excluded.', fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, 'fig_stage_macs_vs_latency')


def fig_kernels(points):
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    for key, pts in points.items():
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=22, marker=MARKER[key], color=COLOR[key],
                   edgecolor='white', linewidth=0.6, label=f'{LABEL[key]} ({len(pts)} kernels)', zorder=3)
    floor = min(p[1] for pts in points.values() for p in pts)
    ax.axhline(floor, color=INK2, linewidth=0.8, linestyle='--', zorder=2)
    ax.text(0.0105, floor - 1.0, f'per-kernel floor ≈ {floor:.0f} µs', fontsize=7.5, color=INK2, va='top')
    ax.set_xscale('log')
    ax.set_xlim(0.01, None)
    ax.set_ylim(0, None)
    ax.set_xlabel('Convolution kernel compute (MMACs, log scale)')
    ax.set_ylabel('TensorRT kernel time (µs)')
    ax.legend(loc='upper left')
    fig.tight_layout()
    save(fig, 'fig_kernel_time_vs_macs')


def groups(report, key):
    """TensorRT time and kernel count by operator type: {name: (kernels, ms)}."""
    out = {name: [0, 0.0] for name in OP_TYPES}
    for n, ms in report[key]['trt_layers'].items():
        a = out[op_type(n)]
        a[0] += 1
        a[1] += ms
    return out


# Myelin runs SiLU as x/(1+exp(-x)) ("NegExpAddDivMul"), optionally fused with the residual Add.
OP_TYPES = ['Convolution', 'SiLU activation\n(+ residual Add)', 'Tensor copies\n(split/concat/resize)',
            'Layout\nreformatting', 'Other\n(maxpool, DFL, decode)']


def op_type(name):
    if name.startswith('Reformatting'):
        return OP_TYPES[3]
    if name.rstrip().endswith('Conv') or '/Conv ||' in name:
        return OP_TYPES[0]
    if 'NegExpAddDivMul' in name or name.startswith('PWN'):
        return OP_TYPES[1]
    if 'copy' in name:
        return OP_TYPES[2]
    return OP_TYPES[4]


def fig_breakdown(report):
    g = {key: groups(report, key) for key in MODELS}
    names = list(g['baseline'])
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    w = 0.36
    for j, key in enumerate(MODELS):
        xs = [i + (j - 0.5) * w for i in range(len(names))]
        vals = [g[key][n][1] for n in names]
        bars = ax.bar(xs, vals, width=w - 0.04, color=COLOR[key], hatch=HATCH[key], edgecolor='white',
                      linewidth=0.4, label=LABEL[key])
        for b, n, v in zip(bars, names, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f'{v:.2f}\n×{g[key][n][0]}', ha='center',
                    va='bottom', fontsize=6.5, color=INK2, linespacing=1.1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel('TensorRT time (ms)')
    ax.set_ylim(0, max(v[1] for k in g for v in g[k].values()) * 1.2)
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper right')
    fig.text(0.0, -0.03, 'Bar labels: summed IProfiler time (ms) and number of TensorRT kernels.', fontsize=7,
             color=INK2)
    fig.tight_layout()
    save(fig, 'fig_latency_breakdown')
    return g


def tables(report, g):
    b, l = report['baseline'], report['lrf_ch8']
    tot = {k: dict(convs=sum(v['convs'] for v in report[k]['stages'].values()),
                   gmacs=sum(v['gmacs'] for v in report[k]['stages'].values())) for k in MODELS}

    def pct(new, old):
        return f'{(new - old) / old * 100:+.1f}%'

    rows1 = [
        ('Parameters (M)', f"{b['params']/1e6:.3f}", f"{l['params']/1e6:.3f}", pct(l['params'], b['params'])),
        ('Compute (GMACs)', f"{tot['baseline']['gmacs']:.2f}", f"{tot['lrf_ch8']['gmacs']:.2f}",
         pct(tot['lrf_ch8']['gmacs'], tot['baseline']['gmacs'])),
        ('Conv2d layers', f"{tot['baseline']['convs']}", f"{tot['lrf_ch8']['convs']}",
         pct(tot['lrf_ch8']['convs'], tot['baseline']['convs'])),
        ('TensorRT layers (after fusion)', f"{b['trt_layer_count']}", f"{l['trt_layer_count']}",
         pct(l['trt_layer_count'], b['trt_layer_count'])),
        ('TensorRT FP32 latency (ms)', f"{b['trt_ms']:.3f}", f"{l['trt_ms']:.3f}", pct(l['trt_ms'], b['trt_ms'])),
        ('PyTorch eager latency (ms)', f"{b['torch_ms']:.3f}", f"{l['torch_ms']:.3f}", pct(l['torch_ms'], b['torch_ms'])),
        ('mAP50-95 (%)', f"{MAP['baseline']:.2f}", f"{MAP['lrf_ch8']:.2f}", f"{MAP['lrf_ch8'] - MAP['baseline']:+.2f} pp"),
    ]

    rows2 = []
    for s in COMPRESSED:
        rows2.append((f'{s}', RES[s], f"{stage(report, 'baseline', s, 'convs'):.0f} → {stage(report, 'lrf_ch8', s, 'convs'):.0f}",
                      f"{stage(report, 'baseline', s, 'gmacs')*1e3:.0f} → {stage(report, 'lrf_ch8', s, 'gmacs')*1e3:.0f}",
                      pct(stage(report, 'lrf_ch8', s, 'gmacs'), stage(report, 'baseline', s, 'gmacs')),
                      f"{trt_ms(report, 'baseline', s)*1e3:.0f} → {trt_ms(report, 'lrf_ch8', s)*1e3:.0f}",
                      pct(trt_ms(report, 'lrf_ch8', s), trt_ms(report, 'baseline', s))))
    sb = lambda k, f: sum(stage(report, k, s, f) for s in COMPRESSED)
    st = lambda k: sum(trt_ms(report, k, s) for s in COMPRESSED)
    rows2.append(('Total', '', f"{sb('baseline','convs'):.0f} → {sb('lrf_ch8','convs'):.0f}",
                  f"{sb('baseline','gmacs')*1e3:.0f} → {sb('lrf_ch8','gmacs')*1e3:.0f}",
                  pct(sb('lrf_ch8', 'gmacs'), sb('baseline', 'gmacs')),
                  f"{st('baseline')*1e3:.0f} → {st('lrf_ch8')*1e3:.0f}", pct(st('lrf_ch8'), st('baseline'))))

    names = list(g['baseline'])
    tb, tl = (sum(v[1] for v in g[k].values()) for k in MODELS)
    nb, nl = (sum(v[0] for v in g[k].values()) for k in MODELS)
    rows3 = [(n.replace('\n', ' '), f"{g['baseline'][n][0]}", f"{g['baseline'][n][1]:.3f}",
              f"{g['baseline'][n][1]/tb*100:.1f}", f"{g['lrf_ch8'][n][0]}", f"{g['lrf_ch8'][n][1]:.3f}",
              f"{g['lrf_ch8'][n][1]/tl*100:.1f}", pct(g['lrf_ch8'][n][1], g['baseline'][n][1])) for n in names]
    rows3.append(('Total (IProfiler sum)', f'{nb}', f'{tb:.3f}', '100.0', f'{nl}', f'{tl:.3f}', '100.0', pct(tl, tb)))

    specs = [
        ('Overall comparison of the baseline and the compressed model (KITTI, 640×640, batch 1, RTX A2000).',
         'tab:lrf_overall', ['Metric', 'Baseline', 'LRF + pruning', 'Change'], 'lrrr', rows1),
        ('Per-stage effect of compression. Conv layers, compute (MMACs) and TensorRT time (µs), baseline → compressed.',
         'tab:lrf_stages', ['Stage', 'Size', 'Conv layers', 'MMACs', 'ΔMACs', 'TRT µs', 'ΔTime'], 'llrrrrr', rows2),
        ('TensorRT time by operator type (IProfiler per-layer medians): kernel count, ms and share of the layer sum.',
         'tab:lrf_breakdown', ['Operator', 'Base kernels', 'Base ms', '%', 'LRF kernels', 'LRF ms', '%', 'ΔTime'],
         'lrrrrrrr', rows3),
    ]
    md, tex = [], []
    for cap, lab, head, align, rows in specs:
        md += [f'**{cap}**', '', '| ' + ' | '.join(head) + ' |', '|' + '|'.join(['---'] * len(head)) + '|']
        md += ['| ' + ' | '.join(r) + ' |' for r in rows] + ['']
        esc = lambda t: (t.replace('%', r'\%').replace('→', r'$\rightarrow$').replace('Δ', r'$\Delta$')
                         .replace('×', r'$\times$').replace('µ', r'$\mu$').replace('–', '--'))
        tex += [r'\begin{table}[t]', r'\centering', rf'\caption{{{esc(cap)}}}', rf'\label{{{lab}}}',
                rf'\begin{{tabular}}{{{align}}}', r'\toprule', ' & '.join(esc(h) for h in head) + r' \\', r'\midrule']
        for r in rows:
            if r[0].startswith('Total'):
                tex.append(r'\midrule')
            tex.append(' & '.join(esc(c) for c in r) + r' \\')
        tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    (HERE / 'tables.md').write_text('\n'.join(md), encoding='utf-8')
    (HERE / 'tables.tex').write_text('% requires \\usepackage{booktabs}\n' + '\n'.join(tex), encoding='utf-8')
    print('\n'.join(md))


def main():
    report = json.loads((HERE / 'report.json').read_text(encoding='utf-8'))
    fig_stages(report)
    fig_kernels(kernel_points(report))
    g = fig_breakdown(report)
    tables(report, g)


if __name__ == '__main__':
    main()
