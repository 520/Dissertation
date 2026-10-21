"""Plots of the per-layer Tucker-2 sensitivity (outputs/sensitivity_tucker2/sensitivity.csv).

sensitivity_heatmap  rows: (layer, mode), most sensitive (largest mean drop) at the top; columns: rank / channels
                     at the eighths 12.5 % ... 100 % (a 16-channel mode only has 50 % and 100 %, blank elsewhere)
sensitivity_lines    one panel per layer, mAP50-95 drop against rank at every step-8 rank, modes 'out' and 'in'
The drop is the 'drop' column: raw, or the smaller of raw and adaptive-BN when calibration ran.
Negative drops (subset noise) show as 0.

Example:
    .venv/Scripts/python.exe -m Decomposition.plot_tucker2_sensitivity
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

ROOT = Path(__file__).resolve().parents[1]
BLUES = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']   # sequential 100 -> 700
INK, MUTED, EMPTY = '#1f1f1e', '#6b6a66', '#f1f0ec'
MODE_COLOR = {'out': '#256abf', 'in': '#d9730d'}
EIGHTHS = [k / 8 for k in range(1, 9)]


def heatmap(rows, out_dir, reference):
    keys = list(dict.fromkeys((r['layer'], r['mode']) for r in rows))
    drop = np.full((len(keys), len(EIGHTHS)), np.nan)
    mean = np.zeros(len(keys))
    for i, key in enumerate(keys):
        mine = [r for r in rows if (r['layer'], r['mode']) == key]
        mean[i] = np.mean([float(r['drop']) for r in mine])
        for r in mine:
            ratio = float(r['ratio'])
            if any(math.isclose(ratio, e) for e in EIGHTHS):
                drop[i, [math.isclose(ratio, e) for e in EIGHTHS].index(True)] = 100 * float(r['drop'])
    order = np.argsort(-mean)
    data = np.clip(drop[order], 0, None)
    labels = []
    for i in order:
        r = next(r for r in rows if (r['layer'], r['mode']) == keys[i])
        labels.append(f"{r['layer']:<17} {r['mode']:<3} [{r['c_in']}->{r['c_out']}{' s2' if r['stride'] == '2' else ''}]")

    cmap = LinearSegmentedColormap.from_list('seq_blue', BLUES)
    cmap.set_bad(EMPTY)
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=float(np.nanmax(data)))
    fig, ax = plt.subplots(figsize=(9, 0.21 * len(keys) + 2), dpi=200)
    image = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, norm=norm, aspect='auto')
    for i in range(len(keys)):
        for k in range(len(EIGHTHS)):
            if not np.isnan(data[i, k]):
                ax.text(k, i, f'{data[i, k]:.1f}', ha='center', va='center', fontsize=6.5,
                        color='white' if norm(data[i, k]) > 0.55 else INK)
    ax.set_xticks(range(len(EIGHTHS)), [f'{100 * e:g}%' for e in EIGHTHS], fontsize=7.5, color=INK)
    ax.set_yticks(range(len(keys)), labels, fontsize=6.5, color=INK, family='monospace')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Rank kept in the decomposed mode (rank / channels)', fontsize=9, color=INK, labelpad=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(EIGHTHS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(keys)), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', length=0)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, aspect=40)
    bar.set_label('mAP50-95 drop (points)', fontsize=8, color=INK)
    bar.ax.tick_params(labelsize=7, colors=MUTED, length=0)
    bar.outline.set_visible(False)
    fig.suptitle('YOLOv8n-KITTI Tucker-2 sensitivity per layer and mode', fontsize=11, color=INK, x=0.02,
                 ha='left', y=0.995)
    fig.text(0.02, 0.985 - 0.4 / fig.get_figheight(),
             f"One k x k conv decomposed at a time, other mode at full rank; 500 val images; adaptive BN when the raw "
             f"drop > 0.2 points.\nReference {100 * reference:.2f} mAP50-95. Sorted by mean drop over all step-8 ranks; "
             f"negative drops shown as 0.", fontsize=7.5, color=MUTED, ha='left', va='top')
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.9 / fig.get_figheight()))
    save(fig, out_dir/'sensitivity_heatmap')


def lines(rows, out_dir):
    layers = list(dict.fromkeys(r['layer'] for r in rows))
    cols = 6
    fig, axes = plt.subplots(math.ceil(len(layers) / cols), cols, figsize=(16, 2.1 * math.ceil(len(layers) / cols)),
                             dpi=200, squeeze=False)
    top = max(100 * float(r['drop']) for r in rows)
    for ax, layer in zip(axes.flat, layers):
        mine = [r for r in rows if r['layer'] == layer]
        for mode, color in MODE_COLOR.items():
            pts = sorted((int(r['rank']), max(0.0, 100 * float(r['drop']))) for r in mine if r['mode'] == mode)
            if pts:
                ax.plot(*zip(*pts), color=color, linewidth=1.2, marker='o', markersize=2, label=mode)
        ax.set_yscale('symlog', linthresh=0.5)
        ax.set_ylim(0, top * 1.3)
        ax.set_title(f"{layer} [{mine[0]['c_in']}->{mine[0]['c_out']}]", fontsize=7.5, color=INK, loc='left')
        ax.tick_params(labelsize=6, colors=MUTED, length=2)
        ax.grid(color=EMPTY, linewidth=0.8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    for ax in list(axes.flat)[len(layers):]:
        ax.axis('off')
    handles = [plt.Line2D([], [], color=c, linewidth=1.2, marker='o', markersize=2, label=m) for m, c in MODE_COLOR.items()]
    axes.flat[0].legend(handles=handles, fontsize=6.5, frameon=False, title='mode', title_fontsize=6.5)
    fig.supxlabel('rank kept (step 8)', fontsize=9, color=INK)
    fig.supylabel('mAP50-95 drop (points, symlog)', fontsize=9, color=INK)
    fig.suptitle('YOLOv8n-KITTI Tucker-2 sensitivity: drop against rank, per layer', fontsize=11, color=INK,
                 x=0.01, ha='left')
    fig.tight_layout()
    save(fig, out_dir/'sensitivity_lines')


def save(fig, stem):
    for suffix in ('png', 'pdf'):
        fig.savefig(stem.with_suffix(f'.{suffix}'), facecolor='white')
        print(stem.with_suffix(f'.{suffix}'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT/'outputs/sensitivity_tucker2/sensitivity.csv')
    args = parser.parse_args()
    rows = list(csv.DictReader(args.input.open(encoding='utf-8-sig')))
    manifest = args.input.parent/'manifest.json'
    reference = json.loads(manifest.read_text())['original_trained_bn']['mAP50-95'] if manifest.exists() else float('nan')
    heatmap(rows, args.input.parent, reference)
    lines(rows, args.input.parent)


if __name__ == '__main__':
    main()
