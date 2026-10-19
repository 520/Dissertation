"""Heatmap of the per-group L2 pruning sensitivity (outputs/sensitivity_l2/sensitivity.csv).

Rows: the 59 DepGraph groups, most sensitive (largest mean mAP50-95 drop) at the top. Columns: pruning ratio.
Cell: mAP50-95 drop in points against the unpruned adaptive-BN reference; negative drops (subset noise) show as 0.

Example:
    .venv/Scripts/python.exe -m Pruning.plot_sensitivity
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

ROOT = Path(__file__).resolve().parents[1]
BLUES = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']   # sequential 100 -> 700
INK, MUTED = '#1f1f1e', '#6b6a66'


def merge(paths, merged_path):
    """Rows of several scans with identical settings; a (group, ratio) present in more than one keeps the first
    file's row, and the largest disagreement between duplicates is reported as a reproducibility check."""
    rows, seen, differences = [], {}, []
    for path in paths:
        for r in csv.DictReader(path.open(encoding='utf-8-sig')):
            key = (int(r['group']), float(r['ratio']))
            if key in seen:
                differences.append(abs(float(r['mAP50-95']) - float(seen[key]['mAP50-95'])))
                continue
            seen[key] = r
            rows.append(r)
    rows.sort(key=lambda r: (int(r['group']), float(r['ratio'])))
    if len(paths) > 1:
        with merged_path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'{merged_path}: {len(rows)} rows')
        if differences:
            print(f'{len(differences)} duplicate (group, ratio) points: max |mAP50-95 difference| = '
                  f'{100 * max(differences):.2f} points, mean {100 * np.mean(differences):.3f}')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, nargs='+', default=[ROOT/'outputs/sensitivity_l2/sensitivity.csv'],
                        help='one or more sensitivity.csv files from runs with the same settings')
    parser.add_argument('--name', default='sensitivity_heatmap', help='output file stem, next to the first input')
    args = parser.parse_args()
    out_dir = args.input[0].parent
    rows = merge(args.input, out_dir/f'{args.name}.csv')
    ratios = sorted({float(r['ratio']) for r in rows})
    groups = sorted({int(r['group']) for r in rows})
    drop = np.full((len(groups), len(ratios)), np.nan)
    label = {}
    for r in rows:
        g, k = groups.index(int(r['group'])), ratios.index(float(r['ratio']))
        drop[g, k] = 100 * float(r['drop_mAP50-95'])
        coupled = r['coupled_out_convs'].count(' + ')
        name = r['root'].removesuffix('.conv') + (f' (+{coupled} coupled)' if coupled else '')
        label[g] = f"G{int(r['group']):02d}  {name}  [{r['channels']}]"
    order = np.argsort(-np.nanmean(drop, axis=1))
    data = np.clip(drop[order], 0, None)

    cmap = LinearSegmentedColormap.from_list('seq_blue', BLUES)
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=float(np.nanmax(data)))
    fig, ax = plt.subplots(figsize=(4.5 + 0.55 * len(ratios), 17), dpi=200)
    image = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')
    for i in range(len(groups)):
        for k in range(len(ratios)):
            value = data[i, k]
            ax.text(k, i, f'{value:.1f}', ha='center', va='center', fontsize=6.5,
                    color='white' if norm(value) > 0.55 else INK)
    ax.set_xticks(range(len(ratios)), [f'{100 * r:g}%' for r in ratios], fontsize=7.5, color=INK)
    ax.set_yticks(range(len(groups)), [label[g] for g in order], fontsize=7, color=INK, family='monospace')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Channels removed from the group', fontsize=9, color=INK, labelpad=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(ratios)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(groups)), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)      # 2px-style surface gap between cells
    ax.tick_params(which='minor', length=0)

    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, aspect=40)
    bar.set_label('mAP50-95 drop (points)', fontsize=8, color=INK)
    bar.ax.tick_params(labelsize=7, colors=MUTED, length=0)
    bar.outline.set_visible(False)
    fig.suptitle('YOLOv8n-KITTI pruning sensitivity per dependency group', fontsize=11, color=INK, x=0.02,
                 ha='left', y=0.995)
    fig.text(0.02, 0.982, 'L2 group importance, one group pruned at a time, adaptive BN (512 train images), '
             '500 val images.\nReference 68.59 mAP50-95. Sorted by mean drop; [n] = channels in the group; '
             'negative drops (noise) shown as 0.', fontsize=7.5, color=MUTED, ha='left', va='top')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for suffix in ('png', 'pdf'):
        path = out_dir/f'{args.name}.{suffix}'
        fig.savefig(path, facecolor='white')
        print(path)


if __name__ == '__main__':
    main()
