"""Line charts of the Taylor-1 group sensitivity scan (taylor_sensitivity.py): mAP50-95 drop vs. pruning ratio.

Small multiples, one panel per model block (Detect split by scale), all on one shared y-axis so panels compare
directly. One line per Torch-Pruning group; a group's role (cv0, cv1 / residual, m.k.cv1, cv2) keeps the same
colour in every panel.
Outputs next to sensitivity.csv: sensitivity_lines.png / .pdf
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# Reference categorical palette, light mode, slots 1-5 in their validated order.
SLOT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
BLOCK_KIND = {0: 'Conv', 1: 'Conv', 2: 'C2f', 3: 'Conv', 4: 'C2f', 5: 'Conv', 6: 'C2f', 7: 'Conv', 8: 'C2f',
              9: 'SPPF', 12: 'C2f', 15: 'C2f', 16: 'Conv', 18: 'C2f', 19: 'Conv', 21: 'C2f'}
SCALE = {0: 'P3/8', 1: 'P4/16', 2: 'P5/32'}


def role(root, members):
    """(colour slot, legend label) of a group from its root and coupled members."""
    name = re.sub(r'^model\.\d+\.|\.conv$', '', root)
    if len(members.split()) > 1:                                   # residual-coupled C2f group
        return 1, 'cv1 + m.*.cv2 (residual)'
    m = re.match(r'(cv[23])\.(\d)\.(\d)$', name)                  # Detect: cv2 = box, cv3 = class
    if m:
        branch = 'box' if m.group(1) == 'cv2' else 'cls'
        return (0 if branch == 'box' else 2) + int(m.group(3)), f'{branch} {m.group(1)}.{m.group(2)}.{m.group(3)}'
    if re.fullmatch(r'm\.(\d)\.cv1', name):
        return 2 + int(name[2]), name
    if name == 'm.0.cv2':                                          # neck C2f (no shortcut, n=1): m.1 slot is free
        return 3, name
    if re.fullmatch(r'm\.0\.cv2', name):                        # neck C2f (no shortcut, n=1): m.1 slot is free
        return 3, name
    return {'cv0': 0, 'cv1': 1, 'cv2': 4}.get(name, 0), name


def panel_of(root):
    block = int(root.split('.')[1])
    if block == 22:
        scale = int(re.search(r'cv[23]\.(\d)', root).group(1))
        return (22, scale), f'model.22 Detect {SCALE[scale]}'
    stage = 'backbone' if block <= 9 else 'neck'
    return (block, 0), f'model.{block} {BLOCK_KIND.get(block, "")} ({stage})'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT/'sensitivity_results/kitti_taylor1_group_sensitivity')
    args = parser.parse_args()
    with (args.results/'sensitivity.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    manifest = json.loads((args.results/'manifest.json').read_text(encoding='utf-8'))
    ratios = sorted({float(r['ratio']) for r in rows})

    series = {}
    for r in rows:
        key = (r['root'], r['members'], int(r['channels']))
        series.setdefault(key, {})[float(r['ratio'])] = float(r['drop_mAP50-95'])
    panels = {}
    for (root, members, channels), points in series.items():
        key, title = panel_of(root)
        panels.setdefault(key, [title, []])[1].append((role(root, members), channels, points))
    keys = sorted(panels)
    cols = 5
    rows_n = -(-len(keys) // cols)
    ymax = max(max(p.values()) for *_, p in (s for _, ss in panels.values() for s in ss))
    ymin = min(0.0, min(min(p.values()) for *_, p in (s for _, ss in panels.values() for s in ss)))

    plt.rcParams.update({'font.size': 7, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.titlecolor': INK, 'font.family': 'DejaVu Sans'})
    fig, axes = plt.subplots(rows_n, cols, figsize=(2.6 * cols, 2.15 * rows_n), sharex=True, sharey=True,
                             squeeze=False)
    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)
    for ax, key in zip(axes.flat, keys):
        title, lines = panels[key]
        ax.set_title(title, fontsize=7.5, loc='left', pad=4)
        ax.axhline(0, color=INK2, lw=0.6, zorder=1)
        ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for (slot, label), channels, points in sorted(lines, key=lambda s: s[0][0]):
            xs = [q for q in ratios if q in points]
            ax.plot([100 * q for q in xs], [points[q] for q in xs], color=SLOT[slot], lw=1.5, marker='o', ms=3.2,
                    mec='white', mew=0.6, label=f'{label} ({channels})', zorder=3)
        ax.legend(loc='upper left', fontsize=5.6, frameon=False, handlelength=1.6, borderaxespad=0.3,
                  labelcolor=INK2)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=2, width=0.6)
    ax0 = axes[0, 0]
    ax0.set_ylim(ymin - 1, ymax * 1.08 + 1)
    ax0.set_xticks([10, 25, 50, 75, 90])
    ax0.set_xlim(5, 95)
    for c in range(cols):                                          # lowest visible panel of each column
        visible = [ax for ax in axes[:, c] if ax.get_visible()]
        if visible:
            visible[-1].xaxis.set_tick_params(labelbottom=True)
            visible[-1].set_xlabel('channels pruned in group (%)')
    for ax in axes[:, 0]:
        ax.set_ylabel('mAP50-95 drop (pp)')
    done = len(rows)
    total = len(ratios) * manifest['groups']
    ref = manifest['reference_unpruned_adaptive_bn']['mAP50-95']
    status = '' if done >= total else f'  [partial: {done}/{total} trials]'
    top = 1 - 0.55 / fig.get_figheight()                          # title block: fixed height in inches
    fig.text(0.01, 1 - 0.12 / fig.get_figheight(),
             'Per-group pruning sensitivity, Taylor-1 ranking, adaptive BN, no fine-tuning' + status,
             fontsize=9, color=INK, va='top')
    fig.text(0.01, 1 - 0.33 / fig.get_figheight(), f'One group pruned at a time. Drop vs. unpruned + adaptive BN (mAP50-95 {ref:.3f}) on '
             f'{manifest["val_subset"]["images"]} KITTI val images. Legend: group (channels).',
             fontsize=6.5, color=INK2, va='top')
    fig.tight_layout(rect=(0, 0, 1, top))
    for ext in ('png', 'pdf'):
        fig.savefig(args.results/f'sensitivity_lines.{ext}', dpi=220, facecolor='white')
    print(f'Saved {args.results/"sensitivity_lines.png"} ({done}/{total} trials)')


if __name__ == '__main__':
    main()
