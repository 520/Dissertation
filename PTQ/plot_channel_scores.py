"""Dissertation figures from channel_importance.py results (PDF + PNG).

fig1_layer_scores      per-layer L1 / L2 / Taylor-1 in network order
fig2_cumulative        Taylor-1 removed vs channels pruned (criterion x allocation)
fig3_overlap           overlap of the channel sets selected by L1, L2, Taylor-1
fig4_taylor_spread     per-layer distribution of channel Taylor-1 scores
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed metric -> colour mapping, used identically in every figure.
COLOR = {'l1': '#2a78d6', 'l2': '#eb6834', 'taylor1': '#1baf7a'}
LABEL = {'l1': 'L1', 'l2': 'L2', 'taylor1': 'Taylor-1'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.dpi': 300, 'pdf.fonttype': 42,
})


def save(fig, out, name):
    for ext in ('pdf', 'png'):
        fig.savefig(out/f'{name}.{ext}')
    plt.close(fig)
    print(out/f'{name}.pdf')


def short(name):
    return name.removesuffix('.conv')


def fig_layer_scores(channels, out):
    g = channels.groupby(['layer_index', 'layer'])
    layers = pd.DataFrame({'l1': g.l1_mean_abs.mean(), 'l2': g.l2_rms.mean(),
                           'taylor1': g.taylor1.mean()}).reset_index()
    y = np.arange(len(layers))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 9.0), sharey=True)
    titles = {'l1': 'L1 (mean |w| per weight)', 'l2': 'L2 (RMS per weight)',
              'taylor1': 'Taylor-1 (mean per channel)'}
    fixed = layers.layer.str.contains('.dfl.', regex=False).to_numpy()
    for ax, metric in zip(axes, ('l1', 'l2', 'taylor1')):
        # The fixed DFL projection is not a learned filter: excluded from every panel.
        values = layers[metric].where(~fixed)
        ax.barh(y, values.fillna(0), height=0.7, color=COLOR[metric])
        ax.set_xscale('log')
        ax.set_xlim(values.min()/2, values.max()*1.5)
        for yi in y[fixed]:
            ax.text(values.min()/2, yi, ' n/a (fixed DFL)', va='center', fontsize=6, color=MUTED)
        ax.set_title(titles[metric], loc='left')
        ax.set_ylim(len(layers)-0.5, -0.5)
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(y, [f'{i:02d} {short(n)}' for i, n in zip(layers.layer_index, layers.layer)], fontsize=6)
    axes[0].set_ylabel('Convolution (network order)')
    fig.suptitle('Per-layer channel importance, YOLOv8n on KITTI (log scale)', x=0.02, y=0.995, ha='left', fontsize=11)
    fig.tight_layout()
    save(fig, out, 'fig1_layer_scores')


def removed_curve(p, metric, within, fractions):
    total = p.taylor1.sum()
    shares = []
    for f in fractions:
        if within:
            cut = sum(grp.nsmallest(int(len(grp)*f), metric).taylor1.sum() for _, grp in p.groupby('layer'))
        else:
            cut = p.nsmallest(int(len(p)*f), metric).taylor1.sum()
        shares.append(100*cut/total)
    return shares


def fig_cumulative(p, out):
    fractions = np.linspace(0, 0.6, 25)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    style = {False: ('-', 'o', 'global ranking'), True: ('--', 's', 'uniform per layer')}
    ends = []
    for metric in ('l1', 'taylor1'):
        for within in (False, True):
            ls, mk, alloc = style[within]
            share = removed_curve(p, metric, within, fractions)
            ax.plot(100*fractions, share, ls, color=COLOR[metric], lw=2, marker=mk, markersize=4,
                    markevery=4, label=f'{LABEL[metric]}, {alloc}')
            ends.append(share[-1])
    ax.set_xlabel('Channels pruned (%)')
    ax.set_ylabel('Taylor-1 importance removed (%)')
    ax.set_xlim(0, 60)
    ax.set_ylim(0, max(ends)*1.05)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title('Importance removed as pruning ratio grows', loc='left')
    save(fig, out, 'fig2_cumulative')


def fig_overlap(p, out):
    fractions = (0.2, 0.3, 0.5)
    pairs = (('l1', 'l2'), ('l1', 'taylor1'), ('l2', 'taylor1'))
    def chosen(metric, f):
        return set(i for _, grp in p.groupby('layer') for i in grp.nsmallest(int(len(grp)*f), metric).index)
    x = np.arange(len(fractions))
    width = 0.26
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    # Magnitude-vs-magnitude in grey; the two magnitude-vs-gradient pairs in the Taylor colour, told apart by hatch.
    looks = (('#8a8984', ''), (COLOR['taylor1'], ''), (COLOR['taylor1'], '////'))
    for k, (a, b) in enumerate(pairs):
        values = [100*len(chosen(a, f) & chosen(b, f))/len(chosen(a, f)) for f in fractions]
        color, hatch = looks[k]
        bars = ax.bar(x + (k-1)*(width+0.02), values, width, color=color, hatch=hatch,
                      edgecolor='white', linewidth=0.5, label=f'{LABEL[a]} vs {LABEL[b]}')
        ax.bar_label(bars, fmt='%.0f', fontsize=7, color=INK, padding=1)
    ax.set_xticks(x, [f'{int(f*100)}%' for f in fractions])
    ax.set_xlabel('Within-layer pruning ratio')
    ax.set_ylabel('Shared pruned channels (%)')
    ax.set_ylim(0, 105)
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=8, ncol=3)
    ax.set_title('Agreement between pruning criteria', loc='left')
    save(fig, out, 'fig3_overlap')


def fig_taylor_spread(p, out):
    order = p.groupby('layer').layer_index.first().sort_values()
    data = [p.loc[p.layer == name, 'taylor1'].clip(lower=1e-9).to_numpy() for name in order.index]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    box = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=True,
                     flierprops={'marker': '.', 'markersize': 2, 'markerfacecolor': MUTED, 'markeredgecolor': 'none'},
                     medianprops={'color': INK, 'linewidth': 1}, whiskerprops={'color': MUTED},
                     capprops={'color': MUTED})
    for patch in box['boxes']:
        patch.set(facecolor=COLOR['taylor1'], edgecolor=COLOR['taylor1'], alpha=0.85)
    ax.set_yscale('log')
    ax.set_xticks(np.arange(1, len(order)+1), [f'{i:02d}' for i in order.values], fontsize=6, rotation=90)
    ax.set_xlabel('Convolution index (network order; names in Fig. 1)')
    ax.set_ylabel('Channel Taylor-1 score (log)')
    ax.grid(axis='x', visible=False)
    ax.set_title('Spread of channel Taylor-1 scores within each layer', loc='left')
    save(fig, out, 'fig4_taylor_spread')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path, help='output directory of channel_importance.py')
    args = parser.parse_args()
    out = args.results/'figures'
    out.mkdir(exist_ok=True)
    channels = pd.read_csv(args.results/'channels.csv', encoding='utf-8-sig')
    prunable = channels[channels.pruning_candidate & channels.taylor1.notna()]
    fig_layer_scores(channels, out)
    fig_cumulative(prunable, out)
    fig_overlap(prunable, out)
    fig_taylor_spread(prunable, out)


if __name__ == '__main__':
    main()
