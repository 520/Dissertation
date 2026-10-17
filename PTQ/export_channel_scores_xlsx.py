"""Export channel_importance.py CSV results to one Excel workbook.

Sheets: 说明, 层排序, 通道_全部, 通道排序_L1/L2/Taylor1, and one sheet per layer
with its channels sorted by Taylor-1 (within-layer ranks for all three metrics).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CHANNEL_COLUMNS = ['layer_index', 'layer', 'channel', 'role', 'pruning_candidate', 'kernel',
                   'in_channels', 'weights_per_channel',
                   'l1', 'l2', 'taylor1', 'taylor1_weight_only', 'l1_mean_abs', 'l2_rms',
                   'rank_within_layer_l1', 'rank_within_layer_l2', 'rank_within_layer_taylor1',
                   'rank_global_l1', 'rank_global_l2', 'rank_global_taylor1',
                   'taylor_status', 'onnx_node', 'engine_layers']
LAYER_COLUMNS = ['layer_index', 'layer', 'out_channels', 'role',
                 'l1', 'l2', 'taylor1', 'rank_layer_l1', 'rank_layer_l2', 'rank_layer_taylor1',
                 'mean_channel_l1', 'mean_channel_l2', 'mean_channel_taylor1',
                 'min_channel_taylor1', 'max_channel_taylor1', 'gradient_images']


def autosize(sheet, frame):
    from openpyxl.utils import get_column_letter
    for index, column in enumerate(frame.columns, 1):
        width = max([len(str(column))] + [len(str(v)) for v in frame[column].head(200)])
        sheet.column_dimensions[get_column_letter(index)].width = min(max(width + 2, 8), 45)
    sheet.freeze_panes = 'A2'
    sheet.auto_filter.ref = sheet.dimensions


def write(writer, name, frame):
    frame.to_excel(writer, sheet_name=name, index=False)
    autosize(writer.sheets[name], frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path, help='output directory of channel_importance.py')
    parser.add_argument('--xlsx', type=Path, default=None)
    args = parser.parse_args()
    xlsx = args.xlsx or args.results/'channel_scores.xlsx'
    channels = pd.read_csv(args.results/'channels.csv', encoding='utf-8-sig')[CHANNEL_COLUMNS]
    layers = pd.read_csv(args.results/'layers.csv', encoding='utf-8-sig')
    layers = layers[[c for c in LAYER_COLUMNS if c in layers.columns]]
    manifest = json.loads((args.results/'manifest.json').read_text(encoding='utf-8'))

    notes = pd.DataFrame({'项目': [
        '模型', '状态', '图像数', '卷积层数', '通道数', 'L1', 'L2', 'Taylor-1', '排名', '注意'], '说明': [
        manifest['model'], manifest['status'],
        f"{manifest['processed_images']}/{manifest['total_training_images']} (KITTI train, 640, batch 1, FP32)",
        len(layers), len(channels),
        'Σ|W[c,:,:,:]|，BN 融合后卷积权重（不含 bias）；l1_mean_abs = L1 / 每通道权重数',
        '√ΣW[c,:,:,:]²；l2_rms = L2 / √每通道权重数',
        '全部图像平均 |Σ_hw A[c,h,w]·∂L/∂A[c,h,w]|，A 为激活前卷积输出，L = 7.5·box + 0.5·cls + 1.5·dfl',
        '降序，1 = 最重要；同分共享名次。rank_within_layer_* 为层内排名，rank_global_* 为全网排名',
        'DFL 固定投影层 (model.22.dfl.conv) 不在 loss 图中，Taylor 为空；跨层比较原始 L1/L2 受卷积核大小影响，建议用 l1_mean_abs / l2_rms 或层内排名',
    ]})
    with pd.ExcelWriter(xlsx, engine='openpyxl') as writer:
        write(writer, '说明', notes)
        write(writer, '层排序', layers.sort_values('rank_layer_taylor1', na_position='last'))
        write(writer, '通道_全部', channels.sort_values(['layer_index', 'channel']))
        for metric, title in (('l1', 'L1'), ('l2', 'L2'), ('taylor1', 'Taylor1')):
            ranked = channels.dropna(subset=[metric]).sort_values(f'rank_global_{metric}')
            write(writer, f'通道排序_{title}', ranked)
        for (_, name), frame in channels.groupby(['layer_index', 'layer'], sort=True):
            frame = frame.sort_values(['rank_within_layer_taylor1', 'channel'], na_position='last')
            write(writer, name[:31], frame.drop(columns=['onnx_node', 'engine_layers']))
    print(f'{xlsx}  ({len(layers)} layers, {len(channels)} channels)')


if __name__ == '__main__':
    main()
