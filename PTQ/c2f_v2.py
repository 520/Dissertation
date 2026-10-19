"""C2f with its cv1 split into two convolutions, so each half can be pruned independently.

Ultralytics C2f does cv1(x).chunk(2, 1), which needs an even, equal split; after pruning the two halves
generally differ in width. C2f_v2 computes the halves with cv0 and cv1 and is numerically identical before
pruning. Pruned checkpoints pickle this class: keep this file importable (PTQ on sys.path) to load them.
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from ultralytics.nn.modules.block import C2f


def _slice_conv(conv, rows):
    """Copy of an Ultralytics Conv (conv + BN + act) keeping only output channels `rows`."""
    out = copy.deepcopy(conv)
    out.conv.weight = nn.Parameter(conv.conv.weight.data[rows].clone())
    out.conv.out_channels = out.conv.weight.shape[0]
    bn = out.bn
    bn.weight = nn.Parameter(conv.bn.weight.data[rows].clone())
    bn.bias = nn.Parameter(conv.bn.bias.data[rows].clone())
    bn.running_mean = conv.bn.running_mean[rows].clone()
    bn.running_var = conv.bn.running_var[rows].clone()
    bn.num_features = bn.weight.shape[0]
    return out


class C2f_v2(nn.Module):
    def __init__(self, c2f: C2f):
        super().__init__()
        c = c2f.c
        self.cv0 = _slice_conv(c2f.cv1, slice(0, c))
        self.cv1 = _slice_conv(c2f.cv1, slice(c, 2 * c))
        self.cv2, self.m = c2f.cv2, c2f.m
        for attr in ('f', 'i', 'type', 'np'):           # Ultralytics graph bookkeeping
            if hasattr(c2f, attr):
                setattr(self, attr, getattr(c2f, attr))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def replace_c2f(model: nn.Module) -> dict[str, tuple[str, slice]]:
    """Swap every C2f for C2f_v2 in place. Returns {new conv name: (original conv name, channel slice)}."""
    mapping = {}
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, C2f) and not isinstance(child, C2f_v2):
                full = f'{name}.{child_name}' if name else child_name
                c = child.c
                setattr(module, child_name, C2f_v2(child))
                mapping[f'{full}.cv0'] = (f'{full}.cv1', slice(0, c))
                mapping[f'{full}.cv1'] = (f'{full}.cv1', slice(c, 2 * c))
    return mapping
