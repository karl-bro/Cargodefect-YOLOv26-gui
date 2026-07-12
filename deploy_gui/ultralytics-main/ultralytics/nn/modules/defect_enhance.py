# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""DefectEnhanceBlock: lightweight defect texture enhancement with ECA attention."""

import torch
import torch.nn as nn
import math


class ECAAttention(nn.Module):
    """Efficient Channel Attention with adaptive kernel size."""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        kernel_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class DefectEnhanceBlock(nn.Module):
    """
    Lightweight defect texture enhancement block.
    Structure: 1x1 Conv reduce → 3x3 DWConv → ECA → 1x1 Conv restore → residual
    """

    def __init__(self, c1: int, reduction: int = 4):
        super().__init__()
        c_hidden = max(c1 // reduction, 16)

        self.reduce = nn.Conv2d(c1, c_hidden, 1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(c_hidden)
        self.act_reduce = nn.SiLU(inplace=True)

        self.dwconv = nn.Conv2d(c_hidden, c_hidden, 3, padding=1,
                                 groups=c_hidden, bias=False)
        self.bn_dw = nn.BatchNorm2d(c_hidden)
        self.act_dw = nn.SiLU(inplace=True)

        self.eca = ECAAttention(c_hidden)

        self.restore = nn.Conv2d(c_hidden, c1, 1, bias=False)
        self.bn_restore = nn.BatchNorm2d(c1)
        self.act_out = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.act_reduce(self.bn_reduce(self.reduce(x)))
        out = self.act_dw(self.bn_dw(self.dwconv(out)))
        out = self.eca(out)
        out = self.bn_restore(self.restore(out))
        return self.act_out(out + identity)
