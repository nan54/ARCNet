from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricTaskRouter(nn.Module):
    """
    非对称渐进式任务路由器 (Asymmetric Progressive Task Router)
    🚀 V8.15.1 修复版：消除 Padding 僵尸通道，强化内存连续性。
    """

    def __init__(self, in_channels, task_count=2, alpha=0.0, name="router"):
        super().__init__()
        self.name = name
        self.task_count = task_count
        self.in_channels = in_channels
        self.alpha = alpha

        self.mask_logits = nn.Parameter(torch.zeros(task_count, in_channels))
        nn.init.normal_(self.mask_logits, std=0.01)

    def _get_tau(self, current_epoch):

        return 0.5

    def forward(self, x, current_epoch=100):
        b, c, h, w = x.shape
        tau = self._get_tau(current_epoch)

        if self.mask_logits.shape[1] < c:
            raise ValueError(f"[{self.name}] 致命通道不匹配！输入为 {c}，但路由器初始化为 {self.in_channels}！")

        logits_seg = self.mask_logits[0][:c]
        logits_depth = self.mask_logits[1][:c]

        gate_seg_raw = torch.sigmoid(logits_seg / tau)
        gate_depth_raw = torch.sigmoid(logits_depth / tau)


        min_gate_seg = self.alpha
        min_gate_dep = self.alpha
        # min_gate_dep = min(0.5, self.alpha + 0.15) # 深度任务享有更高的特权保底
        # min_gate_dep = min(0.6, self.alpha + 0.30)

        # 应用不同的下限
        final_gate_seg = min_gate_seg + (1.0 - min_gate_seg) * gate_seg_raw
        final_gate_depth = min_gate_dep + (1.0 - min_gate_dep) * gate_depth_raw

        final_gate_seg = final_gate_seg.reshape(1, c, 1, 1)
        final_gate_depth = final_gate_depth.reshape(1, c, 1, 1)

        return x * final_gate_seg, x * final_gate_depth






