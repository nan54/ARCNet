from __future__ import annotations
import os
import torch


class Config:
    # ==========================================
    # 基础路径设置
    # ==========================================
    DATA_ROOT = r'/home/ubuntu/桌面/1/model_1/data/nyu_official_clean'
    SAVE_DIR = './runs/run_sota_from_scratch'

    # ==========================================
    # 训练超参数 (从头训练的标准设定)
    # ⚙️ 训练超参数
    # ==========================================
    CROP_SIZE = (480, 640)
    BATCH_SIZE =8# BS=2 时，网络中已使用 GroupNorm 替代 BatchNorm
    ACCUMULATION_STEPS = 2  # 等效 Batch Size = 16
    NUM_WORKERS = 4

    TOTAL_EPOCHS = 200

    # 基础学习率（主优化器：Backbone 5e-5, Decoder 1e-4；动态权重优化器独立为 2e-4）
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01

    # ==========================================
    # 🖥️ 环境设置
    # ==========================================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42

    @classmethod
    def make_dirs(cls):
        os.makedirs(cls.SAVE_DIR, exist_ok=True)