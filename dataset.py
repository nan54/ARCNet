from __future__ import annotations
import os
import random
import math
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


class NYUDv2Dataset(Dataset):
    IGNORE_LABEL = 255

    def __init__(self, csv_file, root_dir, crop_size=(480, 640), is_train=True):
        """
        🚀 V9.0 终极黄金版：
        1. 采用 CSV 精确寻址，绝对杜绝文件错位。
        2. 彻底写死 1-based 映射，把 7700 万黑边像素 (0) 完美打入冷宫 (255)，拯救 Wall 和 Floor！
        3. 搭载平方根平滑重采样，专治各种罕见死类！
        """
        self.root_dir = os.path.normpath(root_dir)
        self.is_train = is_train
        self.crop_size = crop_size
        self.current_epoch = 0

        # 强制写死 1-based (因为你的新数据 Raw ID 1=Wall, Raw ID 0=黑边)
        self.label_style = "1-based"

        # ========================================================
        # 🚀 1. 读取 CSV 并构建绝对路径列表
        # ========================================================
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"❌ 找不到 CSV 文件: {csv_file}")

        df = pd.read_csv(csv_file)

        # 兼容 Windows 和 Linux 路径
        self.image_paths = [os.path.join(self.root_dir, str(p).replace('\\', os.sep)) for p in df['rgb']]
        self.depth_paths = [os.path.join(self.root_dir, str(p).replace('\\', os.sep)) for p in df['depth']]
        self.seg_paths = [os.path.join(self.root_dir, str(p).replace('\\', os.sep)) for p in df['label']]

        split_name = "TRAIN" if is_train else "VAL"
        print(f"📦 成功通过 CSV 加载 NYUDv2 [{split_name}] 集，共包含 {len(self.image_paths)} 张样本。")

        # ========================================================
        # 🚀 2. 核心绝杀：25个极端困难户（出现率 < 15%）
        # ========================================================
        self.rare_classes = {
            6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
            20, 22, 23, 25, 27, 29, 30, 31, 32, 33,
            34, 36, 37, 38, 39
        }

        self.class_to_indices = {c: [] for c in self.rare_classes}

        if self.is_train:
            print(f"🔍 正在为『25个极端死类』建立专属图库，并计算平滑频率权重...")
            for idx in tqdm(range(len(self.image_paths)), desc="提取类别专属样本", ncols=100):
                seg_path = self.seg_paths[idx]

                if os.path.exists(seg_path):
                    seg_np = np.array(Image.open(seg_path))
                    # 1-based 提取逻辑：先把真值 - 1 再找类别
                    unique_vals = set(np.unique(seg_np))
                    unique_vals = {v - 1 for v in unique_vals if v not in [0, self.IGNORE_LABEL]}

                    for val in self.rare_classes.intersection(unique_vals):
                        self.class_to_indices[val].append(idx)

            # 🛡️ 提取有效类别，并计算【平方根平滑重采样权重】
            self.valid_rare_classes = []
            self.rare_class_weights = []

            for c, indices in self.class_to_indices.items():
                freq = len(indices)
                if freq > 0:
                    self.valid_rare_classes.append(c)
                    weight = 1.0 / math.sqrt(freq)
                    self.rare_class_weights.append(weight)

            print(f"✅ 建池完成！已为 {len(self.valid_rare_classes)} 个罕见类装配【平方根平滑权重保险丝】。")

            self.color_jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            )

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def set_epoch(self, epoch):
        """🚀 同步训练进度，用于触发课程采样衰减"""
        self.current_epoch = epoch

    def _remap_labels(self, seg_tensor):
        seg_remapped = seg_tensor.clone()

        # 🚀 V9 终极映射法则：
        # 1. 抓出所有的 0（背景黑边），标记为需要忽略
        ignore_mask = (seg_remapped == 0) | (seg_remapped == self.IGNORE_LABEL)

        # 2. 把所有的物体类减 1（1->0 Wall, 2->1 Floor, ..., 40->39 otherprop）
        seg_remapped = seg_remapped - 1

        # 3. 将黑边像素全部填成 255
        seg_remapped[ignore_mask] = self.IGNORE_LABEL

        # 4. 越界保护（安全兜底）
        invalid = (seg_remapped < 0) | ((seg_remapped >= 40) & (seg_remapped != self.IGNORE_LABEL))
        seg_remapped[invalid] = self.IGNORE_LABEL

        return seg_remapped

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # ========================================================
        # 🚀 Distribution-Aware Intervention (感知分布平滑重采样)
        # ========================================================
        if self.is_train and hasattr(self, 'valid_rare_classes') and len(self.valid_rare_classes) > 0:
            if self.current_epoch < 35:
                rare_prob = 0.20
            elif self.current_epoch < 50:
                rare_prob = 0.10
            else:
                rare_prob = 0.05

            if rare_prob > 0 and random.random() < rare_prob:
                target_rare_class = random.choices(self.valid_rare_classes, weights=self.rare_class_weights, k=1)[0]
                idx = random.choice(self.class_to_indices[target_rare_class])

        rgb_path = self.image_paths[idx]
        depth_path = self.depth_paths[idx]
        seg_path = self.seg_paths[idx]

        image = Image.open(rgb_path).convert('RGB')
        if self.is_train and random.random() > 0.5:
            image = self.color_jitter(image)
        image_tensor = TF.to_tensor(image)

        seg_np = np.array(Image.open(seg_path))
        seg_tensor = torch.from_numpy(seg_np).long().unsqueeze(0)
        # 转换标签（0变255，1~40变0~39）
        seg_tensor = self._remap_labels(seg_tensor)

        depth_np = np.array(Image.open(depth_path), dtype=np.float32) / 1000.0
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)

        if self.is_train:
            th, tw = self.crop_size
            _, curr_h, curr_w = image_tensor.shape

            scale = random.uniform(0.75, 1.5)
            new_h, new_w = int(curr_h * scale), int(curr_w * scale)

            # 🚀 优化插值算法：
            # 缩小图片时 (scale < 1.0)，使用 AREA 或 BICUBIC 抗锯齿。
            # 放大图片时 (scale > 1.0)，使用 BILINEAR 保持平滑。
            interp_mode = TF.InterpolationMode.BICUBIC if scale < 1.0 else TF.InterpolationMode.BILINEAR

            image_tensor = TF.resize(image_tensor, [new_h, new_w], interpolation=interp_mode)

            # 深度图强烈建议用 BILINEAR，保持物理曲面平滑，绝不能用 NEAREST
            depth_tensor = TF.resize(depth_tensor, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)

            # 分割标签必须用 NEAREST，防止出现不存在的“半个类别” (比如 1.5 类)
            seg_tensor = TF.resize(seg_tensor, [new_h, new_w], interpolation=TF.InterpolationMode.NEAREST)

            depth_tensor = depth_tensor / scale

            pad_h = max(th - new_h, 0)
            pad_w = max(tw - new_w, 0)
            if pad_h > 0 or pad_w > 0:
                image_tensor = TF.pad(image_tensor, (0, 0, pad_w, pad_h), fill=0.0)
                seg_tensor = TF.pad(seg_tensor, (0, 0, pad_w, pad_h), fill=self.IGNORE_LABEL)
                depth_tensor = TF.pad(depth_tensor, (0, 0, pad_w, pad_h), fill=0.0)

            _, padded_h, padded_w = image_tensor.shape

            best_i, best_j = 0, 0
            for _ in range(10):
                i = random.randint(0, padded_h - th)
                j = random.randint(0, padded_w - tw)
                temp_seg = TF.crop(seg_tensor, i, j, th, tw)
                unique_classes = torch.unique(temp_seg)
                valid_classes = unique_classes[unique_classes != self.IGNORE_LABEL]
                best_i, best_j = i, j
                if len(valid_classes) >= 3:
                    break

            image_tensor = TF.crop(image_tensor, best_i, best_j, th, tw)
            seg_tensor = TF.crop(seg_tensor, best_i, best_j, th, tw)
            depth_tensor = TF.crop(depth_tensor, best_i, best_j, th, tw)

            if random.random() > 0.5:
                image_tensor = TF.hflip(image_tensor)
                seg_tensor = TF.hflip(seg_tensor)
                depth_tensor = TF.hflip(depth_tensor)

        else:
            th, tw = self.crop_size
            _, curr_h, curr_w = image_tensor.shape
            if curr_h != th or curr_w != tw:
                image_tensor = TF.center_crop(image_tensor, (th, tw))
                seg_tensor = TF.center_crop(seg_tensor, (th, tw))
                depth_tensor = TF.center_crop(depth_tensor, (th, tw))

        image_tensor = self.normalize(image_tensor)
        seg_tensor = seg_tensor.squeeze(0)

        return {
            'images': image_tensor,
            'segmentation': seg_tensor,
            'depth': depth_tensor
        }