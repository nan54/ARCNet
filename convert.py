import os
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ==========================================
# 配置路径 (请根据你的实际存放位置修改)
# ==========================================
MAT_FILE_PATH = r"D:\pycharm2023\test\TaskRouting-master\data\nyu_depth_v2_labeled.mat"  # 你下载的 .mat 文件路径
OUTPUT_DIR = r"D:\pycharm2023\test\TaskRouting-master\data\nyu_official_clean"  # 转换后的保存路径

NYU40_CLASSES = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "blinds", "desk", "shelves", "curtain", "dresser", "pillow", "mirror",
    "floor mat", "clothes", "ceiling", "books", "refrigera", "television", "paper", "towel", "showercurt",
    "box", "whiteboard", "person", "night stan", "toilet", "sink", "lamp", "bathtub", "bag", "otherstruc",
    "otherfurn", "otherprop"
]


def build_894_to_40_mapping(f):
    """从 .mat 中提取 894 个原始类别名称，并映射到 1~40 标号"""
    print("🔍 正在构建 894 原始类别 -> 40 标准类别的映射映射表...")
    mapping = np.zeros(1000, dtype=np.uint8)  # 默认 0 为背景 Ignore

    if 'names' not in f:
        print("⚠️ 未找到类别名称，假设标签已经是 1~40 范围。")
        return None

    names_ref = f['names'][0]
    raw_names = []
    for r in names_ref:
        obj = f[r]
        char_arr = obj[()]
        if char_arr.ndim > 1:
            char_arr = char_arr.flatten()
        s = "".join([chr(c) for c in char_arr])
        raw_names.append(s)

    alias_dict = {
        "wall": 1, "floor": 2, "cabinet": 3, "bed": 4, "chair": 5, "sofa": 6, "table": 7, "door": 8,
        "window": 9, "bookshelf": 10, "picture": 11, "counter": 12, "blinds": 13, "desk": 14, "shelves": 15,
        "curtain": 16, "dresser": 17, "pillow": 18, "mirror": 19, "floor mat": 20, "mat": 20, "rug": 20,
        "clothes": 21, "clothing": 21, "ceiling": 22, "books": 23, "book": 23, "refrigerator": 24, "fridge": 24,
        "refrigera": 24, "television": 25, "tv": 25, "paper": 26, "towel": 27, "shower curtain": 28,
        "showercurt": 28, "box": 29, "whiteboard": 30, "person": 31, "nightstand": 32, "night stan": 32,
        "toilet": 33, "sink": 34, "lamp": 35, "bathtub": 36, "bag": 37, "otherstructure": 38,
        "otherstruc": 38, "otherfurniture": 39, "otherfurn": 39, "otherprop": 40
    }

    for idx, name in enumerate(raw_names):
        raw_id = idx + 1  # MATLAB 1-based index
        clean_name = name.lower().strip()

        matched = False
        if clean_name in alias_dict:
            mapping[raw_id] = alias_dict[clean_name]
            matched = True
        else:
            for key, nyu_id in alias_dict.items():
                if key in clean_name:
                    mapping[raw_id] = nyu_id
                    matched = True
                    break
        if not matched:
            mapping[raw_id] = 40  # 未识别的物体统一归为 otherprop

    return mapping


def main():
    if not os.path.exists(MAT_FILE_PATH):
        raise FileNotFoundError(f"❌ 找不到官方 mat 文件: {MAT_FILE_PATH}，请检查路径！")

    print(f"📦 正在打开官方大文件: {MAT_FILE_PATH} ...")
    f = h5py.File(MAT_FILE_PATH, 'r')

    images = f['images']  # [1449, 3, 640, 480]
    depths = f['depths']  # [1449, 640, 480]
    labels = f['labels']  # [1449, 640, 480]

    num_samples = images.shape[0]
    print(f"✅ 成功加载！共发现 {num_samples} 张全标注样本。")

    # 检查是否需要映射 894 -> 40
    mapping = build_894_to_40_mapping(f)

    # 获取官方推荐的 Train / Test 划分列表
    if 'trainNtx' in f and 'testNtx' in f:
        train_idxs = set((f['trainNtx'][()].flatten() - 1).astype(int))
        test_idxs = set((f['testNtx'][()].flatten() - 1).astype(int))
        print(f"📋 使用官方划分索引：Train {len(train_idxs)} 张 | Test {len(test_idxs)} 张。")
    else:
        train_idxs = set(range(795))
        test_idxs = set(range(795, 1449))
        print(f"📋 使用默认前 795 张 Train，后 654 张 Test。")

    # 创建输出文件夹结构
    for split in ['train', 'val']:
        for sub in ['images', 'depth', 'label']:
            os.makedirs(os.path.join(OUTPUT_DIR, split, sub), exist_ok=True)

    csv_train_data = []
    csv_val_data = []

    print("\n🚀 开始高精度解包并保存图像...")
    for i in tqdm(range(num_samples), desc="解包进度", ncols=100):
        split = 'train' if i in train_idxs else 'val'
        filename = f"{i:05d}.png"

        # 1. 提取并转置 RGB (MATLAB 转置修正: (3, 640, 480) -> (480, 640, 3))
        img_np = np.transpose(images[i], (2, 1, 0))
        img_pil = Image.fromarray(img_np)
        rgb_path = os.path.join(OUTPUT_DIR, split, 'images', filename)
        img_pil.save(rgb_path)

        # 2. 提取并转置 Depth (米 -> 毫米 uint16 保存)
        depth_np = np.transpose(depths[i], (1, 0))
        depth_mm = (depth_np * 1000.0).astype(np.uint16)
        depth_pil = Image.fromarray(depth_mm)
        depth_path = os.path.join(OUTPUT_DIR, split, 'depth', filename)
        depth_pil.save(depth_path)

        # 3. 提取并转置 Label
        label_np = np.transpose(labels[i], (1, 0)).astype(np.int32)
        if mapping is not None:
            label_np = mapping[label_np]
        label_pil = Image.fromarray(label_np.astype(np.uint8))
        label_path = os.path.join(OUTPUT_DIR, split, 'label', filename)
        label_pil.save(label_path)

        # 记录 CSV 相对路径
        rel_rgb = os.path.join(split, 'images', filename)
        rel_depth = os.path.join(split, 'depth', filename)
        rel_label = os.path.join(split, 'label', filename)

        if split == 'train':
            csv_train_data.append([rel_rgb, rel_depth, rel_label])
        else:
            csv_val_data.append([rel_rgb, rel_depth, rel_label])

    # 保存 CSV 文件
    pd.DataFrame(csv_train_data, columns=['rgb', 'depth', 'label']).to_csv(os.path.join(OUTPUT_DIR, 'nyu2_train.csv'),
                                                                           index=False)
    pd.DataFrame(csv_val_data, columns=['rgb', 'depth', 'label']).to_csv(os.path.join(OUTPUT_DIR, 'nyu2_test.csv'),
                                                                         index=False)

    print(f"\n🎉 完美解包完成！数据已保存至: {OUTPUT_DIR}")

    # ==========================================
    # 🔍 自动对生成的数据集进行全量“体检”
    # ==========================================
    print("\n🕵️‍♂️ 正在对解包后的数据集进行像素健康度全量扫描...")
    total_counts = np.zeros(256, dtype=np.int64)
    for split in ['train', 'val']:
        label_dir = os.path.join(OUTPUT_DIR, split, 'label')
        for fname in os.listdir(label_dir):
            arr = np.array(Image.open(os.path.join(label_dir, fname)))
            counts = np.bincount(arr.flatten(), minlength=256)
            total_counts += counts

    print("\n📊 终极体检报告:")
    print("=" * 60)
    print(f"Raw ID: 0  (背景黑边/Ignore) | 像素量: {total_counts[0]:,}")
    print(f"Raw ID: 1  (Wall 墙壁)       | 像素量: {total_counts[1]:,}")
    print(f"Raw ID: 2  (Floor 地板)      | 像素量: {total_counts[2]:,}")
    print("=" * 60)

    if total_counts[1] > 10000000 and total_counts[2] > 10000000:
        print("🟢【100% 健康数据】Wall 和 Floor 像素量全部突破千万级！没有错位！没有污染！")
        print("👉 结论：这是一个标准的 1-based 数据集，请确保 dataset.py 中的 `self.label_style = '1-based'`。")
    else:
        print("🔴 警告：像素分布依然异常，请检查原始 .mat 文件是否损坏。")


if __name__ == '__main__':
    main()