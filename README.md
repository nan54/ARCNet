# ARCNet: An Asymmetric Routing and Cross-Task Collaboration Network for Multi-Task Dense Prediction

本仓库是论文 "An Asymmetric Routing and Cross-Task Collaboration Network for Multi-Task Dense Prediction"的官方 PyTorch 实现。

ARCNet 是一种面向多任务密集预测（语义分割 + 单目深度估计）的端到端轻量级网络框架，遵循“先特征解耦、后跨任务协同（Decoupling Prior to Collaboration）”的设计范式。模型在仅使用 91.73 M 参数量和 261.68 G FLOPs 的条件下，在标准室内场景数据集 NYUD-v2（mIoU 58.59%，RMSE 0.4456 m）与透明物体数据集 **Syn-TODD**（mIoU $94.98\%$，RMSE $0.143\text{m}$）上均取得了优异的联合预测性能。

---

核心特性 (Key Features)

1. 视觉基础模型与多尺度适配器 (`dinov3_adapter.py`)：采用预训练的 DINOv3-ViT-Base 作为特征提取主干，结合轻量级 `SimpleViTAdapter` 构建 1/4, 1/8, 1/16, 1/32四级多尺度特征金字塔。
2. 非对称任务路由器 (`multi_task_routing.py` - ATR)：摒弃复杂的动态退火与手工正则化惩罚，采用固定锐化温度（tau = 0.5）与非对称通道保底下限（alpha = [0.3, 0.3, 0.2, 0.1]），保护梯度稀疏任务的表征容量并抑制任务间恶性竞争。
3. 形变对齐共享解码器与跨任务注意力 (`routed_resenet.py` - DCNv2 & CTA)**：在共享 MLP 解码器中引入可变形卷积（DCNv2）消除多尺度特征边缘错位，并通过通道-空间双门控跨任务注意力（CTA）在解耦特征上恢复受控的双向互补协同。
4. 任务专属预测头 (`routed_resenet.py`)：包含基于度量学习与类别原型的语义锚点查询头（`SemanticAnchoredQueryHead`）以及集成 ASPP 与非负物理约束的增强深度头（`EnhancedDepthHead`）。

---

推荐项目目录结构 (Directory Structure)

在上传或克隆本项目时，建议保持如下目录结构：

```text
ARCNet/
├── dinov3/                          # DINOv3 官方底层依赖模块 (layers, utils 等)
├── pretrain/                        # 存放预训练模型权重
│   └── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
├── data/                            # 数据集根目录
│   └── nyu_official_clean/
│       ├── nyu2_train.csv           # 训练集索引文件
│       ├── nyu2_test.csv            # 测试/验证集索引文件
│       ├── train/                   # 训练集 RGB、Depth、Label 图像
│       └── test/                    # 测试集 RGB、Depth、Label 图像
├── runs/                            # 训练输出日志、权重与可视化结果 (自动生成)
├── config.py                        # 全局路径与训练超参数配置
├── dataset.py                       # 数据集加载、1-based 标签重映射与分布感知平滑重采样
├── dinov3_core.py                   # DINOv3 Vision Transformer 核心定义
├── dinov3_adapter.py                # DINOv3 多尺度主干与 ViT-Adapter 桥接器
├── multi_task_routing.py            # 非对称任务路由器 (ATR)
├── routed_resenet.py                # ARCNet 主网络、共享解码器 (DCNv2)、CTA 与预测头
├── loss.py                          # 多任务损失函数 (wCE + Lovász, SiLog + L1 + Grad, Smooth 等)
├── metrics.py                       # 多任务评估器 (mIoU, PixelAcc, RMSE, REL, δ1/δ2/δ3)
├── visualize_router.py              # 路由器通道激活热力图与分布直方图可视化工具
├── train_resent.py                  # 模型训练、EMA 权重更新、FLOPs/FPS 测速与验证主入口
├── requirements.txt                 # 环境依赖列表
└── README.md                        # 项目说明文档
```

---

## 1. 环境安装 (Environment Setup)

本项目在 Ubuntu 18.04、Python 3.9、PyTorch 2.6.0*以及单张 NVIDIA GeForce RTX 3090 Ti GPU 环境下开发与测试。

### 1.1 创建虚拟环境
```bash
conda create -n arcnet python=3.9 -y
conda activate arcnet
```

### 1.2 安装依赖包
请先根据你的 CUDA 版本安装对应的 PyTorch 2.6.0 与 TorchVision（需支持 `DeformConv2d` 算子），随后安装其余依赖：
```bash
pip install -r requirements.txt
```

---

## 2. 预训练权重与数据集准备 (Preparation)

### 2.1 下载预训练权重
1. 在项目根目录下创建 `./pretrain` 文件夹：
   ```bash
   mkdir -p ./pretrain
   ```
2. 下载 DINOv3-ViT-Base预训练权重 `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`，并将其放置在 `./pretrain/` 目录下（对应 `routed_resenet.py` 中的 `dinov3_weight` 路径）。

 2.2 准备 NYUD-v2 数据集
本项目采用 CSV 索引方式精确读取数据，包含 795 张训练样本和 654 张测试样本：
1. 数据集目录下需包含 `nyu2_train.csv` 和 `nyu2_test.csv` 两个文件，每个 CSV 文件需包含以下三列相对路径表头：
   - `rgb`：RGB 彩色图像路径
   - `depth`：16-bit 单通道深度图路径（单位为毫米，读取时会自动除以 $1000.0$ 转换为米）
   - `label`：语义分割标签路径（采用 `1-based` 编码：像素值 0 为无效黑边，将在训练时自动重映射为忽略标签 255；有效物体类别 1 - 40 将自动映射为 0 - 39）
2. 打开 `config.py`，将 `DATA_ROOT` 修改为你本地的数据集路径：
   ```python
   class Config:
       DATA_ROOT = r'/path/to/your/data/nyu_official_clean'
       SAVE_DIR = './runs/run_sota_from_scratch'
   ```

---

## 3. 模型训练 (Training)

### 3.1 超参数说明
论文实验部分（Section 4.3.1）的默认训练配置已在 `config.py` 和 `train_resent.py` 中预设好：
- **输入裁剪尺寸 (`CROP_SIZE`)**：480 x 640
- **批大小 (`BATCH_SIZE`)**：单卡 Batch Size 为 8，梯度累积步数 `ACCUMULATION_STEPS = 2`（等效 Batch Size 为 16）
- **总训练轮数 (`TOTAL_EPOCHS`)**：100 Epochs
- **分层学习率 (`AdamW`)**：
  - **分层学习率设置（AdamW）**：
  - DINOv3 主干网络：`5e-6`（Weight Decay = `0.05`）
  - 共享解码器：`5e-5`（Weight Decay = `0.05`）
  - 语义与深度预测头：`1e-4`（Weight Decay = `0.05`）
  - 路由器门控参数：`5e-3`（Weight Decay = `0.0`）
- **学习率调度策略**：前 `8` 个 Epoch 采用线性预热（Linear Warm-up），随后采用多项式衰减（PolyLR，幂指数为 `0.9`，最小学习率为 `1e-7`）。

### 3.2 启动训练
在项目根目录下运行以下命令即可启动端到端训练与验证：
```bash
python train_resent.py
```

### 3.3 训练过程输出
运行 `train_resent.py` 后，程序会自动执行以下操作并保存在 `Config.SAVE_DIR`（默认 `./runs/run_sota_from_scratch`）：
1. **基准性能测试**：在训练开始前自动统计总参数量、可训练参数量、计算量（FLOPs）及纯净推理速度（FPS）。
2. **EMA 影子模型验证**：每个 Epoch 结束后，使用指数移动平均（ModelEMA）权重结合水平翻转测试时增强（TTA）与长尾频率校准（`predict_all`）在验证集上评估指标。
3. **权重与日志保存**：
   - `best_model.pth`：验证集上综合表现最佳的 EMA 模型权重。
   - `latest_checkpoint.pth`：最新一轮的完整断点（含主模型、EMA 模型、优化器及混合精度 Scaler 状态）。
   - `training_curves.png`：训练损失曲线及验证集 mIoU / RMSE 变化曲线图。
   - `router_visuals/`：特定 Epoch 下的路由器通道激活热力图与门控分布直方图。

---

## 4. 模型推理与单张图像预测 (Inference)

如果你想加载训练好的 `best_model.pth` 对单张或多张 RGB 图像进行语义分割与深度估计推理，可以参考以下 Python 脚本：

```python
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from routed_resenet import RoutedNYUDv2Net

# 1. 初始化模型并加载最佳权重
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RoutedNYUDv2Net(task_count=2, num_classes=40, align_corners=False).to(device)

checkpoint = torch.load('./runs/run_sota_from_scratch/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. 读取并预处理输入 RGB 图像
img_path = 'sample_indoor.jpg'
image = Image.open(img_path).convert('RGB')
image_tensor = TF.to_tensor(image)
image_tensor = TF.center_crop(image_tensor, (480, 640))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
input_tensor = normalize(image_tensor).unsqueeze(0).to(device)  # [1, 3, 480, 640]

# 3. 执行多任务联合推理 (内置水平翻转概率融合与长尾频率校准)
with torch.no_grad():
    pred_seg_logits, pred_depth = model.predict_all(input_tensor)

# 4. 解析输出结果
pred_seg_mask = torch.argmax(pred_seg_logits, dim=1).squeeze(0).cpu().numpy()  # [480, 640], 类别范围 0~39
pred_depth_map = pred_depth.squeeze().cpu().numpy()                            # [480, 640], 物理深度(单位:米)

print(f"分割预测尺寸: {pred_seg_mask.shape}, 深度预测范围: {pred_depth_map.min():.2f}m ~ {pred_depth_map.max():.2f}m")
```

---

## 5. 实验结果 (Experimental Results)

### 5.1 NYUD-v2 数据集对比结果
| 模型 (Model) | 参数量 Params (M) $\downarrow$ | 计算量 FLOPs (G) $\downarrow$ | 语义分割 Semseg (mIoU, %) $\uparrow$ | 深度估计 Depth (RMSE, m) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: |
| InvPT | 402.00 | 555.57 | 53.56 | 0.5183 |
| MQTransformer | 204.30 | 365.25 | 54.84 | 0.5325 |
| TaskPrompter | 373.00 | 416.00 | 55.30 | 0.5152 |
| MTMamba | 307.99 | 540.81 | 55.82 | 0.5066 |
| MLORE | 407.00 | 571.00 | 55.96 | 0.5080 |
| PAME | - | - | 56.63 | 0.5102 |
| **ARCNet (Ours)** | **91.73** | **261.68** | **58.59** | **0.4456** |

### 5.2 Syn-TODD 透明物体数据集对比结果
| 模型 (Model) | 输入模态 | Depth RMSE $\downarrow$ | Depth SqRel $\downarrow$ | Depth MAE $\downarrow$ | Seg mIoU (%) $\uparrow$ | Seg mAP (%) $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| InvPT | Monocular RGB | 0.167 | 0.146 | 0.163 | 93.19 | 89.63 |
| TaskPrompter | Monocular RGB | 0.245 | 0.231 | 0.245 | 90.52 | 96.91 |
| MVTrans | Monocular RGB | 0.152 | 0.144 | 0.151 | 80.19 | 84.42 |
| **ARCNet (Ours)** | **Monocular RGB** | **0.143** | **0.017** | **0.063** | **94.98** | **98.34** |

---

## 6. 引用 (Citation)

如果本项目的代码或方法对你的研究有所帮助，欢迎引用我们的论文：

```bibtex
@article{xiong2026arcnet,
  title={An Asymmetric Routing and Cross-Task Collaboration Network for Multi-Task Dense Prediction},
  author={Xiong, ZhengNan and Miao, Jun and Xie, ShouPing},
  journal={Nanchang Hangkong University},
  year={2026}
}
```
