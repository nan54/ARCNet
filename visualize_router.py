import os
import torch
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from multi_task_routing import AsymmetricTaskRouter


def plot_router_visualizations(model, epoch, save_dir, target_epochs=[0, 49, 99]):
    """
    📊 论文级可视化：绘制路由器通道的掩码分布与热图
    - 图A：掩码分布直方图 (展示极化过程)
    - 图B：通道抢占热图 (展示任务解耦)
    """
    if epoch not in target_epochs:
        return

    os.makedirs(save_dir, exist_ok=True)
    module_to_parse = model.module if hasattr(model, 'module') else model

    # 寻找所有的 AsymmetricTaskRouter，我们挑最深层的一个（比如 enc_router4，特征最抽象，极化最明显）
    target_router = None
    for name, module in module_to_parse.named_modules():
        if isinstance(module, AsymmetricTaskRouter) and 'enc_router4' in name:
            target_router = module
            break

    if target_router is None:
        return  # 消融实验中如果没有 AsymmetricTaskRouter 则跳过

    tau = target_router._get_tau(epoch)

    # 获取经过 Sigmoid 与 Tau 缩放后的最终 Gate 激活值 [0, 1]
    gate_seg = torch.sigmoid(target_router.mask_logits[0] / tau).detach().cpu().numpy()
    gate_depth = torch.sigmoid(target_router.mask_logits[1] / tau).detach().cpu().numpy()

    # ========================================================
    # 📉 图 A: 训练过程中路由器掩码分布的直方图演变
    # 预期效果：Epoch 1 全堆积在 0.5 附近，Epoch 100 分别堆积在 0 和 1
    # ========================================================
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(gate_seg, bins=40, color='blue', alpha=0.7, range=(0, 1))
    plt.title(f'Segmentation Gate Dist (Epoch {epoch + 1})', fontsize=14)
    plt.xlabel('Activation Value', fontsize=12)
    plt.ylabel('Channel Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.hist(gate_depth, bins=40, color='green', alpha=0.7, range=(0, 1))
    plt.title(f'Depth Gate Dist (Epoch {epoch + 1})', fontsize=14)
    plt.xlabel('Activation Value', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    hist_path = os.path.join(save_dir, f'router_hist_epoch_{epoch + 1}.png')
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ========================================================
    # 🗺️ 图 B: 分割与深度通道抢占热图 (Heatmap)
    # 预期效果：上下两行呈现明显的“互补”或“错位”现象
    # ========================================================
    heatmap_data = np.vstack([gate_seg, gate_depth])

    # 如果通道太多(如512)，只取前 100 个通道展示以保证图片美观性
    display_channels = min(100, len(gate_seg))
    heatmap_data_subset = heatmap_data[:, :display_channels]

    plt.figure(figsize=(18, 3.5))
    ax = sns.heatmap(
        heatmap_data_subset,
        cmap="coolwarm",  # 用冷暖色调凸显 0(蓝) 到 1(红)
        cbar=True,
        yticklabels=['Seg Task', 'Depth Task'],
        vmin=0.0, vmax=1.0,
        linewidths=0.5
    )
    plt.title(f'Channel Activation Heatmap (Epoch {epoch + 1}, First {display_channels} Channels)', fontsize=16)
    plt.xlabel('Channel Index', fontsize=12)

    plt.tight_layout()
    heatmap_path = os.path.join(save_dir, f'router_heatmap_epoch_{epoch + 1}.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📸 已生成 Epoch {epoch + 1} 的路由器状态可视化至: {save_dir}")