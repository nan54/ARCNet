import os
import math
import random
import warnings
import matplotlib
import numpy as np
import gc
import copy  # 🚀 1. 导入 copy 模块用于 EMA 影子模型深拷贝

warnings.filterwarnings("ignore", category=UserWarning)

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from visualize_router import plot_router_visualizations
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import time
from thop import profile, clever_format
from dataset import NYUDv2Dataset
from routed_resenet import RoutedNYUDv2Net
from metrics import MultiTaskEvaluator
from config import Config

from loss import (
    AdvancedSegmentationLoss,
    AdvancedGeometricDepthLoss,
    DynamicMultiTaskLoss,
    PresenceLoss,
    SemanticGuidedDepthSmoothLoss
)



if hasattr(RoutedNYUDv2Net, 'forward_mtl'):
    RoutedNYUDv2Net.forward = RoutedNYUDv2Net.forward_mtl


# ==========================================
# 🚀 2. 稳盘神器：ModelEMA (加入动态预热引擎)
# ==========================================
class ModelEMA:
    """ EMA 动态预热版：前期快速跟进主网络，后期死死稳住全局极值点 """
    def __init__(self, model, decay=0.999):
        # 兼容 DataParallel，剥离出底层单卡模型进行深拷贝
        self.ema_model = copy.deepcopy(model.module if hasattr(model, 'module') else model)
        self.ema_model.eval()
        self.decay = decay
        self.step = 0  # 🚀 增加计步器
        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        # 🚀 绝杀修复：动态 Decay！
        # 第 1 步 decay 是 0.18，让 EMA 瞬间抄袭主模型。
        # 随着 step 增加，decay 会平滑上升，直到永远锁定在 0.999
        current_decay = min(self.decay, (1 + self.step) / (10 + self.step))

        msd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        for k, ema_v in self.ema_model.state_dict().items():
            if ema_v.dtype.is_floating_point:
                # 使用 current_decay 而不是死板的 self.decay
                ema_v.copy_(ema_v * current_decay + (1. - current_decay) * msd[k].detach())


# 🚀 提分外挂：极其平滑的 PolyLR
class PolyLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, min_lr=1e-7, last_epoch=-1):
        self.max_iters = max_iters
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]


def plot_metrics(history, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = history['epoch']
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Total Loss', color='purple')
    plt.plot(epochs, history['seg_loss'], label='Seg Loss', color='blue', linestyle='--')
    plt.plot(epochs, history['depth_loss'], label='Depth Loss', color='green', linestyle='--')
    plt.title('Training Losses Over Epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['val_miou'], label='Val mIoU (EMA)', color='red', linewidth=2)
    plt.title('Validation Segmentation mIoU (EMA)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['val_rmse'], label='Val RMSE (EMA)', color='darkgreen', linewidth=2)
    plt.title('Validation Depth RMSE (EMA)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def build_optimizers(model, dynamic_loss):
    backbone_params = []
    seg_head_params = []
    depth_head_params = []
    decoder_params = []
    router_params = []  # 🚀 1. 新增：专门存放路由器的掩码参数

    module_to_parse = model.module if hasattr(model, 'module') else model

    for name, param in module_to_parse.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        elif 'mask_logits' in name:  # 🚀 2. 抓取所有动态路由器的关键参数
            router_params.append(param)
        elif any(k in name for k in ['head_seg', 'seg_adapter', 'aux_head', 'cross_task_align']):
            seg_head_params.append(param)
        elif any(k in name for k in ['head_depth', 'depth_adapter', 'uni_guidance', 'feature_edge']):
            depth_head_params.append(param)
        else:
            decoder_params.append(param)

    # 🛡️ 保护 DINOv3 主干，降低 LR 至 5e-6，提升 Weight Decay 至 0.05
    opt_main = optim.AdamW([
        {'params': backbone_params, 'lr': 5e-6, 'weight_decay': 0.05},
        {'params': decoder_params, 'lr': 5e-5, 'weight_decay': 0.05},
        {'params': seg_head_params, 'lr': 1e-4, 'weight_decay': 0.05},
        {'params': depth_head_params, 'lr': 1e-4, 'weight_decay': 0.05},
        # 🚀 3. 路由器特权：高学习率 (5e-3) 快速寻优，彻底免除权重衰减 (0.0)！
        {'params': router_params, 'lr': 5e-3, 'weight_decay': 0.0}
    ])
    opt_dyn = optim.AdamW(dynamic_loss.parameters(), lr=2e-4, weight_decay=0.0)
    return opt_main, opt_dyn


def print_model_parameters(model):
    """🖨️ 打印模型参数量统计与基准测速"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 50)
    print("📊 模型参数量统计")
    print("=" * 50)
    print(f"🔹 总参数量 (Total):      {total_params / 1e6:.2f} M")
    print(f"🔸 可训练参数 (Trainable): {trainable_params / 1e6:.2f} M")
    print(f"🔒 冻结参数 (Frozen):     {(total_params - trainable_params) / 1e6:.2f} M")

    # ========================================================
    # 🚀 论文级性能基准测试 (FLOPs & FPS)
    # 严谨做法：在训练前用 Dummy Tensor 进行标准测速，排除 DataLoader 干扰
    # ========================================================
    print("\n" + "=" * 50)
    print("🏎️ 开始进行论文级性能基准测试 (FLOPs & FPS)")
    print("=" * 50)

    bench_model = model.module if hasattr(model, 'module') else model
    bench_model.eval()  # 切换到评估模式测速
    dummy_input = torch.randn(1, 3, Config.CROP_SIZE[0], Config.CROP_SIZE[1]).to(Config.DEVICE)

    # 1. 计算 FLOPs (MACs * 2 ≈ FLOPs)
    try:
        # 只测前向传播网络本身，传入输入和 current_epoch=0
        macs, params = profile(bench_model, inputs=(dummy_input, 0), verbose=False)
        flops, params_str = clever_format([macs * 2, params], "%.3f")
        print(f"🔹 理论计算量 (FLOPs) : {flops}")
        print(f"🔹 实际参数量 (Params): {params_str}")
    except Exception as e:
        print(f"⚠️ FLOPs 计算失败: {e}")

    # 2. 计算标准纯净 FPS
    with torch.no_grad():
        with autocast('cuda'):
            print("⏳ 正在预热 GPU 并测速，请稍候...")
            # 2.1 Warmup 10次 (唤醒 GPU 时钟频率，论文测速必备)
            for _ in range(10):
                _ = bench_model.forward_mtl(dummy_input, current_epoch=0)

            # 2.2 正式测速 100次
            torch.cuda.synchronize()  # 确保之前的计算全部完成
            start_time = time.time()
            for _ in range(100):
                _ = bench_model.forward_mtl(dummy_input, current_epoch=0)
            torch.cuda.synchronize()  # 确保 100 次计算全部完成
            end_time = time.time()

            avg_time = (end_time - start_time) / 100.0
            pure_fps = 1.0 / avg_time
            print(f"⚡ 纯净推理速度 (FPS) : {pure_fps:.2f} 帧/秒 (分辨率: {Config.CROP_SIZE})")

    print("=" * 50 + "\n")
    bench_model.train()  # 测完记得切回训练模式，不要影响后面的训练！

def main():
    Config.make_dirs()
    Config.TOTAL_EPOCHS = 100
    print(f"📂 日志将保存在: {Config.SAVE_DIR}")

    torch.manual_seed(Config.SEED)
    torch.cuda.manual_seed_all(Config.SEED)
    torch.backends.cudnn.benchmark = True

    train_csv = os.path.join(Config.DATA_ROOT, 'nyu2_train.csv')
    val_csv = os.path.join(Config.DATA_ROOT, 'nyu2_test.csv')

    train_dataset = NYUDv2Dataset(
        csv_file=train_csv,
        root_dir=Config.DATA_ROOT,
        crop_size=Config.CROP_SIZE,
        is_train=True
    )

    val_dataset = NYUDv2Dataset(
        csv_file=val_csv,
        root_dir=Config.DATA_ROOT,
        crop_size=Config.CROP_SIZE,
        is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=True)

    model = RoutedNYUDv2Net(task_count=2, num_classes=40, align_corners=False).to(Config.DEVICE)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # 🖨️ 打印模型参数量
    print_model_parameters(model)

    # 🚀 3. 实例化 EMA 影子更新器
    ema_updater = ModelEMA(model, decay=0.999)

    criterion_seg = AdvancedSegmentationLoss(ignore_index=255).to(Config.DEVICE)
    criterion_depth = AdvancedGeometricDepthLoss(l1_weight=1.0).to(Config.DEVICE)
    dynamic_loss_weighting = DynamicMultiTaskLoss(num_tasks=2).to(Config.DEVICE)
    smooth_loss_fn = SemanticGuidedDepthSmoothLoss().to(Config.DEVICE)

    optimizer, optimizer_dyn = build_optimizers(model, dynamic_loss_weighting)

    warmup_epochs = 8
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)

    poly_scheduler = PolyLR(
        optimizer,
        max_iters=Config.TOTAL_EPOCHS - warmup_epochs,
        power=0.9,
        min_lr=1e-7
    )

    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, poly_scheduler],
                                                milestones=[warmup_epochs])

    scaler = GradScaler('cuda')

    history = {'epoch': [], 'train_loss': [], 'seg_loss': [], 'depth_loss': [], 'val_miou': [], 'val_rmse': []}
    best_miou = 0.0
    best_rmse = float('inf')
    accum_steps = Config.ACCUMULATION_STEPS

    print(f"\n==========================================================================")
    print(f"🔥 V24 终极收官版: 强化 EMA 稳盘 + 验证集激活 predict_all + ClassMix 贯穿！")
    print(f"==========================================================================\n")

    # NYUDv2 的 25 个罕见类
    rare_classes = [6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 22, 23, 25, 27, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39]

    for epoch in range(Config.TOTAL_EPOCHS):
        train_dataset.set_epoch(epoch)
        model.train()
        epoch_loss, epoch_seg, epoch_depth = 0.0, 0.0, 0.0
        num_batches = len(train_loader)

        optimizer.zero_grad()
        optimizer_dyn.zero_grad()

        if epoch >= 70:
            for param_group in optimizer_dyn.param_groups:
                param_group['lr'] = 1e-4

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{Config.TOTAL_EPOCHS}] Train", ncols=125,
                    dynamic_ncols=False)

        for i, batch in enumerate(pbar):
            images = batch['images'].to(Config.DEVICE)
            targets_seg = batch['segmentation'].to(Config.DEVICE)
            targets_depth = batch['depth'].to(Config.DEVICE)

            paste_mask = torch.zeros_like(targets_seg, dtype=torch.bool)



            with autocast('cuda'):
                outputs = model(images, current_epoch=epoch, target_seg=targets_seg)

                if isinstance(outputs[0], tuple):
                    preds_seg = outputs[0][0]
                    aux_pred = outputs[0][1]
                else:
                    preds_seg = outputs[0]
                    aux_pred = None

                preds_depth = outputs[1]

                seg_loss_dict = criterion_seg(preds_seg, targets_seg, epoch=epoch)
                raw_loss_seg = seg_loss_dict["loss"]

                if aux_pred is not None:
                    loss_aux = F.cross_entropy(aux_pred, targets_seg, ignore_index=255)
                    raw_loss_seg = raw_loss_seg + 0.4 * loss_aux

                raw_loss_depth = criterion_depth(preds_depth, targets_depth, epoch=epoch, paste_mask=paste_mask)

                dummy_loss_ctx = torch.tensor(0.0, device=images.device, requires_grad=True)
                loss_mtl, w_seg, w_dep, w_ctx = dynamic_loss_weighting(
                    raw_loss_seg, raw_loss_depth, dummy_loss_ctx, epoch=epoch
                )

                module_to_parse = model.module if hasattr(model, 'module') else model

                valid_mask = (targets_depth > 0.001) & (targets_depth < 10.0)
                if paste_mask.any():
                    valid_mask = valid_mask & (~paste_mask.unsqueeze(1))

                if valid_mask.any():
                    pred_mean_clamped = torch.clamp(preds_depth[valid_mask].mean(), min=1e-6)
                    target_mean_clamped = torch.clamp(targets_depth[valid_mask].mean(), min=1e-6)
                    loss_mean = torch.abs(torch.log(pred_mean_clamped) - torch.log(target_mean_clamped.detach()))
                else:
                    loss_mean = torch.tensor(0.0, device=images.device)

                loss_smooth = smooth_loss_fn(preds_depth, targets_seg, paste_mask=paste_mask) * 0.1

                decay_factor = max(0.0, 1.0 - epoch / Config.TOTAL_EPOCHS)

                overlap_weight = (0.01 if epoch < 20 else 0.05) * decay_factor



                loss_query_div = torch.tensor(0.0, device=images.device)
                if hasattr(module_to_parse, 'head_seg') and hasattr(module_to_parse.head_seg,
                                                                    'get_query_diversity_loss'):
                    loss_query_div = module_to_parse.head_seg.get_query_diversity_loss(epoch=epoch) * decay_factor

                loss_adapter_reg = torch.tensor(0.0, device=images.device)
                for name, param in module_to_parse.named_parameters():
                    if 'adapter.scale' in name:
                        loss_adapter_reg += (param - 0.0) ** 2
                loss_adapter_reg = loss_adapter_reg * (0.3 * decay_factor)

                total_loss = loss_mtl + loss_adapter_reg + loss_query_div + 0.05 * loss_mean + loss_smooth

            scaler.scale(total_loss / accum_steps).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == num_batches:
                scaler.unscale_(optimizer)
                scaler.unscale_(optimizer_dyn)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(dynamic_loss_weighting.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.step(optimizer_dyn)
                scaler.update()

                # 🚀 4. 在每次参数更新后，同步平滑更新 EMA 影子模型的权重
                ema_updater.update(model)

                optimizer.zero_grad()
                optimizer_dyn.zero_grad()

            epoch_loss += total_loss.item()
            epoch_seg += raw_loss_seg.item()
            epoch_depth += raw_loss_depth.item()

            with torch.no_grad():
                if valid_mask.any():
                    depth_mu = preds_depth[valid_mask].mean().item()
                    gt_mu = targets_depth[valid_mask].mean().item()
                else:
                    depth_mu, gt_mu = 0.0, 0.0
                ratio = (depth_mu / gt_mu) if gt_mu > 0 else 0.0

            pbar.set_postfix({
                'L': f"{total_loss.item():.2f}",
                'S': f"{raw_loss_seg.item():.2f}",
                'Smth': f"{loss_smooth.item():.3f}",
                'D_μ': f"{depth_mu:.2f}",
                'Ratio': f"{ratio:.2f}"
            })

        scheduler.step()

        avg_loss = epoch_loss / num_batches
        avg_seg = epoch_seg / num_batches
        avg_depth = epoch_depth / num_batches
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"\n📢 Epoch [{epoch + 1}/{Config.TOTAL_EPOCHS}] 训练总结 (LR: {current_lr:.6e}):\n"
            f"   Total Loss: {avg_loss:.4f} | Seg Loss: {avg_seg:.4f} | Dep Loss: {avg_depth:.4f}\n"
        )

        # 🚀 5. 验证阶段：使用解耦出的 EMA 影子模型进行稳健评估
        eval_model = ema_updater.ema_model
        eval_model.eval()
        evaluator = MultiTaskEvaluator(num_classes=40, ignore_index=255)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Valid (EMA Model)", ncols=110, dynamic_ncols=False):
                images = batch['images'].to(Config.DEVICE)
                targets_seg = batch['segmentation'].to(Config.DEVICE)
                targets_depth = batch['depth'].to(Config.DEVICE)

                with autocast('cuda'):
                    # 调用 EMA 模型的 predict_all！享受概率融合与频率补偿
                    pred_seg, pred_depth = eval_model.predict_all(images, current_epoch=epoch)

                evaluator.update_seg(pred_seg.float(), targets_seg)
                evaluator.update_depth(pred_depth.float(), targets_depth)

        eval_metrics = evaluator.get_metrics()
        miou = eval_metrics.get('mIoU', 0.0)
        rmse = eval_metrics.get('RMSE', 999.0)
        pixel_acc = eval_metrics.get('PixelAcc', eval_metrics.get('pixel_acc', 0.0))

        print(f"📊 验证结果 (EMA Seg)  : mIoU: {miou:.4f} | Pixel Acc: {pixel_acc:.4f}")
        print(f"📐 验证结果 (EMA Depth): RMSE: {rmse:.4f}m\n")

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['seg_loss'].append(avg_seg)
        history['depth_loss'].append(avg_depth)
        history['val_miou'].append(miou)
        history['val_rmse'].append(rmse)

        plot_metrics(history, os.path.join(Config.SAVE_DIR, 'training_curves.png'))

        # 🚀 6. 保存最佳模型时，直接保存 EMA 模型的权重
        if miou > best_miou or (miou >= best_miou - 0.005 and rmse < best_rmse):
            best_miou = max(best_miou, miou)
            best_rmse = min(best_rmse, rmse)
            torch.save(
                {
                    'model_state_dict': eval_model.state_dict(),
                    'best_miou': best_miou,
                    'best_rmse': best_rmse
                },
                os.path.join(Config.SAVE_DIR, 'best_model.pth')
            )
            print(f"💾 极品断点！EMA Best Model 已保存! (EMA 历史最佳 -> mIoU: {best_miou:.4f}, RMSE: {best_rmse:.4f})\n")

        # 7. 保存常规检查点，同时记录主模型和 EMA 影子模型的状态
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'ema_state_dict': eval_model.state_dict(),
            'dynamic_loss_state_dict': dynamic_loss_weighting.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_dyn_state_dict': optimizer_dyn.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_miou': best_miou,
            'best_rmse': best_rmse
        }, os.path.join(Config.SAVE_DIR, 'latest_checkpoint.pth'))
        # 📸 8. 触发路由器可解释性探针可视化
        # ==========================================
        plot_router_visualizations(
            model=model,
            epoch=epoch,
            save_dir=os.path.join(Config.SAVE_DIR, 'router_visuals'),
            target_epochs=[0, 5, 15, 24, 49, 60, 74, 85, 99]  # 你希望提取快照的 Epoch
        )

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()