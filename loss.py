import torch
import torch.nn as nn
import torch.nn.functional as F

# NYUv2 真实像素出现频率 (0~39 顺序)
NYU40_LOCAL_FREQS = [
    0.2851, 0.1652, 0.0412, 0.0629, 0.0385, 0.0330, 0.0215, 0.0318,
    0.0200, 0.0150, 0.0081, 0.0128, 0.0125, 0.0184, 0.0096, 0.0116,
    0.0027, 0.0069, 0.0084, 0.0222, 0.0171, 0.1201, 0.0100, 0.0052,
    0.0127, 0.0039, 0.0081, 0.0028, 0.0186, 0.0022, 0.0036, 0.0049,
    0.0058, 0.0016, 0.0041, 0.0056, 0.0088, 0.0031, 0.0062, 0.0088
]


# ==========================================
# 🥇 1. 存在性多标签监督 (Presence Loss)
# ==========================================
def compute_presence_target(seg_gt, num_classes=40, ignore_index=255):
    B = seg_gt.shape[0]
    target = torch.zeros(B, num_classes, device=seg_gt.device)

    for b in range(B):
        labels = torch.unique(seg_gt[b])
        labels = labels[(labels != ignore_index) & (labels < num_classes)]
        if len(labels) > 0:
            target[b, labels.long()] = 1.0

    return target


class PresenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        freq = torch.tensor(NYU40_LOCAL_FREQS, dtype=torch.float32)
        pos_weight = 1.0 / torch.sqrt(freq + 1e-6)
        pos_weight = pos_weight / pos_weight.mean()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, pred_presence, seg_gt):
        target = compute_presence_target(seg_gt, num_classes=pred_presence.shape[1])
        return F.binary_cross_entropy_with_logits(
            pred_presence, target, pos_weight=self.pos_weight
        )


# ==========================================
# 🥈 2. Lovász-Softmax Loss
# ==========================================
def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes='present'):
    if probas.numel() == 0:
        return probas * 0.

    max_pixels = 200000
    if probas.size(0) > max_pixels:
        indices = torch.randperm(probas.size(0), device=probas.device)[:max_pixels]
        probas = probas[indices]
        labels = labels[indices]

    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))

    if len(losses) == 0:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


def lovasz_softmax(probas, labels, ignore=255):
    if probas.dim() == 4:
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().reshape(-1, C)
        labels = labels.reshape(-1)
    keep = labels != ignore
    if keep.sum() == 0:
        return probas.sum() * 0.0
    probas = probas[keep]
    labels = labels[keep]
    return lovasz_softmax_flat(probas, labels, classes='present')


# ==========================================
# 🥉 3. 纯净版 CE 与 Dice Loss (保留 75% OHEM)
# ==========================================
class LogitAdjustedOHEMLoss(nn.Module):
    def __init__(self, num_classes=40, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        freqs = torch.tensor(NYU40_LOCAL_FREQS, dtype=torch.float32)
        weights = 1.0 / torch.sqrt(freqs + 1e-4)
        weights = weights / weights.mean()
        self.register_buffer('class_weights', weights)

    def forward(self, preds, targets, epoch=0):
        mask = (targets != self.ignore_index)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        # 🚀 绝对纯净：直接用 reduction='mean'，不保留任何 topk 截断！
        # 让 DINOv3 舒舒服服地看到 100% 的全局像素
        loss = F.cross_entropy(
            preds,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction='mean'
        )
        return loss

class StructuralPreservingDiceLoss(nn.Module):
    def __init__(self, num_classes=40, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, preds, targets):
        prob = F.softmax(preds, dim=1)
        targets_clamped = torch.clamp(targets.long(), 0, self.num_classes - 1)
        targets_one_hot = F.one_hot(targets_clamped, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        mask = (targets != self.ignore_index).unsqueeze(1).float()

        intersection = torch.sum(prob * targets_one_hot * mask, dim=(0, 2, 3))
        union = torch.sum((prob + targets_one_hot) * mask, dim=(0, 2, 3))

        dice_loss = 1.0 - torch.mean((2. * intersection + 1e-5) / (union + 1e-5))
        return dice_loss


# ==========================================
# 🚀 4. 分割 Loss 大一统装配车间
# ==========================================
class AdvancedSegmentationLoss(nn.Module):
    def __init__(self, ignore_index=255, num_classes=40, **kwargs):
        super().__init__()
        self.ignore_index = ignore_index

        # 🚀 只保留纯净的加权 CE，彻底告别 OHEM
        self.ce_fn = LogitAdjustedOHEMLoss(num_classes=num_classes, ignore_index=ignore_index)
        # 🚀 Dice 已经彻底滚蛋，停止 Loss 内耗

    def forward(self, seg_logits, targets, epoch=0):
        # 1. 主力军：纯净 CE，打好全局像素基础
        total_loss = self.ce_fn(seg_logits, targets, epoch=epoch)

        lovasz_loss = torch.tensor(0.0, device=seg_logits.device)

        # 2. 冲刺军：Lovasz，从 20 轮开始极其平滑地介入，专攻 IoU！
        if epoch >= 20:
            # 极度温和的线性爬升：20轮(0.0) -> 40轮(0.5)，杜绝悬崖震荡
            w_lovasz = 0.5 * min(1.0, (epoch - 20) / 20.0)

            prob = F.softmax(seg_logits, dim=1)
            lovasz_loss = lovasz_softmax(prob, targets, ignore=self.ignore_index)
            total_loss += w_lovasz * lovasz_loss

        return {
            "loss": total_loss,
            "ce": total_loss - lovasz_loss if epoch >= 20 else total_loss,
            "dice": torch.tensor(0.0, device=seg_logits.device),  # 仅做字典占位，防报错
            "lovasz": lovasz_loss
        }

# ==========================================
# 🚀 5. 深度 Loss：尺度不变与 3D 边界对齐 (加入梯度屏蔽)
# ==========================================
class DepthSiLogLoss(nn.Module):
    def __init__(self, lambd=0.85, log_min=1e-3):
        super().__init__()
        self.lambd = lambd
        self.log_min = log_min

    def forward(self, pred, target, paste_mask=None):
        valid_mask = (target > 0.001) & (target < 10.0)

        # 🛡️ 屏蔽外挂：剔除物理造假的拼接区域
        if paste_mask is not None:
            valid_mask = valid_mask & (~paste_mask.unsqueeze(1))

        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_valid = torch.clamp(pred[valid_mask], min=self.log_min)
        target_valid = target[valid_mask]

        diff = torch.log(pred_valid) - torch.log(target_valid)
        loss_var = torch.mean(diff ** 2)
        loss_mean = (torch.mean(diff)) ** 2
        silog_loss = torch.sqrt(torch.clamp(loss_var - self.lambd * loss_mean, min=1e-6))
        return silog_loss


class AdvancedGeometricDepthLoss(nn.Module):
    def __init__(self, l1_weight=1.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.silog = DepthSiLogLoss(lambd=0.85)

    def _compute_gradients(self, depth):
        grad_x = depth[:, :, :, 1:] - depth[:, :, :, :-1]
        grad_y = depth[:, :, 1:, :] - depth[:, :, :-1, :]
        return grad_x, grad_y

    def forward(self, preds, target, epoch=100, paste_mask=None):
        if isinstance(preds, tuple):
            preds = preds[0]

        valid_mask = (target > 0.001) & (target < 10.0)
        if not valid_mask.any():
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        preds_clamped = torch.clamp(preds, min=0.05, max=10.0)
        pred_valid = preds_clamped[valid_mask]
        target_valid = target[valid_mask]

        # 1. 核心大局：SiLog 抓取相对结构
        silog_loss = self.silog(preds_clamped, target)

        # 2. 核心锚点：L1 抓取绝对物理尺度
        l1_loss = F.l1_loss(pred_valid, target_valid)

        # 3. 核心边缘：Grad 保持 2D 平滑度
        pred_grad_x, pred_grad_y = self._compute_gradients(preds_clamped)
        target_grad_x, target_grad_y = self._compute_gradients(target)

        valid_x = valid_mask[:, :, :, 1:] & valid_mask[:, :, :, :-1]
        valid_y = valid_mask[:, :, 1:, :] & valid_mask[:, :, :-1, :]
        grad_loss = torch.tensor(0.0, device=preds.device)
        if valid_x.any(): grad_loss += F.l1_loss(pred_grad_x[valid_x], target_grad_x[valid_x])
        if valid_y.any(): grad_loss += F.l1_loss(pred_grad_y[valid_y], target_grad_y[valid_y])

        # 🚀 斩草除根 2：剔除逼迫网络学习噪音的拉普拉斯 VNL！
        # 让网络专心输出平滑的深度图！
        return silog_loss + 0.15 * l1_loss + 0.2 * grad_loss

# ==========================================
# 🌟 6. 语义引导深度平滑损失 (防毒版)
# ==========================================
class SemanticGuidedDepthSmoothLoss(nn.Module):
    def __init__(self):
        super().__init__()

    # 🚀 救命修复 2：加入 paste_mask 参数
    def forward(self, pred_depth, seg_targets, paste_mask=None, ignore_index=255):
        depth_grad_x = torch.abs(pred_depth[:, :, :, 1:] - pred_depth[:, :, :, :-1])
        depth_grad_y = torch.abs(pred_depth[:, :, 1:, :] - pred_depth[:, :, :-1, :])

        seg_same_x = (seg_targets[:, :, 1:] == seg_targets[:, :, :-1]) & (seg_targets[:, :, 1:] != ignore_index)
        seg_same_y = (seg_targets[:, 1:, :] == seg_targets[:, :-1, :]) & (seg_targets[:, 1:, :] != ignore_index)

        # 🛡️ 屏蔽投毒：如果使用了 ClassMix，绝对不在拼接的假边缘处计算平滑度！
        if paste_mask is not None:
            valid_mask_x = ~paste_mask[:, :, 1:]
            valid_mask_y = ~paste_mask[:, 1:, :]
            seg_same_x = seg_same_x & valid_mask_x
            seg_same_y = seg_same_y & valid_mask_y

        loss_x = depth_grad_x[seg_same_x.unsqueeze(1)].mean() if seg_same_x.any() else torch.tensor(0.0, device=pred_depth.device)
        loss_y = depth_grad_y[seg_same_y.unsqueeze(1)].mean() if seg_same_y.any() else torch.tensor(0.0, device=pred_depth.device)

        return loss_x + loss_y


# ==========================================
# 🚀 7. 动态多任务权重平衡
# ==========================================
# ==========================================
# 🚀 7. 动态多任务权重平衡 (加入 Dummy 参数防报错版)
# ==========================================
class DynamicMultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super().__init__()
        # 🚀 塞入一个毫无作用的占位参数，专门为了骗过 optimizer_dyn，防止它报 "empty parameter list" 的错！
        self.dummy_param = nn.Parameter(torch.tensor(0.0))

    def forward(self, loss_seg, loss_depth, loss_ctx=None, epoch=None):
        # 分割 Loss 和 深度 Loss 采用固定最佳硬比例 1 : 1.5
        # 加上 self.dummy_param * 0.0，让这个假参数参与计算但完全不影响数值，避免 PyTorch 报找不到梯度的警告
        total_loss = loss_seg + loss_depth * 1.5 + self.dummy_param * 0.0

        return total_loss, 1.0, 1.5, 0.0