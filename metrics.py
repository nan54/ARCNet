from __future__ import annotations
import torch
import torch.nn.functional as F
import torch.distributed as dist


class MultiTaskEvaluator:
    def __init__(self, num_classes=40, ignore_index=255, apply_eigen_crop=True, align_scale=False):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.apply_eigen_crop = apply_eigen_crop
        self.align_scale = align_scale
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 语义分割基础统计
        self.total_intersect = torch.zeros(num_classes, dtype=torch.int64, device=self.device)
        self.total_union = torch.zeros(num_classes, dtype=torch.int64, device=self.device)
        self.total_target = torch.zeros(num_classes, dtype=torch.int64, device=self.device)
        self.total_correct = torch.tensor(0, dtype=torch.int64, device=self.device)
        self.total_label = torch.tensor(0, dtype=torch.int64, device=self.device)

        # 🚀 探针系统：追踪 Logit、Softmax 概率、预测像素数以及【GT Margin 相变指标】
        self.class_max_logits = torch.full((num_classes,), -999.0, device=self.device)
        self.class_sum_probs = torch.zeros(num_classes, dtype=torch.float64, device=self.device)
        self.class_prob_samples = torch.zeros(num_classes, dtype=torch.int64, device=self.device)
        self.class_pred_pixels = torch.zeros(num_classes, dtype=torch.int64, device=self.device)

        # 🔥 核心增加：GT 区域内 Logit Margin 累加器 (z_gt - max(z_others))
        self.class_sum_margins = torch.zeros(num_classes, dtype=torch.float64, device=self.device)

        # 深度估计统计
        self.total_sq_error = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.total_abs_rel = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.total_delta1 = torch.tensor(0, dtype=torch.int64, device=self.device)
        self.total_delta2 = torch.tensor(0, dtype=torch.int64, device=self.device)
        self.total_delta3 = torch.tensor(0, dtype=torch.int64, device=self.device)
        self.total_valid_pixels = torch.tensor(0, dtype=torch.int64, device=self.device)

    def reset(self):
        self.total_intersect.zero_()
        self.total_union.zero_()
        self.total_target.zero_()
        self.total_correct.zero_()
        self.total_label.zero_()
        self.class_max_logits.fill_(-999.0)
        self.class_sum_probs.zero_()
        self.class_prob_samples.zero_()
        self.class_pred_pixels.zero_()
        self.class_sum_margins.zero_()
        self.total_sq_error.zero_()
        self.total_abs_rel.zero_()
        self.total_delta1.zero_()
        self.total_delta2.zero_()
        self.total_delta3.zero_()
        self.total_valid_pixels.zero_()

    @torch.no_grad()
    def update_seg(self, pred_logits, target):
        if isinstance(pred_logits, tuple):
            pred_logits = pred_logits[0]
        if len(target.shape) == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        B, C, H, W = pred_logits.shape
        probs = F.softmax(pred_logits, dim=1)
        pred_labels = torch.argmax(pred_logits, dim=1)

        mask = (target != self.ignore_index)

        # 🚀 提取每个像素位置前 2 大的 Logit，精准计算与最大竞争者的 Margin
        top2_vals, top2_idxs = torch.topk(pred_logits, k=2, dim=1)

        for c in range(C):
            c_mask = mask & (target == c)
            if c_mask.any():
                c_max = pred_logits[:, c][c_mask].max()
                if c_max > self.class_max_logits[c]:
                    self.class_max_logits[c] = c_max
                self.class_sum_probs[c] += probs[:, c][c_mask].sum().item()
                self.class_prob_samples[c] += c_mask.sum().item()

                # 🔥 计算该类在 GT 区域内的真实 Margin: z_gt - max_{k != c}(z_k)
                z_gt = pred_logits[:, c][c_mask]
                top1_v = top2_vals[:, 0][c_mask]
                top2_v = top2_vals[:, 1][c_mask]
                top1_i = top2_idxs[:, 0][c_mask]

                # 如果第一名就是类别 c，则竞争者为第二名；否则竞争者为第一名
                z_max_others = torch.where(top1_i == c, top2_v, top1_v)
                margins = z_gt - z_max_others
                self.class_sum_margins[c] += margins.sum().item()

        p = pred_labels[mask]
        t = target[mask]

        if p.numel() == 0:
            return

        intersect = p[p == t]

        self.total_correct += intersect.numel()
        self.total_label += t.numel()

        area_intersect = torch.bincount(intersect, minlength=self.num_classes)[:self.num_classes]
        area_pred = torch.bincount(p, minlength=self.num_classes)[:self.num_classes]
        area_target = torch.bincount(t, minlength=self.num_classes)[:self.num_classes]
        area_union = area_pred + area_target - area_intersect

        if 0 <= self.ignore_index < self.num_classes:
            area_intersect[self.ignore_index] = 0
            area_union[self.ignore_index] = 0
            area_target[self.ignore_index] = 0

        self.total_intersect += area_intersect
        self.total_union += area_union
        self.total_target += area_target
        self.class_pred_pixels += area_pred

    @torch.no_grad()
    def update_depth(self, pred, target):
        if isinstance(pred, tuple):
            pred = pred[0]
        if pred.shape != target.shape:
            pred = pred.view_as(target)

        valid_mask = (target > 1e-3) & (target < 10.0)

        h, w = target.shape[-2], target.shape[-1]
        if self.apply_eigen_crop and h >= 471 and w >= 601:
            crop_mask = torch.zeros_like(valid_mask)
            if crop_mask.dim() == 4:
                crop_mask[:, :, 45:471, 41:601] = True
            elif crop_mask.dim() == 3:
                crop_mask[:, 45:471, 41:601] = True
            elif crop_mask.dim() == 2:
                crop_mask[45:471, 41:601] = True
            valid_mask = valid_mask & crop_mask

        if valid_mask.sum() == 0:
            return

        valid_pred = pred[valid_mask].to(torch.float64)
        valid_target = target[valid_mask].to(torch.float64)

        if self.align_scale:
            scale = torch.sum(valid_target * valid_pred) / (torch.sum(valid_pred * valid_pred) + 1e-8)
            valid_pred = valid_pred * scale

        self.total_sq_error += torch.sum((valid_pred - valid_target) ** 2)
        self.total_abs_rel += torch.sum(torch.abs(valid_pred - valid_target) / valid_target)

        ratio = torch.max(valid_pred / valid_target, valid_target / valid_pred)
        self.total_delta1 += torch.sum(ratio < 1.25)
        self.total_delta2 += torch.sum(ratio < (1.25 ** 2))
        self.total_delta3 += torch.sum(ratio < (1.25 ** 3))
        self.total_valid_pixels += valid_mask.sum()

    def get_metrics(self):
        intersect = self.total_intersect.clone()
        union = self.total_union.clone()
        correct = self.total_correct.clone()
        label_total = self.total_label.clone()

        sq_error = self.total_sq_error.clone()
        abs_rel = self.total_abs_rel.clone()
        d1, d2, d3 = self.total_delta1.clone(), self.total_delta2.clone(), self.total_delta3.clone()
        valid_pixels = self.total_valid_pixels.clone()
        pred_pixels = self.class_pred_pixels.clone()

        if dist.is_available() and dist.is_initialized():
            for tensor in [intersect, union, correct, label_total, sq_error, abs_rel, d1, d2, d3, valid_pixels, pred_pixels]:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        intersect, union = intersect.cpu(), union.cpu()
        correct, label_total = correct.cpu(), label_total.cpu()
        sq_error, abs_rel = sq_error.cpu(), abs_rel.cpu()
        d1, d2, d3 = d1.cpu(), d2.cpu(), d3.cpu()
        valid_pixels = valid_pixels.cpu()
        pred_pixels = pred_pixels.cpu()

        iou_per_class = intersect.float() / (union.float() + 1e-10)
        iou_per_class[union == 0] = float('nan')
        if 0 <= self.ignore_index < self.num_classes:
            iou_per_class[self.ignore_index] = float('nan')

        miou = torch.nanmean(iou_per_class).item() if not torch.isnan(torch.nanmean(iou_per_class)) else 0.0
        pixel_acc = (correct.float() / (label_total.float() + 1e-10)).item() if label_total > 0 else 0.0

        iou_array = iou_per_class.cpu().numpy().tolist()

        mean_prob_per_class = []
        mean_margin_per_class = []
        for c in range(self.num_classes):
            if self.class_prob_samples[c] > 0:
                mean_prob_per_class.append((self.class_sum_probs[c] / self.class_prob_samples[c]).item())
                mean_margin_per_class.append((self.class_sum_margins[c] / self.class_prob_samples[c]).item())
            else:
                mean_prob_per_class.append(0.0)
                mean_margin_per_class.append(-999.0)

        max_logit_per_class = self.class_max_logits.cpu().numpy().tolist()
        pred_pixels_per_class = pred_pixels.cpu().numpy().tolist()

        n_pixels = valid_pixels.item()
        if n_pixels > 0:
            rmse = (sq_error.item() / n_pixels) ** 0.5
            rel = abs_rel.item() / n_pixels
            delta1, delta2, delta3 = d1.item() / n_pixels, d2.item() / n_pixels, d3.item() / n_pixels
        else:
            rmse = rel = delta1 = delta2 = delta3 = 0.0

        return {
            'mIoU': miou,
            'PixelAcc': pixel_acc,
            'RMSE': rmse,
            'REL': rel,
            'd1': delta1,
            'd2': delta2,
            'd3': delta3,
            'IoU_Array': iou_array,
            'Max_Logits': max_logit_per_class,
            'Mean_Probs': mean_prob_per_class,
            'Pred_Pixels': pred_pixels_per_class,
            'GT_Margins': mean_margin_per_class  # 🚀 关键导出：GT 区域平均 Logit Margin
        }