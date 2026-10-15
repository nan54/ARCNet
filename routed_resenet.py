import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d  # 🚀 导入神器：可变形卷积

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "./pretrain/huggingface"
os.makedirs("./pretrain/huggingface", exist_ok=True)

try:
    from dinov3_adapter import DINOv3MultiScaleBackbone
except ImportError:
    print("⚠️ 警告: 未找到 dinov3_adapter.py！请确保文件已创建。")

from multi_task_routing import AsymmetricTaskRouter

NYU40_LOCAL_FREQS = [
    0.2851, 0.1652, 0.0412, 0.0629, 0.0385, 0.0330, 0.0215, 0.0318,
    0.0200, 0.0150, 0.0081, 0.0128, 0.0125, 0.0184, 0.0096, 0.0116,
    0.0027, 0.0069, 0.0084, 0.0222, 0.0171, 0.1201, 0.0100, 0.0052,
    0.0127, 0.0039, 0.0081, 0.0028, 0.0186, 0.0022, 0.0036, 0.0049,
    0.0058, 0.0016, 0.0041, 0.0056, 0.0088, 0.0031, 0.0062, 0.0088
]


class TaskAwareGroupNorm2d(nn.Module):
    def __init__(self, num_features, task_count=2):
        super().__init__()
        if num_features % 16 == 0:
            num_groups = 16
        elif num_features % 8 == 0:
            num_groups = 8
        elif num_features % 4 == 0:
            num_groups = 4
        else:
            num_groups = 1
        self.gns = nn.ModuleList(
            [nn.GroupNorm(num_groups=num_groups, num_channels=num_features) for _ in range(task_count)])

    def forward(self, x, task_id):
        return self.gns[task_id](x.contiguous()).contiguous()


class TaskAwareConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, task_count=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        self.gn = TaskAwareGroupNorm2d(out_channels, task_count)
        self.relu = nn.ReLU(True)

    def forward(self, x, task_id):
        return self.relu(self.gn(self.conv(x), task_id))


class SharedSegFormerMLPDecoder(nn.Module):
    def __init__(self, in_channels_list=[72, 144, 288, 576], embedding_dim=256, task_count=2, dropout_ratio=0.1):
        super().__init__()
        self.c1_proj = TaskAwareConvBlock(in_channels_list[0], embedding_dim, 1, task_count)
        self.c2_proj = TaskAwareConvBlock(in_channels_list[1], embedding_dim, 1, task_count)
        self.c3_proj = TaskAwareConvBlock(in_channels_list[2], embedding_dim, 1, task_count)
        self.c4_proj = TaskAwareConvBlock(in_channels_list[3], embedding_dim, 1, task_count)

        # 🚀 引入 DCNv2 引擎：取代刻板的 3x3 卷积，缝合边缘狗牙
        # 18 通道用于偏移 (Offset)，9 通道用于掩码 (Mask) = 27 通道
        self.offset_mask_conv = nn.Conv2d(embedding_dim * 4, 27, kernel_size=3, padding=1)
        # 初始化为 0，让模型前期按标准 3x3 卷积运行，后期再平滑弯曲！
        nn.init.constant_(self.offset_mask_conv.weight, 0.)
        nn.init.constant_(self.offset_mask_conv.bias, 0.)

        self.spatial_mixer = DeformConv2d(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim * 4,
            kernel_size=3,
            padding=1,
            groups=embedding_dim * 4,
            bias=False
        )

        self.fuse_conv = nn.Conv2d(embedding_dim * 4, embedding_dim, 1, bias=False)
        self.fuse_gn = TaskAwareGroupNorm2d(embedding_dim, task_count)
        self.fuse_relu = nn.ReLU(True)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, features, task_id):
        c1, c2, c3, c4 = features
        target_size = c1.shape[2:]

        _c1 = self.c1_proj(c1, task_id)
        _c2 = F.interpolate(self.c2_proj(c2, task_id), size=target_size, mode='bilinear', align_corners=False)
        _c3 = F.interpolate(self.c3_proj(c3, task_id), size=target_size, mode='bilinear', align_corners=False)
        _c4 = F.interpolate(self.c4_proj(c4, task_id), size=target_size, mode='bilinear', align_corners=False)

        concat_feat = torch.cat([_c1, _c2, _c3, _c4], dim=1)

        # 🚀 DCNv2 贴合边界运算
        out = self.offset_mask_conv(concat_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        mixed_feat = self.spatial_mixer(concat_feat, offset, mask)

        fused = self.fuse_relu(self.fuse_gn(self.fuse_conv(mixed_feat), task_id))
        return self.dropout(fused)


class SemanticAnchoredQueryHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=40, num_heads=8):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.class_queries = nn.Parameter(torch.randn(1, num_classes, in_channels) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(in_channels)
        self.norm_kv = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_channels), nn.Linear(in_channels, in_channels * 2),
            nn.GELU(), nn.Linear(in_channels * 2, in_channels)
        )
        self.feat_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.GroupNorm(16, in_channels),
            nn.GELU()
        )
        self.temperature = nn.Parameter(torch.tensor(15.0))
        self.mask_refine = nn.Sequential(
            nn.Conv2d(num_classes + in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, in_channels // 2),
            nn.ReLU(True),
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=3, padding=1)
        )

    def load_clip_embeddings(self, clip_embeddings_path="nyu40_clip_queries.pt"):
        if os.path.exists(clip_embeddings_path):
            embeds = torch.load(clip_embeddings_path, map_location='cpu')
            self.class_queries.data = embeds.unsqueeze(0).to(self.class_queries.device)
            print("✅ 成功加载 CLIP Semantic Prior 初始化 Query！")

    def get_query_diversity_loss(self, epoch=0):
        q = self.class_queries.squeeze(0)
        q_norm = F.normalize(q, p=2, dim=-1)
        sim_matrix = torch.matmul(q_norm, q_norm.transpose(0, 1))
        identity = torch.eye(self.num_classes, device=q.device)
        off_diagonal_sim = torch.abs(sim_matrix - identity)
        loss_orth = (off_diagonal_sim ** 2).mean()
        return loss_orth * 0.05

    def forward(self, x, return_queries=False):
        B, C, H, W = x.shape
        pixel_feats = self.feat_proj(x)
        flat_feats = pixel_feats.flatten(2).transpose(1, 2)
        q_input = self.class_queries.expand(B, -1, -1)

        if self.training:
            query_mask = (torch.rand(B, self.num_classes, 1, device=q_input.device) > 0.2).float()
            q_input = q_input * query_mask

        q_out, _ = self.cross_attn(self.norm_q(q_input), self.norm_kv(flat_feats), self.norm_kv(flat_feats))
        q_dynamic = q_input + q_out + self.ffn(q_input + q_out)
        q_norm = F.normalize(q_dynamic, p=2, dim=-1)
        f_norm = F.normalize(pixel_feats, p=2, dim=1)
        mask_logits_raw = torch.einsum('bqc, bchw -> bqhw', q_norm, f_norm) * self.temperature
        concat_feats = torch.cat([mask_logits_raw, pixel_feats], dim=1)
        mask_logits_refined = self.mask_refine(concat_feats)

        if return_queries:
            return mask_logits_raw + mask_logits_refined, q_dynamic
        return mask_logits_raw + mask_logits_refined


class EnhancedDepthHead(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.global_depth_query = nn.Parameter(torch.randn(1, 1, in_channels) * 0.02)

        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(16, in_channels // 2)

        self.branch_1 = nn.Conv2d(in_channels // 2, in_channels // 4, 1, bias=False)
        self.branch_2 = nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=4, dilation=4, bias=False)
        self.branch_3 = nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=8, dilation=8, bias=False)

        self.fuse_conv = nn.Conv2d((in_channels // 4) * 3, in_channels // 4, 1, bias=False)
        self.fuse_gn = nn.GroupNorm(8, in_channels // 4)

        self.query_gate = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.Sigmoid()
        )

        self.pred = nn.Sequential(
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Softplus(),
            nn.Threshold(0.1, 0.1)
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.relu(self.gn1(self.conv1(x)))

        b1 = self.branch_1(feat)
        b2 = self.branch_2(feat)
        b3 = self.branch_3(feat)
        feat_aspp = torch.cat([b1, b2, b3], dim=1)
        feat = self.relu(self.fuse_gn(self.fuse_conv(feat_aspp)))

        global_context = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        current_depth_query = self.global_depth_query.expand(B, -1, -1).squeeze(1) + global_context

        gate = self.query_gate(current_depth_query).unsqueeze(-1).unsqueeze(-1)
        feat_modulated = feat * (1.0 + gate)

        return self.pred(feat_modulated)

class CrossTaskAttention(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels // 4, 1),
            nn.ReLU(True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.GroupNorm(16, in_channels),
            nn.Sigmoid()
        )

    def forward(self, f_concat):
        c_attn = self.channel_gate(f_concat)
        s_attn = self.spatial_gate(f_concat)
        return c_attn * s_attn



# ========================================================
# 🚀 4. 大一统多任务主网络: V24 终极拔高版
# ========================================================
class RoutedNYUDv2Net(nn.Module):
    def __init__(self, task_count=2, num_classes=40, align_corners=False):
        super(RoutedNYUDv2Net, self).__init__()
        self.task_count = task_count
        self.num_classes = num_classes
        self.align_corners = align_corners

        dinov3_weight = "./pretrain/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

        self.backbone = DINOv3MultiScaleBackbone(
            model_size='base',
            patch_size=16,
            pretrained_weights=dinov3_weight if os.path.exists(dinov3_weight) else None
        )

        self.enc_router1 = AsymmetricTaskRouter(64, alpha=0.3, name="enc_tr1")
        self.enc_router2 = AsymmetricTaskRouter(128, alpha=0.3, name="enc_tr2")
        self.enc_router3 = AsymmetricTaskRouter(256, alpha=0.2, name="enc_tr3")
        self.enc_router4 = AsymmetricTaskRouter(512, alpha=0.1, name="enc_tr4")

        self.shared_decoder = SharedSegFormerMLPDecoder(
            in_channels_list=[64, 128, 256, 512],
            embedding_dim=256,
            task_count=task_count
        )

        self.cross_task_align_seg = CrossTaskAttention(in_channels=256)
        self.cross_task_align_dep = CrossTaskAttention(in_channels=256)
        self.head_seg = SemanticAnchoredQueryHead(in_channels=256, num_classes=num_classes)
        self.head_depth = EnhancedDepthHead(in_channels=256)


        self.aux_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def _apply_stage_freeze(self, current_epoch):
        if not self.training:
            return

        for p in self.backbone.parameters(): p.requires_grad = True

        if hasattr(self.backbone, 'vit'):
            for p in self.backbone.vit.patch_embed.parameters(): p.requires_grad = False
            self.backbone.vit.cls_token.requires_grad = False

    def forward(self, x, current_epoch=0, target_seg=None):
        return self.forward_mtl(x, current_epoch, target_seg=target_seg)

    def forward_mtl(self, x, current_epoch=0, return_feat=False, target_seg=None):
        self._apply_stage_freeze(current_epoch)

        h_orig, w_orig = x.shape[2:]
        feat1, feat2, feat3, feat4 = self.backbone.forward_features(x)

        aux_pred = None
        if self.training:
            aux_pred = self.aux_head(feat4)
            aux_pred = F.interpolate(aux_pred, size=(h_orig, w_orig), mode='bilinear', align_corners=self.align_corners)

        r1_s, r1_d = self.enc_router1(feat1, current_epoch)
        r2_s, r2_d = self.enc_router2(feat2, current_epoch)
        r3_s, r3_d = self.enc_router3(feat3, current_epoch)
        r4_s, r4_d = self.enc_router4(feat4, current_epoch)

        f_seg_base = self.shared_decoder([r1_s, r2_s, r3_s, r4_s], task_id=0)
        f_dep_base = self.shared_decoder([r1_d, r2_d, r3_d, r4_d], task_id=1)

        f_concat = torch.cat([f_seg_base, f_dep_base], dim=1)
        gate_seg = self.cross_task_align_seg(f_concat)
        gate_dep = self.cross_task_align_dep(f_concat)

        f_seg_final = f_seg_base * (1.0 + gate_seg)
        f_dep_final = f_dep_base * (1.0 + gate_dep)
        f_seg_guided = f_seg_final

        seg_pred_quarter, seg_queries = self.head_seg(f_seg_guided, return_queries=True)
        depth_pred_quarter = self.head_depth(f_dep_final)

        seg_pred = F.interpolate(seg_pred_quarter, size=(h_orig, w_orig), mode='bilinear',
                                 align_corners=self.align_corners)
        depth_pred = F.interpolate(depth_pred_quarter, size=(h_orig, w_orig), mode='bilinear',
                                   align_corners=self.align_corners)

        if return_feat:
            return (seg_pred, aux_pred), depth_pred, None, f_seg_guided
        return (seg_pred, aux_pred), depth_pred


    @torch.no_grad()
    def predict_all(self, x, current_epoch=0, tau_eval=0.08):
        module_eval_flag = self.training
        self.eval()

        # 🚀 坚决听你的！彻底关闭坑人的多尺度缩放！
        # 只保留 1.0 原比例，确保 DINOv3 几何特征完美。
        # 允许水平翻转 (Flip)，因为翻转不改变物体宽高比，是安全的。
        scales = [1.0]
        flips = [False, True]

        b, c, h, w = x.shape
        # 🛡️ 存放 Softmax 后的概率，防止 Logit 数值膨胀
        final_seg_prob = torch.zeros(b, self.num_classes, h, w, device=x.device)
        final_depth = torch.zeros(b, 1, h, w, device=x.device)

        for scale in scales:
            # 因为只有 1.0，不会有任何形变，原汁原味！
            scaled_x = x

            for flip in flips:
                input_x = torch.flip(scaled_x, dims=[3]) if flip else scaled_x
                outputs = self.forward_mtl(input_x, current_epoch=current_epoch, target_seg=None)

                # 提取预测结果
                seg = outputs[0][0]
                dep = outputs[1]

                if flip:
                    seg = torch.flip(seg, dims=[3])
                    dep = torch.flip(dep, dims=[3])

                # 🚀 核心修复：就算是原图和翻转图融合，也必须转成概率后再相加！
                final_seg_prob += F.softmax(seg, dim=1)
                final_depth += dep

        count_seg = len(scales) * len(flips)
        count_dep = len(scales) * len(flips)

        raw_seg_prob = final_seg_prob / count_seg
        raw_depth = final_depth / count_dep

        freq_tensor = torch.tensor(NYU40_LOCAL_FREQS, dtype=torch.float32, device=x.device)
        freq_tensor = freq_tensor / (freq_tensor.sum() + 1e-6)
        logit_adj = torch.log(freq_tensor + 1e-8)
        logit_adj = torch.clamp(logit_adj, min=-4.0, max=0.0)

        # 🚀 概率转换回 Logit，再做频率长尾惩罚
        calibrated_seg = torch.log(raw_seg_prob + 1e-8) - tau_eval * logit_adj.view(1, self.num_classes, 1, 1)

        if module_eval_flag:
            self.train()

        return calibrated_seg, raw_depth