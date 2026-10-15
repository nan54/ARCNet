from __future__ import annotations
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 🚀 空间寻址魔法：解决 "No module named 'dinov3'"
# 自动把上一级目录加入系统路径，完美兼容绝对导入
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ==========================================
# 🚀 强制导入官方核心
# ==========================================
try:
    from dinov3_core import DinoVisionTransformer, vit_small, vit_base, vit_large
except ImportError as e:
    print("\n" + "=" * 60)
    print(f"❌ 致命导入错误 (Fatal ImportError):")
    print(f"无法正确加载 dinov3_core.py！具体原因是: {e}")
    print("=" * 60 + "\n")
    raise e


class SimpleViTAdapter(nn.Module):
    """
    🌟 ViT-Small 降维打击桥接器 (输入 embed_dim=384，输出通道统一对齐为 [64, 128, 256, 512])
    """

    def __init__(self, embed_dim=384, out_dims=[64, 128, 256, 512]):
        super().__init__()

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.GroupNorm(16, embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, out_dims[0], kernel_size=2, stride=2),
            nn.GroupNorm(16, out_dims[0]),
            nn.GELU()
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, out_dims[1], kernel_size=2, stride=2),
            nn.GroupNorm(16, out_dims[1]),
            nn.GELU()
        )

        self.identity = nn.Sequential(
            nn.Conv2d(embed_dim, out_dims[2], kernel_size=1),
            nn.GroupNorm(16, out_dims[2]),
            nn.GELU()
        )

        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(embed_dim, out_dims[3], kernel_size=1),
            nn.GroupNorm(16, out_dims[3]),
            nn.GELU()
        )

    def forward(self, features, original_hw):
        H, W = original_hw
        size1 = (H // 4, W // 4)
        size2 = (H // 8, W // 8)
        size3 = (H // 16, W // 16)
        size4 = (H // 32, W // 32)

        f1 = F.interpolate(self.up2(features[0]), size=size1, mode='bilinear', align_corners=False)
        f2 = F.interpolate(self.up1(features[1]), size=size2, mode='bilinear', align_corners=False)
        f3 = F.interpolate(self.identity(features[2]), size=size3, mode='bilinear', align_corners=False)
        f4 = F.interpolate(self.down1(features[3]), size=size4, mode='bilinear', align_corners=False)

        return f1, f2, f3, f4


class DINOv3MultiScaleBackbone(nn.Module):
    """
    🌟 DINOv3 Small 主干网络 (指定 model_size='small', patch_size=16)
    """

    def __init__(self, model_size='small', patch_size=16, pretrained_weights=None):
        super().__init__()
        self.patch_size = patch_size

        # 🛡️ 救命修复：传入 LayerScale, StorageTokens 等控制参数，接住被抛弃的官方权重！
        if model_size == 'small':
            self.vit = vit_small(
                patch_size=patch_size,
                layerscale_init=1e-5,  # 👈 核心：开启 LayerScale，接住 ls1.gamma
                n_storage_tokens=4,  # 👈 修复：官方权重的 storage_tokens 维度是 4 而不是 1
                mask_k_bias=True  # 👈 核心：接住 bias_mask
            )
            embed_dim = 384
            self.out_indices = [2, 5, 8, 11]
        elif model_size == 'base':
            self.vit = vit_base(
                patch_size=patch_size,
                layerscale_init=1e-5,
                n_storage_tokens=4,  # 👈 修复对齐
                mask_k_bias=True
            )
            embed_dim = 768
            self.out_indices = [2, 5, 8, 11]
        elif model_size == 'large':
            self.vit = vit_large(
                patch_size=patch_size,
                layerscale_init=1e-5,
                n_storage_tokens=4,  # 👈 修复对齐
                mask_k_bias=True
            )
            embed_dim = 1024
            self.out_indices = [5, 11, 17, 23]
        else:
            raise ValueError(f"Unsupported model size: {model_size}")

        if pretrained_weights and os.path.exists(pretrained_weights):
            print(f"📥 正在加载 DINOv3 预训练权重: {pretrained_weights}")
            state_dict = torch.load(pretrained_weights, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']

            msg = self.vit.load_state_dict(state_dict, strict=False)
            print(f"✅ 权重加载反馈: {msg}")

        # 🛡️ 强制解锁 Backbone 所有参数的梯度回传权限
        for param in self.vit.parameters():
            param.requires_grad = True

        self.adapter = SimpleViTAdapter(embed_dim=embed_dim, out_dims=[64, 128, 256, 512])

        self._register_gradient_probes()

    def _register_gradient_probes(self):
        def make_hook(name):
            def gradient_hook(grad):
                if grad is not None:
                    grad_norm = grad.norm().item()
                    if grad_norm == 0.0:
                        print(f"⚠️ [梯度警报] 主干网络层 {name} 的梯度为 0.0！")
                else:
                    print(f"❌ [严重阻断] 主干网络层 {name} 的梯度为 None！")

            return gradient_hook

        for name, param in self.vit.named_parameters():
            if 'blocks.11' in name and 'weight' in name:
                param.register_hook(make_hook(name))
                break

    def forward_features(self, x):
        B, C, H, W = x.shape

        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h))
        else:
            x_padded = x

        intermediate_outputs = self.vit.get_intermediate_layers(
            x_padded, n=self.out_indices, reshape=True, return_class_token=False, norm=True
        )

        f1, f2, f3, f4 = self.adapter(intermediate_outputs, original_hw=(H + pad_h, W + pad_w))

        f1 = f1[..., :H // 4, :W // 4]
        f2 = f2[..., :H // 8, :W // 8]
        f3 = f3[..., :H // 16, :W // 16]
        f4 = f4[..., :H // 32, :W // 32]

        return f1, f2, f3, f4