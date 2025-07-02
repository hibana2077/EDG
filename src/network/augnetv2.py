import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K
from timm.models.tiny_vit import TinyVitBlock
from timm.models.convnext import ConvNeXtBlock

# ===== 1. 動態 Normalize 函式 =====
def dynamic_normalize(x, mean, std, eps=1e-6):
    """
    x   : (B, C, H, W)
    mean: (B, C)           逐 batch + channel
    std : (B, C)           逐 batch + channel，需為正值
    """
    B, C, H, W = x.shape
    mean = mean.view(B, C, 1, 1)
    std  = std.view(B, C, 1, 1).clamp(min=eps)  # 避免除 0
    return (x - mean) / std

# ===== 2. AugNet =====
class AugNet(nn.Module):
    def __init__(self, dim: int = 224, num_heads: int = 8, out_dim: int = 10):
        super().__init__()
        # (a) ViT backbone（你自己的 TinyViTBlock）
        # self.backbone = TinyVitBlock(dim=dim, num_heads=num_heads)
        self.backbone = ConvNeXtBlock(in_chs=3)

        # (b) 產生 μ / σ 的 head
        self.param_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (B, 3, 1, 1)
            nn.Flatten(),              # (B, 3)
            nn.Linear(3, 6)          # 3 通道 * (μ, σ)
        )

    def forward(self, x):
        # 1) 先用 ViT 抽特徵
        feats = self.backbone(x)            # (B, dim, H', W')

        # 2) 由特徵預測 μ / σ
        params = self.param_head(feats)     # (B, 6)
        mu, log_sigma = params[:, :3], params[:, 3:]
        sigma = F.softplus(log_sigma)       # 保證 σ > 0

        # 3) 對「原始影像 x」做動態 Normalize
        x_norm = dynamic_normalize(x, mu, sigma)

        # 4) 這裡示範把 x_norm 再丟進 backbone 做分類
        #    你也可以把 x_norm 傳給別的支線
        feats_norm = self.backbone(x_norm)
        return feats_norm

if __name__ == "__main__":
    # Example usage
    model = AugNet()
    input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image
    output = model(input_tensor)
    print(output.shape)  # Output shape will depend on the TinyVitBlock implementation
    # cal mse
    target_tensor = torch.randn_like(output)
    mse_loss = F.mse_loss(output, target_tensor)
    l1_loss = F.l1_loss(output, target_tensor)
    print(f'L1 Loss: {l1_loss.item()}')  # Print the L1 loss value
    print(f'MSE Loss: {mse_loss.item()}')  # Print the MSE loss value