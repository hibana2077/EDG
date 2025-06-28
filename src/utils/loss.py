import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ---------- 工具函式 ---------- #
def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """
    feat: (B, C, H, W)
    回傳 Gram matrix，維度 (B, C, C)
    """
    B, C, H, W = feat.size()
    f = feat.view(B, C, H * W)          # (B, C, N)
    g = f.transpose(1, 2)               # (B, N, C)
    gram = torch.bmm(f, g) / (C * H * W)
    return gram

def get_probs(img: torch.Tensor) -> torch.Tensor:
    """
    將 (B,3,224,224) 影像映成 pixel-wise 機率分佈。
    這裡直接 softmax 到最後一維 (每張圖 3*224*224 個值)。
    若您有分類 logits，改成 softmax(logits, dim=1) 即可。
    """
    B = img.size(0)
    probs = F.softmax(img.view(B, -1), dim=1)
    return probs

# ---------- HoM + div Loss ---------- #
class HoMDivLoss(nn.Module):
    def __init__(self, lambda_=1.0, gamma_=1.0, device='cuda'):
        super().__init__()
        self.lambda_ = lambda_
        self.gamma_ = gamma_

        # 取 VGG19 直到 conv4_1（含）
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.feat_layers = nn.Sequential(*vgg[:22]).to(device).eval()  # relu4_1
        for p in self.feat_layers.parameters():
            p.requires_grad_(False)

        # Layer indices（0-based）對應 relu3_1, relu4_1
        self.target_idxs = [10, 19]

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """
        x, x_prime: (B,3,224,224) in [0,1] or normalized到 ImageNet
        回傳 L_total
        """
        # 特徵擷取
        feats_x = []
        feats_xp = []
        out = x
        out_p = x_prime
        for idx, layer in enumerate(self.feat_layers):
            out = layer(out)
            out_p = layer(out_p)
            if idx in self.target_idxs:
                feats_x.append(out)
                feats_xp.append(out_p)

        # --- L_HoM ---
        l_hom = 0.0
        for f1, f2 in zip(feats_x, feats_xp):
            m1 = gram_matrix(f1)
            m2 = gram_matrix(f2)
            l_hom += F.l1_loss(m1, m2)

        # --- L_div ---
        p = get_probs(x)
        q = get_probs(x_prime)
        # 加上 1e-8 避免 log(0)
        eps = 1e-8
        l_div = F.kl_div((q + eps).log(), p, reduction='batchmean')

        # --- Total ---
        l_total = self.lambda_ * l_hom + self.gamma_ * l_div
        return l_total
