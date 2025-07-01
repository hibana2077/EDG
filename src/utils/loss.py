import torch
import torch.nn.functional as F
from torch import nn

def centralized_moments(x: torch.Tensor, order: int, eps: float = 1e-8):
    """
    x: (B, C, H, W) 影像，值域可為 0~1 或已標準化
    order: 3 → Skewness；4 → Kurtosis
    以 (H, W) 為樣本維度，回傳 (B, C) 向量
    """
    B, C, H, W = x.shape
    x_flat = x.view(B, C, -1)                  # (B, C, N)
    mean = x_flat.mean(dim=-1, keepdim=True)   # (B, C, 1)
    std = x_flat.std(dim=-1, keepdim=True) + eps

    centralized = x_flat - mean
    cm = (centralized ** order).mean(dim=-1)   # (B, C)
    return cm / (std.squeeze(-1) ** order + eps)

def get_probs(t: torch.Tensor) -> torch.Tensor:
    B = t.size(0)
    return F.softmax(t.view(B, -1), dim=1)

class MomentHoMDivLoss(nn.Module):
    """
    HoM = Σ_c ( |skew_x - skew_x'| + |kurt_x - kurt_x'| )
    div = KL( p(x) ∥ p(x') )   # 與先前相同
    """
    def __init__(self, lambda_=1.0, gamma_=1.0):
        super().__init__()
        self.lambda_ = lambda_
        self.gamma_ = gamma_

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        # ---------- HoM ----------
        skew_x  = centralized_moments(x, 3)      # (B, C)
        skew_xp = centralized_moments(x_prime, 3)
        kurt_x  = centralized_moments(x, 4)
        kurt_xp = centralized_moments(x_prime, 4)

        l_hom = (skew_x  - skew_xp ).abs().mean() + (kurt_x - kurt_xp).abs().mean()

        # ---------- L_div ----------
        p = get_probs(x)
        q = get_probs(x_prime)
        l_div = F.kl_div(q.log(), p, reduction='batchmean')

        # ---------- Total ----------
        return self.lambda_ * l_hom + self.gamma_ * l_div