import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K
from timm.models.tiny_vit import TinyVitBlock
from timm.models.convnext import ConvNeXtBlock

class LearnableContrast(nn.Module):
    def __init__(self, init_factor: float = 0.5):
        super().__init__()
        # 單一 learnable factor，或你想學兩端範圍都可以
        self.factor = nn.Parameter(torch.tensor(init_factor))

    def __repr__(self):
        return f"factor: {self.factor}"
    
    def forward(self, x):
        # (B,C,H,W)；center = 每張圖各自的平均亮度
        mean = x.mean(dim=[2,3], keepdim=True)
        return torch.clamp((x - mean) * self.factor + mean, 0., 1.)

class AugNet(nn.Module):
    def __init__(self, dim: int = 224, num_heads:int=8):
        super().__init__()


        self.norm_mean = nn.Parameter(torch.zeros(3))
        self.norm_std = nn.Parameter(torch.ones(3))
        self.norm = K.Normalize(mean=self.norm_mean, std=self.norm_std)
        # ＝ 幾何增強 ＝

        # ＝ 光度增強 ＝
        self.rc = LearnableContrast()
        
        # 按順序收好，方便迴圈處理
        self.aug_layers = nn.ModuleList([
            # self.norm,
            self.rc,
        ])

    def show(self):
        print("Augmentation Layers:")
        for aug in self.aug_layers:
            print(aug)

    def forward(self, x):
        for aug in self.aug_layers:
            x = aug.to(x.device)(x)
        return x
    
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