import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K
from timm.models.tiny_vit import TinyVitBlock
from timm.models.convnext import ConvNeXtBlock

class AugNet(nn.Module):
    def __init__(self, dim: int = 224, num_heads:int=8):
        super().__init__()


        self.norm_mean = nn.Parameter(torch.zeros(3))
        self.norm_std = nn.Parameter(torch.ones(3))
        self.norm = K.Normalize(mean=self.norm_mean, std=self.norm_std)
        # ＝ 幾何增強 ＝

        # ＝ 光度增強 ＝
        self.st_1 = nn.Parameter(torch.tensor(0.5))
        self.rs = K.RandomSharpness(self.st_1,p=1.)
        
        # 按順序收好，方便迴圈處理
        self.aug_layers = nn.ModuleList([
            self.norm,
            self.rs,
        ])

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