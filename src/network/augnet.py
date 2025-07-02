import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.tiny_vit import TinyVitBlock
from timm.models.convnext import ConvNeXtBlock

class AugNet(nn.Module):
    def __init__(self, dim=224, num_heads=8):
        super(AugNet, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        # self.backbone1 = TinyVitBlock(dim=dim, num_heads=num_heads)
        self.backbone1 = ConvNeXtBlock(in_chs=3)
        # self.backbone2 = TinyVitBlock(dim=dim, num_heads=num_heads)
        self.backbone2 = ConvNeXtBlock(in_chs=3)

    def forward(self, x):
        x = self.backbone1(x)
        with torch.no_grad():
            # Apply Monte Carlo Gaussian noise
            noise = torch.randn_like(x) * 0.1
        x = x + noise
        x = self.backbone2(x)
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