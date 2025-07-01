import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.tiny_vit import TinyVitBlock

class AugNet(nn.Module):
    def __init__(self, dim=224, num_heads=8):
        super(AugNet, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.backbone = TinyVitBlock(dim=dim, num_heads=num_heads)

    def forward(self, x):
        return self.backbone(x)
    

if __name__ == "__main__":
    # Example usage
    model = AugNet()
    input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image
    output = model(input_tensor)
    print(output.shape)  # Output shape will depend on the TinyVitBlock implementation