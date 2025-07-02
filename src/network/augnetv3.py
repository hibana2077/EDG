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
        self.st_2 = nn.Parameter(torch.tensor(2.))
        self.rs = K.RandomSharpness(saturation = (self.st_1,self.st_2),p=1.)
        
        # 按順序收好，方便迴圈處理
        self.aug_layers = nn.ModuleList([
            self.norm,
            self.rs,
        ])

    def forward(self, x):
        for aug in self.aug_layers:
            x = aug.to(x.device)(x)
        return x