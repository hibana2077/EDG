import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for contrastive learning on gradient features
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features1, features2):
        """
        features1, features2: (B, D) normalized gradient features
        """
        batch_size = features1.size(0)
        
        # Normalize features
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        
        # Concatenate features
        features = torch.cat([features1, features2], dim=0)  # (2B, D)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature  # (2B, 2B)
        
        # Create labels for positive pairs
        labels = torch.cat([torch.arange(batch_size) + batch_size, 
                           torch.arange(batch_size)], dim=0)
        labels = labels.to(features.device)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(features.device)
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        
        # Compute InfoNCE loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


class GradientHook:
    """
    Hook to capture gradient features during backward pass
    """
    def __init__(self):
        self.gradient_features = []
        
    def hook_fn(self, grad):
        self.gradient_features.append(grad.clone())
        return grad
    
    def register_hook(self, tensor):
        tensor.register_hook(self.hook_fn)
    
    def get_gradient_features(self):
        if len(self.gradient_features) >= 2:
            return self.gradient_features[-2], self.gradient_features[-1]
        return None, None
    
    def clear(self):
        self.gradient_features.clear()
