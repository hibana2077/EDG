import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for contrastive learning on either image features or gradient features
    
    Feature Types:
    - 'image': Uses the projection head output features directly for contrastive learning.
               This computes similarity between the learned feature representations.
    - 'grad': Uses gradient features captured during backward pass for contrastive learning.
              This computes similarity between the gradients of the features, which can
              capture different aspects of the learning dynamics.
    """
    def __init__(self, temperature=0.1, feature_type="grad"):
        super().__init__()
        self.temperature = temperature
        self.feature_type = feature_type  # "image" or "grad"
    
    def forward(self, features1, features2):
        """
        features1, features2: (B, D) features (either image projections or gradient features)
        For 'image' type: directly use the provided features
        For 'grad' type: features will be replaced by gradient features during backward pass
        """
        batch_size = features1.size(0)
        
        if self.feature_type == "image":
            # Use image features directly
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
            
        elif self.feature_type == "grad":
            # For gradient features, we compute the loss but the actual features
            # will be replaced by gradient features during the backward pass
            # This is the original implementation for gradient-based InfoNCE
            
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
        else:
            raise ValueError(f"Invalid feature_type: {self.feature_type}. Must be 'image' or 'grad'")


class GradientHook:
    """
    Hook to capture gradient features during backward pass
    Only used when infonce_feature_type is set to "grad"
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
