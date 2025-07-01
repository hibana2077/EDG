import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from network.augnet import AugNet
from utils.contrastive_loss import InfoNCELoss, GradientHook


class ContrastiveModel(nn.Module):
    """
    Modified SimCLR with learnable augmentation network
    Architecture:
    x -> augnet(x) -> x1, x2 -> backbone -> f1, f2 -> hook -> g1, g2
    Loss: infonce(g1,g2) + 0.5 * ce(cls(f1), label) + 0.5 * ce(cls(f2), label)
    """
    
    def __init__(self, 
                 backbone: nn.Module,
                 num_classes: int,
                 projection_dim: int = 128,
                 hidden_dim: int = 2048,
                 augnet_dim: int = 224,
                 augnet_heads: int = 8,
                 temperature: float = 0.1):
        super().__init__()
        
        # Components
        self.augnet = AugNet(dim=augnet_dim, num_heads=augnet_heads)
        self.backbone = backbone
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_output = self.backbone(dummy_input)
            backbone_dim = backbone_output.view(backbone_output.size(0), -1).size(1)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # Classification head
        self.classification_head = nn.Linear(backbone_dim, num_classes)
        
        # Loss functions
        self.infonce_loss = InfoNCELoss(temperature=temperature)
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Gradient hook
        self.gradient_hook = GradientHook()
        
    def forward_backbone(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward through backbone and get both features and projections
        """
        features = self.backbone(x)
        features_flat = features.view(features.size(0), -1)
        
        # Get projections for contrastive learning
        projections = self.projection_head(features_flat)
        
        # Register hook for gradient features only if requires_grad
        if projections.requires_grad:
            projections.register_hook(self.gradient_hook.hook_fn)
        
        return features_flat, projections
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> dict:
        """
        Forward pass through the entire model
        """
        batch_size = x.size(0)
        
        # Generate two augmented views
        x1 = self.augnet(x)
        x2 = self.augnet(x)
        
        # Forward through backbone
        f1, proj1 = self.forward_backbone(x1)
        f2, proj2 = self.forward_backbone(x2)
        
        # Classification predictions
        cls_pred1 = self.classification_head(f1)
        cls_pred2 = self.classification_head(f2)
        
        results = {
            'projections': (proj1, proj2),
            'features': (f1, f2),
            'augmented_views': (x1, x2),
            'cls_predictions': (cls_pred1, cls_pred2)
        }
        
        if labels is not None:
            results['losses'] = self.compute_losses(proj1, proj2, cls_pred1, cls_pred2, labels)
        
        return results
    
    def compute_losses(self, proj1: torch.Tensor, proj2: torch.Tensor,
                      cls_pred1: torch.Tensor, cls_pred2: torch.Tensor,
                      labels: torch.Tensor) -> dict:
        """
        Compute all losses
        """
        # Validate labels are within valid range
        num_classes = cls_pred1.size(1)
        valid_mask = (labels >= 0) & (labels < num_classes)
        
        if not valid_mask.all():
            # Clamp invalid labels to valid range
            labels = torch.clamp(labels, 0, num_classes - 1)
        
        # Classification losses
        ce_loss1 = self.ce_loss(cls_pred1, labels)
        ce_loss2 = self.ce_loss(cls_pred2, labels)
        classification_loss = 0.5 * (ce_loss1 + ce_loss2)
        
        # Clamp projections to prevent NaN/Inf values
        proj1 = torch.clamp(proj1, min=-50, max=50)
        proj2 = torch.clamp(proj2, min=-50, max=50)
        
        # Replace NaN values with zeros
        proj1 = torch.where(torch.isnan(proj1), torch.zeros_like(proj1), proj1)
        proj2 = torch.where(torch.isnan(proj2), torch.zeros_like(proj2), proj2)
        
        # InfoNCE loss (will be computed on gradient features during backward)
        contrastive_loss = self.infonce_loss(proj1, proj2)
        
        return {
            'contrastive_loss': contrastive_loss,
            'classification_loss': classification_loss,
            'total_loss': contrastive_loss + classification_loss
        }


class ContrastiveTrainer:
    """
    Trainer for the contrastive model with separate optimization for AugNet
    """
    
    def __init__(self, 
                 model: ContrastiveModel,
                 augnet_lr: float = 1e-4,
                 model_lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 lambda_moment: float = 1.0,
                 gamma_div: float = 1.0):
        
        self.model = model
        
        # Separate optimizers for AugNet and main model
        self.augnet_optimizer = torch.optim.Adam(
            self.model.augnet.parameters(), 
            lr=augnet_lr, 
            weight_decay=weight_decay
        )
        
        main_model_params = [
            *self.model.backbone.parameters(),
            *self.model.projection_head.parameters(),
            *self.model.classification_head.parameters()
        ]
        self.model_optimizer = torch.optim.Adam(
            main_model_params,
            lr=model_lr,
            weight_decay=weight_decay
        )
        
        # Import MomentHoMDivLoss for AugNet training
        from utils.loss import MomentHoMDivLoss
        self.moment_loss = MomentHoMDivLoss(lambda_=lambda_moment, gamma_=gamma_div)
        
    def train_step(self, x: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Single training step with separate backward passes
        """
        self.model.train()
        
        # Step 1: Train AugNet with MomentHoMDivLoss
        self.augnet_optimizer.zero_grad()
        
        with torch.no_grad():
            # Generate augmented views
            x1 = self.model.augnet(x)
            x2 = self.model.augnet(x)
        
        # Recompute with gradients for AugNet training
        x1_grad = self.model.augnet(x)
        x2_grad = self.model.augnet(x)
        
        # Compute moment loss
        augnet_loss = self.moment_loss(x1_grad, x2_grad)
        augnet_loss.backward()
        self.augnet_optimizer.step()
        
        # Step 2: Train main model
        self.model_optimizer.zero_grad()
        self.model.gradient_hook.clear()
        
        # Forward pass through the entire model
        results = self.model(x, labels)
        losses = results['losses']
        
        # Backward pass for main model
        total_loss = losses['total_loss']
        total_loss.backward()
        self.model_optimizer.step()
        
        return {
            'augnet_loss': augnet_loss.item(),
            'contrastive_loss': losses['contrastive_loss'].item(),
            'classification_loss': losses['classification_loss'].item(),
            'total_main_loss': total_loss.item()
        }
    
    def validate_step(self, x: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Validation step
        """
        self.model.eval()
        
        with torch.no_grad():
            results = self.model(x, labels)
            losses = results['losses']
            
            # Compute accuracy
            cls_pred1, cls_pred2 = results['cls_predictions']
            pred1 = cls_pred1.argmax(dim=1)
            pred2 = cls_pred2.argmax(dim=1)
            
            acc1 = (pred1 == labels).float().mean()
            acc2 = (pred2 == labels).float().mean()
            avg_acc = (acc1 + acc2) / 2
            
        return {
            'val_contrastive_loss': losses['contrastive_loss'].item(),
            'val_classification_loss': losses['classification_loss'].item(),
            'val_total_loss': losses['total_loss'].item(),
            'val_accuracy': avg_acc.item()
        }
