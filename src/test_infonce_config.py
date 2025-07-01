#!/usr/bin/env python3
"""
Test script to demonstrate InfoNCE loss enable/disable functionality
"""

import torch
import yaml
from models.contrastive_model import ContrastiveModel
from train import create_backbone

def test_infonce_configuration():
    """Test that InfoNCE loss can be enabled/disabled"""
    
    # Create test configuration
    config_with_infonce = {
        'backbone': 'resnet18',
        'pretrained': False,
        'num_classes': 10,
        'projection_dim': 64,
        'hidden_dim': 128,
        'augnet_dim': 224,
        'augnet_heads': 4,
        'temperature': 0.1,
        'enable_infonce': True
    }
    
    config_without_infonce = config_with_infonce.copy()
    config_without_infonce['enable_infonce'] = False
    
    # Create backbone
    backbone = create_backbone(
        model_name=config_with_infonce['backbone'],
        pretrained=config_with_infonce['pretrained']
    )
    
    # Test model with InfoNCE enabled
    print("Testing model with InfoNCE enabled...")
    model_with_infonce = ContrastiveModel(
        backbone=backbone,
        num_classes=config_with_infonce['num_classes'],
        projection_dim=config_with_infonce['projection_dim'],
        hidden_dim=config_with_infonce['hidden_dim'],
        augnet_dim=config_with_infonce['augnet_dim'],
        augnet_heads=config_with_infonce['augnet_heads'],
        temperature=config_with_infonce['temperature'],
        enable_infonce=config_with_infonce['enable_infonce']
    )
    
    # Test model with InfoNCE disabled
    print("Testing model with InfoNCE disabled...")
    model_without_infonce = ContrastiveModel(
        backbone=create_backbone(config_without_infonce['backbone'], config_without_infonce['pretrained']),
        num_classes=config_without_infonce['num_classes'],
        projection_dim=config_without_infonce['projection_dim'],
        hidden_dim=config_without_infonce['hidden_dim'],
        augnet_dim=config_without_infonce['augnet_dim'],
        augnet_heads=config_without_infonce['augnet_heads'],
        temperature=config_without_infonce['temperature'],
        enable_infonce=config_without_infonce['enable_infonce']
    )
    
    # Create dummy data
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, config_with_infonce['num_classes'], (batch_size,))
    
    # Test forward pass with InfoNCE enabled
    print("\\nForward pass with InfoNCE enabled:")
    model_with_infonce.eval()
    with torch.no_grad():
        results_with = model_with_infonce(x, labels)
        losses_with = results_with['losses']
        print(f"  Contrastive Loss: {losses_with['contrastive_loss'].item():.4f}")
        print(f"  Classification Loss: {losses_with['classification_loss'].item():.4f}")
        print(f"  Total Loss: {losses_with['total_loss'].item():.4f}")
    
    # Test forward pass with InfoNCE disabled
    print("\\nForward pass with InfoNCE disabled:")
    model_without_infonce.eval()
    with torch.no_grad():
        results_without = model_without_infonce(x, labels)
        losses_without = results_without['losses']
        print(f"  Contrastive Loss: {losses_without['contrastive_loss'].item():.4f}")
        print(f"  Classification Loss: {losses_without['classification_loss'].item():.4f}")
        print(f"  Total Loss: {losses_without['total_loss'].item():.4f}")
    
    # Verify that contrastive loss is 0 when disabled
    assert losses_without['contrastive_loss'].item() == 0.0, "Contrastive loss should be 0 when disabled"
    assert losses_with['contrastive_loss'].item() > 0.0, "Contrastive loss should be > 0 when enabled"
    
    print("\\n✓ All tests passed! InfoNCE loss can be successfully enabled/disabled.")
    print("✓ When disabled, contrastive loss = 0 and only classification loss is used.")
    print("✓ When enabled, both contrastive and classification losses are computed.")


if __name__ == "__main__":
    test_infonce_configuration()
