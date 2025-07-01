#!/usr/bin/env python3
"""
Example usage of the InfoNCE enable/disable functionality

This script demonstrates how to train the model with and without InfoNCE loss.
"""

import os
import yaml
from train import train_contrastive_model, load_config
from utils.data_utils import create_contrastive_dataloaders

def train_with_infonce():
    """Train model with InfoNCE contrastive loss enabled"""
    print("=" * 60)
    print("TRAINING WITH INFONCE CONTRASTIVE LOSS ENABLED")
    print("=" * 60)
    
    # Load config with InfoNCE enabled
    config = load_config('config_contrastive.yaml')
    
    # Ensure InfoNCE is enabled
    config['enable_infonce'] = True
    config['num_epochs'] = 2  # Short training for demo
    config['checkpoint_dir'] = 'checkpoints_with_infonce'
    
    print(f"InfoNCE enabled: {config['enable_infonce']}")
    
    # Create dummy dataloaders for demo (replace with your actual data)
    train_loader, val_loader = create_contrastive_dataloaders(config)
    
    # Start training
    train_contrastive_model(config, train_loader, val_loader)
    print("Training with InfoNCE completed!\\n")

def train_without_infonce():
    """Train model with InfoNCE contrastive loss disabled"""
    print("=" * 60)
    print("TRAINING WITH INFONCE CONTRASTIVE LOSS DISABLED")
    print("=" * 60)
    
    # Load config without InfoNCE
    config = load_config('config_no_infonce.yaml')
    
    # Ensure InfoNCE is disabled
    config['enable_infonce'] = False
    config['num_epochs'] = 2  # Short training for demo
    config['checkpoint_dir'] = 'checkpoints_without_infonce'
    
    print(f"InfoNCE enabled: {config['enable_infonce']}")
    
    # Create dummy dataloaders for demo (replace with your actual data)
    train_loader, val_loader = create_contrastive_dataloaders(config)
    
    # Start training
    train_contrastive_model(config, train_loader, val_loader)
    print("Training without InfoNCE completed!\\n")

def compare_configurations():
    """Compare the two configurations"""
    print("=" * 60)
    print("CONFIGURATION COMPARISON")
    print("=" * 60)
    
    # Load both configs
    config_with = load_config('config_contrastive.yaml')
    config_without = load_config('config_no_infonce.yaml')
    
    print("With InfoNCE:")
    print(f"  - enable_infonce: {config_with.get('enable_infonce', True)}")
    print(f"  - Loss components: InfoNCE + Classification")
    print(f"  - Training mode: Contrastive Learning")
    
    print("\\nWithout InfoNCE:")
    print(f"  - enable_infonce: {config_without.get('enable_infonce', True)}")
    print(f"  - Loss components: Classification only")
    print(f"  - Training mode: Supervised Learning")
    
    print("\\nBoth configurations:")
    print(f"  - Use the same backbone: {config_with['backbone']}")
    print(f"  - Same number of classes: {config_with['num_classes']}")
    print(f"  - Same batch size: {config_with['batch_size']}")
    print(f"  - AugNet is still trained with MomentHoMDivLoss in both cases")

if __name__ == "__main__":
    # Show configuration comparison
    compare_configurations()
    
    print("\\n" + "=" * 60)
    print("To run actual training, uncomment the lines below:")
    print("=" * 60)
    print("# Uncomment to train with InfoNCE enabled:")
    print("# train_with_infonce()")
    print()
    print("# Uncomment to train with InfoNCE disabled:")
    print("# train_without_infonce()")
    
    # Uncomment to actually run training:
    # train_with_infonce()
    # train_without_infonce()
    
    print("\\nDone! Use the configurations as needed for your experiments.")
