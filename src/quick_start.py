"""
Quick Start Example: Using Contrastive Learning Pipeline
"""

import torch
import torch.nn as nn
import timm
import yaml
from torch.utils.data import DataLoader, TensorDataset

# Import our modules
from models.contrastive_model import ContrastiveModel, ContrastiveTrainer


def create_dummy_dataset(num_samples=100, num_classes=10, image_size=224):
    """Create a dummy dataset for demonstration"""
    # Generate random images and labels
    images = torch.randn(num_samples, 3, image_size, image_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    dataset = TensorDataset(images, labels)
    return dataset


def quick_start_example():
    """Quick start example"""
    print("🚀 Contrastive Learning Pipeline Quick Start")
    print("=" * 50)
    
    # 1. Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Create dummy data
    print("\n📊 Creating dummy dataset...")
    train_dataset = create_dummy_dataset(num_samples=64, num_classes=5)
    val_dataset = create_dummy_dataset(num_samples=32, num_classes=5)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # 3. Create model
    print("\n🏗️ Creating contrastive learning model...")
    backbone = timm.create_model('resnet18', pretrained=False, num_classes=0)
    
    model = ContrastiveModel(
        backbone=backbone,
        num_classes=5,
        projection_dim=64,
        hidden_dim=256,
        augnet_dim=224,
        augnet_heads=4,
        temperature=0.1,
        enable_infonce=True,
        infonce_feature_type="grad"  # Default to gradient features for compatibility
    ).to(device)
    
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Create trainer
    print("\n⚙️ Setting up trainer...")
    trainer = ContrastiveTrainer(
        model=model,
        augnet_lr=1e-4,
        model_lr=1e-3,
        weight_decay=1e-4,
        lambda_moment=1.0,
        gamma_div=1.0
    )
    
    # 5. Train for a few epochs
    print("\n🎯 Starting training...")
    num_epochs = 3
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_losses = []
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Training step
            metrics = trainer.train_step(images, labels)
            train_losses.append(metrics)
            
            if batch_idx == 0:  # Show details for the first batch only
                print(f"  Batch 1 - AugNet Loss: {metrics['augnet_loss']:.4f}, "
                      f"Contrastive: {metrics['contrastive_loss']:.4f}, "
                      f"Classification: {metrics['classification_loss']:.4f}")
        
        # Validation phase
        model.eval()
        val_losses = []
        
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            val_metrics = trainer.validate_step(images, labels)
            val_losses.append(val_metrics)
        
        # Calculate average losses
        avg_train_loss = sum(m['total_main_loss'] for m in train_losses) / len(train_losses)
        avg_val_loss = sum(m['val_total_loss'] for m in val_losses) / len(val_losses)
        avg_val_acc = sum(m['val_accuracy'] for m in val_losses) / len(val_losses)
        
        print(f"  Training loss: {avg_train_loss:.4f}, Validation loss: {avg_val_loss:.4f}, Validation accuracy: {avg_val_acc:.4f}")
    
    # 6. Test inference
    print("\n🔍 Testing inference...")
    model.eval()
    
    with torch.no_grad():
        # Take a test sample
        test_image, test_label = val_dataset[0]
        test_image = test_image.unsqueeze(0).to(device)  # Add batch dimension
        
        # Forward pass
        results = model(test_image)
        cls_pred1, cls_pred2 = results['cls_predictions']
        
        # Prediction results
        pred1 = torch.argmax(cls_pred1, dim=1).item()
        pred2 = torch.argmax(cls_pred2, dim=1).item()
        
        print(f"True label: {test_label.item()}")
        print(f"View 1 prediction: {pred1}")
        print(f"View 2 prediction: {pred2}")
        
        # Feature extraction
        f1, f2 = results['features']
        print(f"Feature vector size: {f1.shape[1]}")
    
    # 7. Test learnable augmentation
    print("\n🎨 Testing learnable augmentation...")
    with torch.no_grad():
        original_image = test_image
        aug1 = model.augnet(original_image)
        aug2 = model.augnet(original_image)
        
        # Calculate difference from original image
        diff1 = torch.abs(original_image - aug1).mean().item()
        diff2 = torch.abs(original_image - aug2).mean().item()
        
        print(f"Augmentation 1 vs original mean difference: {diff1:.4f}")
        print(f"Augmentation 2 vs original mean difference: {diff2:.4f}")
        print(f"Two augmentations are different: {not torch.equal(aug1, aug2)}")
    
    print("\n✅ Quick start example complete!")
    print("\n📋 Next steps:")
    print("1. Replace dummy data with your real dataset")
    print("2. Adjust model parameters and hyperparameters")
    print("3. Use the full train_contrastive.py for formal training")
    print("4. Use inference.py for inference and feature extraction")


if __name__ == "__main__":
    quick_start_example()
