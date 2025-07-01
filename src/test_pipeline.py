import torch
import torch.nn as nn
import timm
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from models.contrastive_model import ContrastiveModel, ContrastiveTrainer
from utils.contrastive_loss import InfoNCELoss, GradientHook
from utils.loss import MomentHoMDivLoss
from network.augnet import AugNet


def test_augnet():
    """Test AugNet functionality"""
    print("Testing AugNet...")
    
    augnet = AugNet(dim=224, num_heads=8)
    x = torch.randn(2, 3, 224, 224)
    
    try:
        x1 = augnet(x)
        x2 = augnet(x)
        print(f"✓ AugNet forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {x1.shape}")
        print(f"  Outputs are different: {not torch.equal(x1, x2)}")
    except Exception as e:
        print(f"✗ AugNet test failed: {e}")
        return False
    
    return True


def test_moment_loss():
    """Test MomentHoMDivLoss"""
    print("\nTesting MomentHoMDivLoss...")
    
    loss_fn = MomentHoMDivLoss(lambda_=1.0, gamma_=1.0)
    x1 = torch.randn(2, 3, 64, 64)
    x2 = torch.randn(2, 3, 64, 64)
    
    try:
        loss = loss_fn(x1, x2)
        print(f"✓ MomentHoMDivLoss computation successful")
        print(f"  Loss value: {loss.item():.4f}")
        print(f"  Loss requires grad: {loss.requires_grad}")
    except Exception as e:
        print(f"✗ MomentHoMDivLoss test failed: {e}")
        return False
    
    return True


def test_infonce_loss():
    """Test InfoNCE Loss"""
    print("\nTesting InfoNCE Loss...")
    
    loss_fn = InfoNCELoss(temperature=0.1)
    features1 = torch.randn(4, 128)
    features2 = torch.randn(4, 128)
    
    try:
        loss = loss_fn(features1, features2)
        print(f"✓ InfoNCE Loss computation successful")
        print(f"  Loss value: {loss.item():.4f}")
    except Exception as e:
        print(f"✗ InfoNCE Loss test failed: {e}")
        return False
    
    return True


def test_gradient_hook():
    """Test Gradient Hook"""
    print("\nTesting Gradient Hook...")
    
    hook = GradientHook()
    x = torch.randn(2, 128, requires_grad=True)
    
    try:
        hook.register_hook(x)
        loss = x.sum()
        loss.backward()
        
        print(f"✓ Gradient Hook registration successful")
        print(f"  Captured gradients: {len(hook.gradient_features)}")
    except Exception as e:
        print(f"✗ Gradient Hook test failed: {e}")
        return False
    
    return True


def test_contrastive_model():
    """Test full ContrastiveModel"""
    print("\nTesting ContrastiveModel...")
    
    # Create a simple backbone
    backbone = timm.create_model('resnet18', pretrained=False, num_classes=0)
    
    try:
        model = ContrastiveModel(
            backbone=backbone,
            num_classes=10,
            projection_dim=128,
            hidden_dim=512,
            augnet_dim=224,
            augnet_heads=8,
            temperature=0.1
        )
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        labels = torch.randint(0, 10, (2,))
        
        results = model(x, labels)
        
        print(f"✓ ContrastiveModel forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Projection shapes: {[p.shape for p in results['projections']]}")
        print(f"  Feature shapes: {[f.shape for f in results['features']]}")
        print(f"  Classification shapes: {[c.shape for c in results['cls_predictions']]}")
        print(f"  Losses: {list(results['losses'].keys())}")
        
    except Exception as e:
        print(f"✗ ContrastiveModel test failed: {e}")
        return False
    
    return True


def test_trainer():
    """Test ContrastiveTrainer"""
    print("\nTesting ContrastiveTrainer...")
    
    # Create a simple model
    backbone = timm.create_model('resnet18', pretrained=False, num_classes=0)
    model = ContrastiveModel(
        backbone=backbone,
        num_classes=10,
        projection_dim=64,  # Smaller for testing
        hidden_dim=256,
        augnet_dim=224,
        augnet_heads=4,
        temperature=0.1
    )
    
    try:
        trainer = ContrastiveTrainer(
            model=model,
            augnet_lr=1e-4,
            model_lr=1e-3,
            weight_decay=1e-4,
            lambda_moment=1.0,
            gamma_div=1.0
        )
        
        # Test training step
        x = torch.randn(2, 3, 224, 224)
        labels = torch.randint(0, 10, (2,))
        
        metrics = trainer.train_step(x, labels)
        
        print(f"✓ ContrastiveTrainer training step successful")
        print(f"  Training metrics: {list(metrics.keys())}")
        print(f"  AugNet loss: {metrics['augnet_loss']:.4f}")
        print(f"  Contrastive loss: {metrics['contrastive_loss']:.4f}")
        print(f"  Classification loss: {metrics['classification_loss']:.4f}")
        
        # Test validation step
        val_metrics = trainer.validate_step(x, labels)
        print(f"✓ ContrastiveTrainer validation step successful")
        print(f"  Validation accuracy: {val_metrics['val_accuracy']:.4f}")
        
    except Exception as e:
        print(f"✗ ContrastiveTrainer test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests"""
    print("Running Contrastive Learning Pipeline Tests")
    print("=" * 50)
    
    tests = [
        test_augnet,
        test_moment_loss,
        test_infonce_loss,
        test_gradient_hook,
        test_contrastive_model,
        test_trainer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The pipeline is ready to use.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    main()
