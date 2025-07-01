import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from models.contrastive_model import ContrastiveModel, ContrastiveTrainer
from utils.data_utils import create_contrastive_dataloaders, create_dataloaders_from_existing


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_file='training.log'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_backbone(model_name='resnet50', pretrained=True):
    """Create backbone model"""
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    return model


def train_contrastive_model(config, train_loader, val_loader):
    """
    Main training function for contrastive learning
    """
    logger = setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Create backbone
    backbone = create_backbone(
        model_name=config.get('backbone', 'resnet50'),
        pretrained=config.get('pretrained', True)
    )
    
    # Create contrastive model
    model = ContrastiveModel(
        backbone=backbone,
        num_classes=config['num_classes'],
        projection_dim=config.get('projection_dim', 128),
        hidden_dim=config.get('hidden_dim', 2048),
        augnet_dim=config.get('augnet_dim', 224),
        augnet_heads=config.get('augnet_heads', 8),
        temperature=config.get('temperature', 0.1)
    ).to(device)
    
    # Create trainer
    trainer = ContrastiveTrainer(
        model=model,
        augnet_lr=config.get('augnet_lr', 1e-4),
        model_lr=config.get('model_lr', 1e-3),
        weight_decay=config.get('weight_decay', 1e-4),
        lambda_moment=config.get('lambda_moment', 1.0),
        gamma_div=config.get('gamma_div', 1.0)
    )
    
    # Training loop
    num_epochs = config.get('num_epochs', 100)
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_metrics = train_epoch(trainer, train_loader, device, logger)
        
        # Validation phase
        model.eval()
        val_metrics = validate_epoch(trainer, val_loader, device, logger)
        
        # Log metrics
        logger.info(f"Train - AugNet Loss: {train_metrics['avg_augnet_loss']:.4f}, "
                   f"Contrastive Loss: {train_metrics['avg_contrastive_loss']:.4f}, "
                   f"Classification Loss: {train_metrics['avg_classification_loss']:.4f}"
                   f", Total Main Loss: {train_metrics['avg_total_main_loss']:.4f}")
        
        logger.info(f"Val - Total Loss: {val_metrics['avg_total_loss']:.4f}, "
                   f"Accuracy: {val_metrics['avg_accuracy']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.get('save_interval', 10) == 0:
            save_checkpoint(model, trainer, epoch, config.get('checkpoint_dir', 'checkpoints'))


def train_epoch(trainer, dataloader, device, logger):
    """Training for one epoch"""
    total_metrics = {
        'augnet_loss': 0.0,
        'contrastive_loss': 0.0,
        'classification_loss': 0.0,
        'total_main_loss': 0.0,
        'train_accuracy': 0.0
    }
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        # Training step
        metrics = trainer.train_step(images, labels)
        
        # Accumulate metrics
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'AugNet': f"{metrics['augnet_loss']:.4f}",
            'Contrastive': f"{metrics['contrastive_loss']:.4f}",
            'Classification': f"{metrics['classification_loss']:.4f}",
            'Train Acc': f"{metrics['train_accuracy']:.4f}"
        })
    
    # Average metrics
    avg_metrics = {f'avg_{key}': value / num_batches for key, value in total_metrics.items()}
    return avg_metrics


def validate_epoch(trainer, dataloader, device, logger):
    """Validation for one epoch"""
    total_metrics = {
        'contrastive_loss': 0.0,
        'classification_loss': 0.0,
        'total_loss': 0.0,
        'accuracy': 0.0
    }
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Validation")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        # Validation step
        metrics = trainer.validate_step(images, labels)
        
        # Accumulate metrics
        total_metrics['contrastive_loss'] += metrics['val_contrastive_loss']
        total_metrics['classification_loss'] += metrics['val_classification_loss']
        total_metrics['total_loss'] += metrics['val_total_loss']
        total_metrics['accuracy'] += metrics['val_accuracy']
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f"{metrics['val_total_loss']:.4f}",
            'Acc': f"{metrics['val_accuracy']:.4f}"
        })
    
    # Average metrics
    avg_metrics = {f'avg_{key}': value / num_batches for key, value in total_metrics.items()}
    return avg_metrics


def save_checkpoint(model, trainer, epoch, checkpoint_dir):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'augnet_optimizer_state_dict': trainer.augnet_optimizer.state_dict(),
        'model_optimizer_state_dict': trainer.model_optimizer.state_dict(),
    }, checkpoint_path)
    
    logging.info(f"Checkpoint saved to {checkpoint_path}")


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    # Load configuration
    config = load_config('config_contrastive.yaml')
    
    # Create data loaders
    try:
        # Try to use existing dataset implementations
        dataset_name = config.get('dataset_name', 'cotton80')  # default to cotton80
        train_loader, val_loader = create_dataloaders_from_existing(dataset_name, config)
    except:
        # Fallback to generic dataloader
        train_loader, val_loader = create_contrastive_dataloaders(config)
    
    # Start training
    train_contrastive_model(config, train_loader, val_loader)
    
    print("Training completed successfully!")