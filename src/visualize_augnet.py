import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import argparse
import yaml
import random
from tqdm import tqdm

from models.contrastive_model import ContrastiveModel
from train import create_backbone, load_config, set_seed
from utils.data_utils import create_dataloaders_from_existing


def tensor_to_pil(img_tensor):
    """Convert a tensor to a PIL image for visualization"""
    # First, denormalize the tensor
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = img_tensor * std + mean
    
    # Convert to PIL image
    img_tensor = img_tensor.clamp(0, 1)
    img_array = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    return img


def visualize_augnet_examples(model, val_loader, num_examples=4, output_path='augnet_visualization.png', device='cuda'):
    """
    Select random images from validation set and visualize the original vs AugNet's augmentations
    
    Args:
        model: Trained ContrastiveModel
        val_loader: DataLoader for validation set
        num_examples: Number of examples to visualize (default: 4)
        output_path: Path to save the visualization
        device: Device to run the model on
    """
    model.eval()
    all_images = []
    all_labels = []
    
    # Collect images from validation set
    for images, labels in tqdm(val_loader, desc="Collecting validation images"):
        all_images.append(images)
        all_labels.append(labels)
        if len(all_images) * images.size(0) >= num_examples * 10:  # Get more than we need to select from
            break
    
    # Concatenate collected tensors
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Select random examples
    indices = torch.randperm(len(all_images))[:num_examples]
    selected_images = all_images[indices].to(device)
    selected_labels = all_labels[indices].to(device)
    
    # Process through AugNet (generate 2 augmented versions for each image)
    with torch.no_grad():
        augmented_images1 = model.augnet(selected_images)
        augmented_images2 = model.augnet(selected_images)  # Generate a second augmentation
    
    # Create visualization grid
    # For each original image, show original and two augmented versions
    vis_tensors = []
    for i in range(num_examples):
        vis_tensors.extend([selected_images[i], augmented_images1[i], augmented_images2[i]])
    
    # Make grid
    nrow = 3  # Original + 2 augmentations
    grid = make_grid(vis_tensors, nrow=nrow, normalize=True, padding=10)
    
    # Create a figure with titles and better formatting
    plt.figure(figsize=(12, 3 * num_examples))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    
    # Add titles for each row
    for i in range(num_examples):
        plt.text(nrow/2 * grid.shape[2] // 2, 
                 i * (grid.shape[1] // num_examples) + 20, 
                 f"Example {i+1} (Class {selected_labels[i].item()})", 
                 ha='center', fontsize=14, color='white',
                 bbox=dict(boxstyle='round,pad=0.5', fc='rgba(0,0,0,0.5)', ec='none'))
        
        # Add column labels for the first row only
        if i == 0:
            plt.text(grid.shape[2] // 6, 10, "Original", ha='center', fontsize=12, 
                     color='white', bbox=dict(boxstyle='round,pad=0.3', fc='rgba(0,0,0,0.5)', ec='none'))
            plt.text(grid.shape[2] // 2, 10, "AugNet #1", ha='center', fontsize=12, 
                     color='white', bbox=dict(boxstyle='round,pad=0.3', fc='rgba(0,0,0,0.5)', ec='none'))
            plt.text(5 * grid.shape[2] // 6, 10, "AugNet #2", ha='center', fontsize=12, 
                     color='white', bbox=dict(boxstyle='round,pad=0.3', fc='rgba(0,0,0,0.5)', ec='none'))
    
    plt.axis('off')
    plt.title(f"AugNet Visualization - Original vs Augmented Images", fontsize=16)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved visualization to {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Visualize AugNet Examples')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', default='config_contrastive.yaml', help='Path to config file')
    parser.add_argument('--output', default='augnet_visualization.png', help='Output image path')
    parser.add_argument('--num_examples', type=int, default=4, help='Number of examples to visualize')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Use CUDA if available
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load configuration
    config = load_config(args.config)
    
    # Create dataloaders
    try:
        dataset_name = config.get('dataset_name', 'Cotton80')
        _, val_loader = create_dataloaders_from_existing(dataset_name, config)
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        return
    
    # Create backbone
    backbone = create_backbone(
        model_name=config.get('backbone', 'resnet50'),
        pretrained=False  # We'll load from checkpoint
    )
    
    # Create contrastive model
    model = ContrastiveModel(
        backbone=backbone,
        num_classes=config['num_classes'],
        projection_dim=config.get('projection_dim', 128),
        hidden_dim=config.get('hidden_dim', 2048),
        augnet_dim=config.get('augnet_dim', 224),
        augnet_heads=config.get('augnet_heads', 8),
        temperature=config.get('temperature', 0.1),
        enable_infonce=config.get('enable_infonce', True),
        infonce_feature_type=config.get('infonce_feature_type', 'grad')
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Generate visualization
    visualize_augnet_examples(
        model=model,
        val_loader=val_loader,
        num_examples=args.num_examples,
        output_path=args.output,
        device=device
    )


if __name__ == "__main__":
    main()
