import os
import argparse
import torch
import yaml

from train import create_backbone, load_config
from models.contrastive_model import ContrastiveModel
from utils.data_utils import create_dataloaders_from_existing
from visualize_augnet import visualize_augnet_examples


def main():
    parser = argparse.ArgumentParser(description="Generate AugNet visualizations")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="config_contrastive.yaml", help="Path to config file")
    parser.add_argument("--output", default="augnet_visualization.png", help="Output path for visualization")
    parser.add_argument("--num_examples", type=int, default=4, help="Number of examples to visualize")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} does not exist")
        return

    # Load config
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    try:
        dataset_name = config.get("dataset_name", "Cotton80")
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
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
        
    # Generate visualization
    print(f"Generating visualization with {args.num_examples} examples...")
    try:
        output_path = visualize_augnet_examples(
            model=model,
            val_loader=val_loader,
            num_examples=args.num_examples,
            output_path=args.output,
            device=device
        )
        print(f"Visualization saved to {output_path}")
    except Exception as e:
        import traceback
        print(f"Error generating visualization: {str(e)}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
