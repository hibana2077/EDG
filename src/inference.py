import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import yaml
import argparse
import numpy as np

from models.contrastive_model import ContrastiveModel
from train_contrastive import create_backbone, load_config


class ContrastiveInference:
    """
    Inference class for the trained contrastive model
    """
    
    def __init__(self, checkpoint_path: str, config_path: str, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Create model
        backbone = create_backbone(
            model_name=self.config.get('backbone', 'resnet50'),
            pretrained=False  # We'll load from checkpoint
        )
        
        self.model = ContrastiveModel(
            backbone=backbone,
            num_classes=self.config['num_classes'],
            projection_dim=self.config.get('projection_dim', 128),
            hidden_dim=self.config.get('hidden_dim', 2048),
            augnet_dim=self.config.get('augnet_dim', 224),
            augnet_heads=self.config.get('augnet_heads', 8),
            temperature=self.config.get('temperature', 0.1)
        ).to(self.device)
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        self.model.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess single image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor.to(self.device)
    
    def predict(self, image_path: str, return_features=False):
        """
        Predict class and optionally return features
        """
        with torch.no_grad():
            # Preprocess image
            image = self.preprocess_image(image_path)
            
            # Forward pass
            results = self.model(image)
            cls_pred1, cls_pred2 = results['cls_predictions']
            
            # Average predictions from both augmented views
            avg_pred = (cls_pred1 + cls_pred2) / 2
            probabilities = F.softmax(avg_pred, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
            
            result = {
                'predicted_class': predicted_class.cpu().item(),
                'confidence': confidence.cpu().item(),
                'probabilities': probabilities.cpu().numpy()[0]
            }
            
            if return_features:
                f1, f2 = results['features']
                avg_features = (f1 + f2) / 2
                result['features'] = avg_features.cpu().numpy()[0]
            
            return result
    
    def extract_features(self, image_path: str):
        """Extract features from image"""
        with torch.no_grad():
            image = self.preprocess_image(image_path)
            results = self.model(image)
            f1, f2 = results['features']
            
            # Return both individual and averaged features
            return {
                'features_view1': f1.cpu().numpy()[0],
                'features_view2': f2.cpu().numpy()[0],
                'avg_features': ((f1 + f2) / 2).cpu().numpy()[0]
            }
    
    def generate_augmentations(self, image_path: str, num_augs=5):
        """
        Generate multiple augmentations using the learnable AugNet
        """
        with torch.no_grad():
            image = self.preprocess_image(image_path)
            
            augmentations = []
            for _ in range(num_augs):
                aug_image = self.model.augnet(image)
                augmentations.append(aug_image.cpu())
            
            return torch.cat(augmentations, dim=0)  # (num_augs, C, H, W)
    
    def compute_similarity(self, image_path1: str, image_path2: str):
        """
        Compute cosine similarity between two images using learned features
        """
        with torch.no_grad():
            # Extract features
            features1 = self.extract_features(image_path1)['avg_features']
            features2 = self.extract_features(image_path2)['avg_features']
            
            # Compute cosine similarity
            features1 = torch.tensor(features1)
            features2 = torch.tensor(features2)
            
            features1_norm = F.normalize(features1, dim=0)
            features2_norm = F.normalize(features2, dim=0)
            
            similarity = torch.dot(features1_norm, features2_norm).item()
            
            return similarity


def main():
    parser = argparse.ArgumentParser(description='Contrastive Model Inference')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--mode', choices=['predict', 'features', 'augment'], 
                       default='predict', help='Inference mode')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--num_augs', type=int, default=5, help='Number of augmentations to generate')
    
    args = parser.parse_args()
    
    # Create inference object
    inference = ContrastiveInference(args.checkpoint, args.config, args.device)
    
    if args.mode == 'predict':
        result = inference.predict(args.image, return_features=True)
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Top 3 probabilities: {np.argsort(result['probabilities'])[-3:][::-1]}")
        
    elif args.mode == 'features':
        features = inference.extract_features(args.image)
        print(f"Feature shape: {features['avg_features'].shape}")
        print(f"Feature norm: {np.linalg.norm(features['avg_features']):.4f}")
        
    elif args.mode == 'augment':
        augmentations = inference.generate_augmentations(args.image, args.num_augs)
        print(f"Generated {augmentations.shape[0]} augmentations")
        print(f"Augmentation shape: {augmentations.shape}")
        
        # Save augmentations (optional)
        from torchvision.utils import save_image
        save_image(augmentations, f'augmentations_{args.num_augs}.png', normalize=True)
        print(f"Saved augmentations to augmentations_{args.num_augs}.png")


if __name__ == "__main__":
    main()
