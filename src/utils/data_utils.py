import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from typing import Tuple, Optional


class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning
    Note: Since we use learnable augmentation (AugNet), we only apply basic transforms here
    """
    
    def __init__(self, 
                 data_root: str, 
                 split: str = 'train',
                 image_size: int = 224,
                 transform: Optional[transforms.Compose] = None):
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        
        # Basic transforms (minimal since AugNet handles augmentation)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Load dataset specific implementation
        self.data, self.labels = self._load_dataset()
        
    def _load_dataset(self):
        """
        Override this method for your specific dataset
        Should return (data_paths, labels) or (data_tensors, labels)
        """
        # Placeholder implementation - replace with your dataset loading logic
        if 'Cotton80' in self.data_root:
            return self._load_cotton80()
        elif 'IP102' in self.data_root:
            return self._load_ip102()
        elif 'SoyLocal' in self.data_root:
            return self._load_soylocal()
        else:
            raise NotImplementedError(f"Dataset loading for {self.data_root} not implemented")
    
    def _load_cotton80(self):
        """Load Cotton80 dataset"""
        # Implement Cotton80 loading logic
        # This should match the implementation in dataset/Cotton80.py
        pass
    
    def _load_ip102(self):
        """Load IP102 dataset"""
        # Implement IP102 loading logic
        # This should match the implementation in dataset/IP102.py
        pass
    
    def _load_soylocal(self):
        """Load SoyLocal dataset"""
        # Implement SoyLocal loading logic
        # This should match the implementation in dataset/SoyLocal.py
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        # Load image
        if isinstance(self.data[idx], str):
            # If data contains paths
            image = Image.open(self.data[idx]).convert('RGB')
            image = self.transform(image)
        else:
            # If data contains tensors
            image = self.data[idx]
            if self.transform:
                image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


def create_contrastive_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for contrastive learning
    """
    
    # Training dataset with minimal augmentation
    train_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation dataset
    val_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ContrastiveDataset(
        data_root=config['data_root'],
        split=config['train_split'],
        image_size=config['image_size'],
        transform=train_transform
    )
    
    val_dataset = ContrastiveDataset(
        data_root=config['data_root'],
        split=config['val_split'],
        image_size=config['image_size'],
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # Important for contrastive learning
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


# Integration with existing dataset implementations
def load_existing_dataset(dataset_name: str, **kwargs):
    """
    Load existing dataset implementations from dataset/ folder
    """
    if dataset_name.lower() == 'cotton80':
        from dataset.Cotton80 import Cotton80Dataset  # Assuming class name
        return Cotton80Dataset(**kwargs)
    elif dataset_name.lower() == 'ip102':
        from dataset.IP102 import IP102Dataset  # Assuming class name
        return IP102Dataset(**kwargs)
    elif dataset_name.lower() == 'soylocal':
        from dataset.SoyLocal import SoyLocalDataset  # Assuming class name
        return SoyLocalDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_dataloaders_from_existing(dataset_name: str, config: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders using existing dataset implementations
    """
    # Load train dataset
    train_dataset = load_existing_dataset(
        dataset_name,
        split='train',
        transform=transforms.Compose([
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    )
    
    # Load validation dataset
    val_dataset = load_existing_dataset(
        dataset_name,
        split='val',
        transform=transforms.Compose([
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader
