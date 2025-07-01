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
        from ..dataset.Cotton80 import Cotton80Dataset
        
        # Create root directory if it doesn't exist
        os.makedirs(self.data_root, exist_ok=True)
        
        # Check if dataset exists, if not download it
        cotton_dir = os.path.join(self.data_root, 'COTTON')
        if not os.path.exists(cotton_dir):
            print(f"Cotton80 dataset not found at {self.data_root}. Downloading...")
            download = True
        else:
            download = False
        
        dataset = Cotton80Dataset(
            root=self.data_root,
            split=self.split,
            transform=None,  # We'll apply transforms in __getitem__
            download=download,
            zip_url="https://huggingface.co/datasets/hibana2077/Ultra-FGVC/resolve/main/Cotton80/Cotton80.zip?download=true"
        )
        
        # Extract image paths and labels
        data_paths = []
        labels = []
        
        cotton_dir = os.path.join(self.data_root, 'COTTON')
        images_dir = os.path.join(cotton_dir, 'images')
        
        for image_name, label in dataset.samples:
            image_path = os.path.join(images_dir, image_name)
            data_paths.append(image_path)
            labels.append(label)
        
        return data_paths, labels
    
    def _load_ip102(self):
        """Load IP102 dataset"""
        from ..dataset.IP102 import IP102Dataset
        
        # Create root directory if it doesn't exist
        os.makedirs(self.data_root, exist_ok=True)
        
        # Check if dataset exists, if not download it
        ip102_dir = os.path.join(self.data_root, 'ip102_v1.1')
        if not os.path.exists(ip102_dir):
            print(f"IP102 dataset not found at {self.data_root}. Downloading...")
            download = True
        else:
            download = False
        
        dataset = IP102Dataset(
            root=self.data_root,
            split=self.split,
            transform=None,  # We'll apply transforms in __getitem__
            download=download
        )
        
        # IP102Dataset already provides data (image paths) and targets
        return dataset.data, dataset.targets
    
    def _load_soylocal(self):
        """Load SoyLocal dataset"""
        from ..dataset.SoyLocal import SoyLocalDataset
        
        # Create root directory if it doesn't exist
        os.makedirs(self.data_root, exist_ok=True)
        
        # Check if dataset exists, if not download it
        soybean_dir = os.path.join(self.data_root, 'soybean200')
        if not os.path.exists(soybean_dir):
            print(f"SoyLocal dataset not found at {self.data_root}. Downloading...")
            download = True
        else:
            download = False
        
        dataset = SoyLocalDataset(
            root=self.data_root,
            split=self.split,
            transform=None,  # We'll apply transforms in __getitem__
            download=download
        )
        
        # Extract image paths and labels
        data_paths = []
        labels = []
        
        soybean_dir = os.path.join(self.data_root, 'soybean200')
        images_dir = os.path.join(soybean_dir, 'images')
        
        for filename, label in zip(dataset.samples, dataset.targets):
            image_path = os.path.join(images_dir, filename)
            data_paths.append(image_path)
            labels.append(label)
        
        return data_paths, labels
    
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
        from ..dataset.Cotton80 import Cotton80Dataset
        return Cotton80Dataset(**kwargs)
    elif dataset_name.lower() == 'ip102':
        from ..dataset.IP102 import IP102Dataset
        return IP102Dataset(**kwargs)
    elif dataset_name.lower() == 'soylocal':
        from ..dataset.SoyLocal import SoyLocalDataset
        return SoyLocalDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_dataloaders_from_existing(dataset_name: str, config: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders using existing dataset implementations
    """
    # Create root directory if it doesn't exist
    data_root = config.get('data_root', './data')
    os.makedirs(data_root, exist_ok=True)
    
    # Check if dataset exists and determine download need
    download_needed = False
    if dataset_name.lower() == 'cotton80':
        dataset_dir = os.path.join(data_root, 'COTTON')
        zip_url = "https://huggingface.co/datasets/hibana2077/Ultra-FGVC/resolve/main/Cotton80/Cotton80.zip?download=true"
    elif dataset_name.lower() == 'ip102':
        dataset_dir = os.path.join(data_root, 'ip102_v1.1')
        zip_url = None  # IP102Dataset has its own URL
    elif dataset_name.lower() == 'soylocal':
        dataset_dir = os.path.join(data_root, 'soybean200')
        zip_url = None  # SoyLocalDataset has its own URL
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if not os.path.exists(dataset_dir):
        print(f"{dataset_name} dataset not found at {data_root}. Will download automatically.")
        download_needed = True
    
    # Common transform for both train and val (minimal since AugNet handles augmentation)
    base_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset-specific parameters
    dataset_kwargs = {
        'root': data_root,
        'transform': base_transform,
        'download': download_needed or config.get('download', False)
    }
    
    # Add zip_url for Cotton80 if needed
    if dataset_name.lower() == 'cotton80' and zip_url:
        dataset_kwargs['zip_url'] = zip_url
    
    # Load train dataset
    train_dataset = load_existing_dataset(
        dataset_name,
        split='train',
        **dataset_kwargs
    )
    
    # Load validation dataset (don't download again)
    val_kwargs = {k: v for k, v in dataset_kwargs.items() if k != 'download'}
    val_kwargs['download'] = False
    
    val_dataset = load_existing_dataset(
        dataset_name,
        split='val',
        **val_kwargs
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


# Utility functions for automatic dataset management
def detect_dataset_type(data_root: str) -> str:
    """
    Automatically detect dataset type based on directory structure
    """
    if os.path.exists(os.path.join(data_root, 'COTTON')):
        return 'cotton80'
    elif os.path.exists(os.path.join(data_root, 'ip102_v1.1')):
        return 'ip102'
    elif os.path.exists(os.path.join(data_root, 'soybean200')):
        return 'soylocal'
    else:
        # Try to infer from path name
        data_root_lower = data_root.lower()
        if 'cotton' in data_root_lower:
            return 'cotton80'
        elif 'ip102' in data_root_lower:
            return 'ip102'
        elif 'soy' in data_root_lower:
            return 'soylocal'
        else:
            raise ValueError(f"Cannot automatically detect dataset type for {data_root}")


def ensure_dataset_exists(dataset_name: str, data_root: str) -> bool:
    """
    Ensure dataset exists, download if necessary
    Returns True if dataset was downloaded, False if already existed
    """
    os.makedirs(data_root, exist_ok=True)
    
    if dataset_name.lower() == 'cotton80':
        dataset_dir = os.path.join(data_root, 'COTTON')
        if not os.path.exists(dataset_dir):
            print(f"Downloading Cotton80 dataset to {data_root}...")
            from ..dataset.Cotton80 import Cotton80Dataset
            Cotton80Dataset(
                root=data_root,
                split='train',
                download=True,
                zip_url="https://huggingface.co/datasets/hibana2077/Ultra-FGVC/resolve/main/Cotton80/Cotton80.zip?download=true"
            )
            return True
    
    elif dataset_name.lower() == 'ip102':
        dataset_dir = os.path.join(data_root, 'ip102_v1.1')
        if not os.path.exists(dataset_dir):
            print(f"Downloading IP102 dataset to {data_root}...")
            from ..dataset.IP102 import IP102Dataset
            IP102Dataset(
                root=data_root,
                split='train',
                download=True
            )
            return True
    
    elif dataset_name.lower() == 'soylocal':
        dataset_dir = os.path.join(data_root, 'soybean200')
        if not os.path.exists(dataset_dir):
            print(f"Downloading SoyLocal dataset to {data_root}...")
            from ..dataset.SoyLocal import SoyLocalDataset
            SoyLocalDataset(
                root=data_root,
                split='train',
                download=True
            )
            return True
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return False  # Dataset already existed


def create_auto_dataloaders(data_root: str, config: dict, dataset_name: str = None) -> Tuple[DataLoader, DataLoader]:
    """
    Automatically create dataloaders with dataset detection and downloading
    
    Args:
        data_root: Root directory for dataset
        config: Configuration dictionary
        dataset_name: Optional dataset name, will auto-detect if None
    """
    # Auto-detect dataset if not specified
    if dataset_name is None:
        try:
            dataset_name = detect_dataset_type(data_root)
            print(f"Auto-detected dataset type: {dataset_name}")
        except ValueError:
            # If can't detect from existing files, try to infer from path
            dataset_name = detect_dataset_type(data_root)
    
    # Ensure dataset exists (download if needed)
    ensure_dataset_exists(dataset_name, data_root)
    
    # Update config with data_root
    config = config.copy()
    config['data_root'] = data_root
    
    # Create dataloaders
    return create_dataloaders_from_existing(dataset_name, config)
