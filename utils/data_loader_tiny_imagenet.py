import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path


class TinyImageNetDataset(datasets.ImageFolder):
    """Custom Tiny ImageNet Dataset"""
    def __init__(self, root, split='train', transform=None, download=False):
        self.root = root
        self.split = split
        self.transform = transform
        
        # Set the correct path based on split
        if split == 'train':
            data_path = os.path.join(root, 'tiny-imagenet-200', 'train')
        elif split == 'val':
            data_path = os.path.join(root, 'tiny-imagenet-200', 'val')
        elif split == 'test':
            data_path = os.path.join(root, 'tiny-imagenet-200', 'test')
        else:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")
        
        # Check if dataset exists
        if not os.path.exists(data_path):
            if download:
                print(f"Tiny ImageNet not found at {data_path}")
                print("Please download Tiny ImageNet from: http://cs231n.stanford.edu/tiny-imagenet-200.zip")
                print(f"Extract it to: {root}")
                raise FileNotFoundError(f"Dataset not found at {data_path}")
            else:
                raise FileNotFoundError(f"Dataset not found at {data_path}. Set download=True to see instructions.")
        
        # For validation set, we need to reorganize the structure
        if split == 'val':
            self._prepare_val_folder(data_path)
            data_path = os.path.join(data_path, 'images')
        
        super(TinyImageNetDataset, self).__init__(data_path, transform=transform)
    
    def _prepare_val_folder(self, val_dir):
        """
        Reorganize validation folder to have the same structure as training folder
        """
        val_img_dir = os.path.join(val_dir, 'images')
        
        # Check if already organized
        if os.path.isdir(val_img_dir):
            subdirs = [d for d in os.listdir(val_img_dir) if os.path.isdir(os.path.join(val_img_dir, d))]
            if len(subdirs) > 1:  # Already organized
                return
        
        # Read val annotations
        val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
        if not os.path.exists(val_annotations_file):
            return
        
        # Create class folders
        with open(val_annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                img_name, class_id = parts[0], parts[1]
                
                # Create class directory if it doesn't exist
                class_dir = os.path.join(val_img_dir, class_id)
                os.makedirs(class_dir, exist_ok=True)
                
                # Move image to class directory
                src = os.path.join(val_img_dir, img_name)
                dst = os.path.join(class_dir, img_name)
                if os.path.exists(src) and not os.path.exists(dst):
                    os.rename(src, dst)


def data_loader_tiny_imagenet(data_dir, val_split=0.1, batch_size=128, num_workers=4):
    """
    Create data loaders for Tiny ImageNet dataset
    
    Args:
        data_dir: Root directory containing 'tiny-imagenet-200' folder
        val_split: Fraction of training data to use for validation (if using train/val split)
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Normalization values for Tiny ImageNet (ImageNet stats work well)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    # Training transforms with data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Validation/Test transforms without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load datasets
    try:
        train_dataset = TinyImageNetDataset(
            root=data_dir, 
            split='train', 
            transform=transform_train,
            download=True
        )
        
        # Use the official validation set as test set
        test_dataset = TinyImageNetDataset(
            root=data_dir,
            split='val',
            transform=transform_test,
            download=True
        )
        
        # Split training data for validation
        train_size = int((1 - val_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Dataset sizes:")
        print(f"  Training: {len(train_dataset)}")
        print(f"  Validation: {len(val_dataset)}")
        print(f"  Test: {len(test_dataset)}")
        
    except FileNotFoundError as e:
        print(f"\n{'='*60}")
        print("ERROR: Tiny ImageNet dataset not found!")
        print(f"{'='*60}")
        print("\nTo download Tiny ImageNet:")
        print("1. Download from: http://cs231n.stanford.edu/tiny-imagenet-200.zip")
        print(f"2. Extract to: {data_dir}")
        print("3. Ensure the structure is:")
        print(f"   {data_dir}/tiny-imagenet-200/train/")
        print(f"   {data_dir}/tiny-imagenet-200/val/")
        print(f"   {data_dir}/tiny-imagenet-200/test/")
        print(f"\n{'='*60}\n")
        raise e
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the data loader
    print("Testing Tiny ImageNet data loader...")
    try:
        train_loader, val_loader, test_loader = data_loader_tiny_imagenet(
            data_dir='./data',
            batch_size=32,
            num_workers=2
        )
        
        # Get a batch
        images, labels = next(iter(train_loader))
        print(f"\nBatch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Number of classes: {max(labels).item() + 1}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        
    except Exception as e:
        print(f"Error: {e}")
