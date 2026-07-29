"""
dl_data_prep.py
Deep Learning PyTorch Data Pipeline for SETI Classification

Configures PyTorch ImageFolder datasets and DataLoaders.
Implements physics-aware augmentations suitable for astronomical spectrograms.
"""

import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# --- CONFIGURATION ---
RAW_DATA_DIR = "data/raw/" 
IMG_SIZE_DL = (256, 256) # Power of 2, optimal for CNN feature maps

def get_dl_dataloaders(batch_size = 32):
    """
    Creates PyTorch DataLoaders reading natively from train/valid/test directories.
    Applies Physics-Aware Data Augmentation (disables flips to preserve Doppler drift).
    """
    print(f">>> Initializing Deep Learning DataLoaders (Resolution: {IMG_SIZE_DL[0]}x{IMG_SIZE_DL[1]})...")
    
    # Safe augmentations: Small rotations and noise simulation via ColorJitter
    train_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE_DL),
        transforms.RandomRotation(degrees = 5), 
        transforms.ColorJitter(brightness = 0.2, contrast = 0.2), 
        transforms.ToTensor(),
        # ImageNet standardization (required for transfer learning backbones like ResNet)
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) 
    ])
    
    # Validation and Test sets don't have to be augmented, only resized and normalized
    val_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE_DL),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    
    # PyTorch automatically maps subfolder names to class labels
    train_dataset = datasets.ImageFolder(root = os.path.join(RAW_DATA_DIR, 'train'), transform = train_transforms)
    val_dataset = datasets.ImageFolder(root = os.path.join(RAW_DATA_DIR, 'valid'), transform = val_transforms)
    test_dataset = datasets.ImageFolder(root = os.path.join(RAW_DATA_DIR, 'test'), transform = val_transforms)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 2, pin_memory = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = 2, pin_memory = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers=2, pin_memory = True)
    
    print(f"DL Pipeline Ready: Train ({len(train_dataset)}), Val ({len(val_dataset)}), Test ({len(test_dataset)})")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test script execution
    train_dl, val_dl, test_dl = get_dl_dataloaders()
    images, labels = next(iter(train_dl))
    print(f"Tensor batch shape: {images.shape} | Labels batch shape: {labels.shape}")