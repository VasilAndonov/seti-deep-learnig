"""
Data Pipeline for Deep Learning (V2 Optimized).
"""

import os
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SETIDatasetV2(Dataset):
    def __init__(self, root_dir, split = "train", transform = None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))
        self.filepaths, self.labels = [], []

        for label, cls in enumerate(self.classes):
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.isdir(cls_dir): 
                continue
            
            for img_name in os.listdir(cls_dir):
                self.filepaths.append(os.path.join(cls_dir, img_name))
                self.labels.append(label)

    def __len__(self): 
        return len(self.filepaths)

    def __getitem__(self, idx):
        image = cv2.imread(self.filepaths[idx], cv2.IMREAD_GRAYSCALE)
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def get_dataloaders_v2(data_dir = "data/raw", batch_size = 32):
    print("Initializing V2 - PyTorch DataLoaders...")

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomVerticalFlip(p = 0.5),
        transforms.ColorJitter(brightness = 0.2, contrast = 0.2),
        transforms.RandomAffine(degrees = 0, translate = (0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5], std = [0.5]),
        transforms.RandomErasing(p = 0.25, scale = (0.02, 0.15), value = 0.0)
    ])

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean = [0.5], std = [0.5])
    ])

    train_dataset = SETIDatasetV2(data_dir, split = "train", transform = train_transform)
    valid_dataset = SETIDatasetV2(data_dir, split = "valid", transform = valid_transform)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 2)
    valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False, num_workers = 2)

    return train_loader, valid_loader, train_dataset.classes