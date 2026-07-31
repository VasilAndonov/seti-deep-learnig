"""
Data Augmentation Visualization code (V2).
Generates a 4x4 grid showing  what the V2  sees during training.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dl_data_prep_v2 import SETIDatasetV2

def generate_augmented_batch_plot(data_dir = "data/raw"):
    print("Initializing V2 Augmentations...")
    
    # 1. Replicate exact V2 training transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomVerticalFlip(p = 0.5),
        transforms.ColorJitter(brightness = 0.2, contrast = 0.2),
        transforms.RandomAffine(degrees = 0, translate = (0.05, 0.05)), 
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5], std = [0.5]),
        transforms.RandomErasing(p = 0.75, scale = (0.02, 0.15), value = 0.0) # Black cutout patches
    ])

    # 2. Load Dataset and fetch one batch
    dataset = SETIDatasetV2(data_dir, split = "train", transform = train_transform)
    loader = DataLoader(dataset, batch_size = 16, shuffle = True)
    
    print("Fetching augmented training batch...")
    images, labels = next(iter(loader))
    classes = dataset.classes

    # 3. Create 4x4 Plot Grid
    fig, axes = plt.subplots(4, 4, figsize = (14, 14))
    fig.suptitle("What the V2 Sees: Augmented Batch", 
                 fontsize = 18, fontweight = 'bold', y = 0.95)

    for i, ax in enumerate(axes.flat):
        # Squeeze out channel dimension (1, 224, 224) -> (224, 224)
        img = images[i].squeeze().numpy()
        
        # Un-normalize from [-1, 1] back to [0, 1] for matplotlib display
        img = (img * 0.5) + 0.5
        img = np.clip(img, 0, 1)
        
        ax.imshow(img, cmap = 'gray')
        ax.set_title(f"Class: {classes[labels[i]]}", fontsize = 11, fontweight = 'bold')
        
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # 4. Save the plot
    save_path = "plots/final_results/00_v2_augmented_samples.png"

    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')

    plt.close()
    
    print(f"Augmentation batch visualization is saved to: {save_path}")

if __name__ == "__main__":
    generate_augmented_batch_plot()