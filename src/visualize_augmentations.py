import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
sys.path.append("src")

from dl_data_prep import get_dataloaders

def visualize_augmentations():
    print("Loading augmented training data...")
    # Grab a small batch of 16 images from the training loader
    train_loader, _, classes = get_dataloaders(data_dir = "data/raw", batch_size = 16)
    
    # Extract one batch
    images, labels = next(iter(train_loader))
    
    # Set up a 4x4 grid for plotting
    fig, axes = plt.subplots(4, 4, figsize = (12, 12))
    fig.suptitle("What the Neural Network Sees: Augmented Training Batch", fontsize = 18, y = 1.02)
    
    for i, ax in enumerate(axes.flat):
        # The images are currently normalized between [-1, 1]. 
        # We need to un-normalize them to [0, 1] for matplotlib to display them properly.
        img = images[i].numpy().squeeze()  # Remove channel dimension
        img = img * 0.5 + 0.5              # Un-normalize
        
        # Display the image
        ax.imshow(img, cmap = 'gray')
        ax.set_title(f"Class: {classes[labels[i].item()]}", fontsize = 12, fontweight = 'bold')
        ax.axis('off')
        
    plt.tight_layout()
    
    # Save the plot to the plots folder
    save_path = "plots/01_augmented_training_samples.png"
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    print(f"Plot saved to: {save_path}")

    plt.show()

if __name__ == "__main__":
    visualize_augmentations()