import os

import sys
sys.path.append("src")

import numpy as np
import matplotlib.pyplot as plt

import torch

from dl_data_prep import get_dataloaders
from dl_model import SETIEfficientNet

def visualize_errors():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running  Error Analysis on device: {device}")
    
    # 1. Load Data and Model
    _, test_loader, classes = get_dataloaders(batch_size = 32)
    model = SETIEfficientNet(num_classes = len(classes)).to(device)
    model.load_state_dict(torch.load("models/seti_efficientnet_best.pth", map_location = device))
    model.eval()

    seen_error_types = set()
    diverse_errors = []
    fallback_errors = []

    print("Scanning test set for model misclassifications...")
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(len(labels)):
                if labels[i] != predicted[i]:
                    t_class = classes[labels[i].item()]
                    p_class = classes[predicted[i].item()]
                    error_pair = (t_class, p_class)
                    
                    if error_pair not in seen_error_types:
                        seen_error_types.add(error_pair)
                        diverse_errors.append((inputs[i].cpu(), t_class, p_class))
                    else:
                        fallback_errors.append((inputs[i].cpu(), t_class, p_class))
                        
            if len(diverse_errors) >= 9:
                break
                
    while len(diverse_errors) < 9 and len(fallback_errors) > 0:
        diverse_errors.append(fallback_errors.pop(0))

    misclassified_images = [e[0] for e in diverse_errors]
    misclassified_trues = [e[1] for e in diverse_errors]
    misclassified_preds = [e[2] for e in diverse_errors]
                
    # 3. Plotting the Grid 
    fig, axes = plt.subplots(3, 3, figsize = (15, 14)) 
    fig.suptitle("Error Analysis: Misclassifications", fontsize = 20, y = 0.98)
    
    for i, ax in enumerate(axes.flat):
        if i < len(misclassified_images):
            img = misclassified_images[i].numpy().squeeze()
            img = img * 0.5 + 0.5 
            
            # Applied colormap for better visability
            ax.imshow(img, cmap='viridis')
            
            title_text = f"True: {misclassified_trues[i]}\nPred: {misclassified_preds[i]}"
            ax.set_title(title_text, fontsize = 12, fontweight = 'bold', color = 'darkred', pad = 10)
            ax.axis('off')
        else:
            ax.axis('off')
            
    plt.tight_layout(pad = 3.0, w_pad = 2.0, h_pad = 2.0)
    
    # 4. Save the plot
    save_path = "plots/03_error_analysis_grid.png"
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    print(f"Error grid saved to: {save_path}")

    plt.show()

if __name__ == "__main__":
    visualize_errors()