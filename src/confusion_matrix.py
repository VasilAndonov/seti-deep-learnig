import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from sklearn.metrics import confusion_matrix, classification_report


sys.path.append("src")
from dl_data_prep import get_dataloaders
from dl_model import SETIEfficientNet

def run_phase1_evaluation():
    # 1. Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # 2. Load Data and Model
    print("Loading validation data and model weights...")
    _, valid_loader, classes = get_dataloaders(batch_size = 32)
    
    model = SETIEfficientNet(num_classes = len(classes)).to(device)
    
    # Load the best weights that are saved during training
    weights_path = "models/seti_efficientnet_best.pth"
    model.load_state_dict(torch.load(weights_path, map_location = device))
    model.eval()

    # 3. Gather Predictions
    print("Running inference on the validation set...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Generate Classification Report
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    report = classification_report(all_labels, all_preds, target_names = classes)
    print(report)

    # 5. Plot Confusion Matrix
    print("Generating Confusion Matrix plot...")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize = (10, 8))
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues',
                xticklabels = classes, yticklabels = classes,
                annot_kws = {"size": 12})
    
    plt.title('Confusion Matrix: EfficientNet-B0 (94.43% Valid Acc)', fontsize = 16, pad = 15)
    plt.ylabel('True Signal Type', fontsize = 12, fontweight = 'bold')
    plt.xlabel('Predicted Signal Type', fontsize = 12, fontweight = 'bold')
    plt.xticks(rotation = 45, ha = 'right', rotation_mode = 'anchor')
    plt.yticks(rotation = 0)
    plt.tight_layout()

    # 6. Save the plot
    save_path = "plots/02_confusion_matrix.png"
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    print(f"Plot saved to: {save_path}")

    plt.show()

if __name__ == "__main__":
    run_phase1_evaluation()