import os

import sys
sys.path.append("src")

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from dl_data_prep import get_dataloaders
from dl_model import SETIEfficientNet

def plot_roc_pr_curves():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Generating ROC and PR curves on device: {device}")

    # 1. Load Data and Best Saved Model
    _, test_loader, classes = get_dataloaders(batch_size = 32)
    n_classes = len(classes)
    
    model = SETIEfficientNet(num_classes = n_classes).to(device)
    model.load_state_dict(torch.load("models/seti_efficientnet_best.pth", map_location = device))
    model.eval()

    # 2. Extract Softmax Probabilities & Ground Truth
    all_probs = []
    all_labels = []

    print("Running inference to collect class probabilities...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim = 1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    y_score = np.concatenate(all_probs, axis = 0)
    y_true = np.concatenate(all_labels, axis = 0)

    # Binarize labels for One-vs-Rest multi-class evaluation
    y_true_bin = label_binarize(y_true, classes = list(range(n_classes)))

    # Color map for 7 classes
    colors = plt.cm.get_cmap('Set1', n_classes)

    # 3. Create Side-by-Side Plot
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize = (18, 7))
    fig.suptitle("Multi-Class Diagnostic Performance: ROC and PR Curves", fontsize = 18, y = 0.98)

    # Panel 1: ROC curves
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color = colors(i), lw = 2,
                    label = f"{classes[i]} (AUC = {roc_auc:.3f})")

    ax_roc.plot([0, 1], [0, 1], 'k--', lw = 1.5, label = 'Random Chance (AUC = 0.500)')
    ax_roc.set_xlim([-0.02, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("False Positive Rate", fontsize = 12, fontweight = "bold")
    ax_roc.set_ylabel("True Positive Rate", fontsize = 12, fontweight = "bold")
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize = 14, pad = 10)
    ax_roc.grid(True, linestyle = "--", alpha = 0.5)
    ax_roc.legend(loc = "lower right", fontsize = 9)

    # Panel 2: Precison Recall (PR) curves
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        ap_score = average_precision_score(y_true_bin[:, i], y_score[:, i])
        ax_pr.plot(recall, precision, color = colors(i), lw = 2,
                   label = f"{classes[i]} (AP = {ap_score:.3f})")

    ax_pr.set_xlim([0.0, 1.02])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel("Recall", fontsize = 12, fontweight = "bold")
    ax_pr.set_ylabel("Precision", fontsize = 12, fontweight = "bold")
    ax_pr.set_title("Precision Recall (PR) Curves", fontsize = 14, pad = 10)
    ax_pr.grid(True, linestyle = "--", alpha = 0.5)
    ax_pr.legend(loc = "lower left", fontsize = 9)

    plt.tight_layout(pad = 2.0)

    # 4. Save plot
    save_path = "plots/06_roc_pr_curves.png"
    plt.savefig(save_path, dpi = 300, bbox_inches = "tight")
    print(f"ROC and PR Curves plot saved to: {save_path}")

    plt.show()

if __name__ == "__main__":
    plot_roc_pr_curves()