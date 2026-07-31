"""
Comparison code for V1 and V2
Generates 5 plots:
  1. Accuracy Curves
  2. Learning Rate Decay
  3. Confusion Matrices
  4. ROC Curves
  5. Performance Comparison Table
"""

import os

import re

import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    classification_report, 
    accuracy_score,
    roc_auc_score
)

from dl_model import SETIEfficientNet as SETIEfficientNetV1
from dl_model_v2 import SETIEfficientNetV2
from dl_data_prep_v2 import SETIDatasetV2


# ==========================================
# 1. PARSE LOG FILES FOR ACCURACY & LR
# ==========================================
def parse_log_file(log_path):
    train_accs, valid_accs, lrs = [], [], []
    if not os.path.exists(log_path):
        print(f"Warning: Could not find log file {log_path}")
        return train_accs, valid_accs, lrs

    with open(log_path, "r") as f:
        for line in f:
            train_match = re.search(r"Train Acc:\s*([\d\.]+)%", line)
            valid_match = re.search(r"Valid Acc:\s*([\d\.]+)%", line)
            lr_match = re.search(r"(?:Head LR:|LR:)\s*([\d\.\-eE]+)", line)

            if train_match and valid_match:
                train_accs.append(float(train_match.group(1)))
                valid_accs.append(float(valid_match.group(1)))
                if lr_match:
                    lrs.append(float(lr_match.group(1)))
                else:
                    lrs.append(0.0)

    return train_accs, valid_accs, lrs


# ==========================================
# 2. RUN TEST INFERENCE FOR METRICS
# ==========================================
def sync_device(device):
    """Helper to ensure accurate time tracking on GPUs."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

def get_test_predictions(data_dir = "data/raw", v1_path = "models/seti_efficientnet_best.pth", v2_path = "models/seti_efficientnet_v2_best.pth"):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5], std = [0.5])
    ])

    test_dataset = SETIDatasetV2(data_dir, split = "test", transform = test_transform)
    test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)
    classes = test_dataset.classes

    # Load V1
    model_v1 = SETIEfficientNetV1(num_classes = len(classes)).to(device)
    model_v1.load_state_dict(torch.load(v1_path, map_location = device))
    model_v1.eval()

    # Load V2
    model_v2 = SETIEfficientNetV2(num_classes = len(classes)).to(device)
    model_v2.load_state_dict(torch.load(v2_path, map_location = device))
    model_v2.eval()

    y_true = []
    y_preds_v1, y_probs_v1 = [], []
    y_preds_v2, y_probs_v2 = [], []
    
    v1_time, v2_time = 0.0, 0.0
    total_images = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            total_images += inputs.size(0)

            # --- V1 Inference & Timing ---
            inputs_v1 = inputs.repeat(1, 3, 1, 1) if getattr(model_v1.backbone.features[0][0], 'in_channels', 1) == 3 else inputs
            
            sync_device(device)
            t0 = time.perf_counter()
            out_v1 = model_v1(inputs_v1)
            sync_device(device)
            v1_time += (time.perf_counter() - t0)

            probs_v1 = F.softmax(out_v1, dim = 1)
            _, preds_v1 = torch.max(out_v1, 1)

            # --- V2 Inference & Timing ---
            sync_device(device)
            t0 = time.perf_counter()
            out_v2 = model_v2(inputs)
            sync_device(device)
            v2_time += (time.perf_counter() - t0)

            probs_v2 = F.softmax(out_v2, dim = 1)
            _, preds_v2 = torch.max(out_v2, 1)

            y_true.extend(labels.cpu().numpy())
            y_preds_v1.extend(preds_v1.cpu().numpy())
            y_probs_v1.extend(probs_v1.cpu().numpy())
            y_preds_v2.extend(preds_v2.cpu().numpy())
            y_probs_v2.extend(probs_v2.cpu().numpy())

    return {
        "classes": classes,
        "y_true": np.array(y_true),
        "v1": {"preds": np.array(y_preds_v1), "probs": np.array(y_probs_v1), "ms_per_img": (v1_time / total_images) * 1000},
        "v2": {"preds": np.array(y_preds_v2), "probs": np.array(y_probs_v2), "ms_per_img": (v2_time / total_images) * 1000}
    }


# ==========================================
# 3. INDIVIDUAL PLOTTING FUNCTIONS
# ==========================================
def plot_accuracy_curves(train_v1, valid_v1, train_v2, valid_v2):
    fig, axes = plt.subplots(1, 2, figsize = (15, 6))
    fig.suptitle("Training Progress: V1 Baseline vs. V2 Optimized", fontsize = 16, fontweight = 'bold', y = 0.98)

    def draw_curve(ax, train, valid, title):
        epochs = list(range(1, len(train) + 1))
        ax.plot(epochs, train, label = 'Train Accuracy', color = '#1f77b4', linewidth = 2)
        ax.plot(epochs, valid, label = 'Validation Accuracy', color = '#ff7f0e' if 'V1' in title else '#2ca02c', linewidth = 2)
        
        if valid:
            best_ep = int(np.argmax(valid)) + 1
            best_acc = max(valid)
            ax.scatter(best_ep, best_acc, color = 'red', s = 100, zorder = 5)
            
            x_offset = 12 if best_ep < 70 else -30
            ax.annotate(f"Best Model\n({best_acc:.2f}% @ Ep {best_ep})", xy = (best_ep, best_acc),
                        xytext = (best_ep + x_offset, best_acc - 6),
                        arrowprops = dict(facecolor = 'black', edgecolor = 'black', shrink = 0.05, width = 2, headwidth = 8),
                        fontsize = 10, fontweight = 'bold', ha = 'left' if x_offset > 0 else 'right')

        ax.set_title(title, fontsize = 13, fontweight = 'bold')
        ax.set_xlabel("Epoch", fontsize = 11, fontweight = 'bold')
        ax.set_ylabel("Accuracy (%)", fontsize = 11, fontweight = 'bold')
        ax.grid(True, linestyle = '--', alpha = 0.4)
        ax.legend(loc = "lower right")

    draw_curve(axes[0], train_v1, valid_v1, "V1 Baseline: Accuracy Curves (Overfitting Gap)")
    draw_curve(axes[1], train_v2, valid_v2, "V2 Optimized: Accuracy Curves (Regularized)")

    plt.tight_layout()
    plt.savefig("plots/final_results/01_comparison_accuracy.png", dpi = 300, bbox_inches = 'tight')
    plt.close()


def plot_lr_decay(lrs_v1, lrs_v2):
    fig, axes = plt.subplots(1, 2, figsize = (15, 6))
    fig.suptitle("Learning Rate Schedules: V1 Baseline vs. V2 Optimized", fontsize = 16, fontweight = 'bold', y = 0.98)

    epochs_v1 = list(range(1, len(lrs_v1) + 1))
    epochs_v2 = list(range(1, len(lrs_v2) + 1))

    axes[0].plot(epochs_v1, lrs_v1, label = 'Cosine LR', color = '#1f77b4', linewidth = 2)
    axes[0].set_title("V1 Baseline: Learning Rate Decay", fontsize = 13, fontweight = 'bold')
    
    axes[1].plot(epochs_v2, lrs_v2, label = 'Warmup + Cosine LR', color = '#2ca02c', linewidth = 2)
    axes[1].set_title("V2 Optimized: Learning Rate Decay", fontsize = 13, fontweight = 'bold')

    for ax in axes:
        ax.set_xlabel("Epoch", fontsize = 11, fontweight = 'bold')
        ax.set_ylabel("Learning Rate", fontsize = 11, fontweight = 'bold')
        ax.grid(True, linestyle = '--', alpha = 0.4)
        ax.legend(loc = "upper right")

    plt.tight_layout()
    plt.savefig("plots/final_results/02_comparison_lr_decay.png", dpi = 300, bbox_inches = 'tight')
    plt.close()


def plot_confusion_matrices(y_true, preds_v1, preds_v2, classes):
    fig, axes = plt.subplots(1, 2, figsize = (18, 7))
    fig.suptitle("Test Set Confusion Matrices: V1 Baseline vs. V2 Optimized", fontsize = 16, fontweight = 'bold', y = 0.98)

    cm_v1 = confusion_matrix(y_true, preds_v1)
    cm_v2 = confusion_matrix(y_true, preds_v2)

    sns.heatmap(cm_v1, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = classes, yticklabels = classes, ax = axes[0], cbar = False, annot_kws = {"size": 11, "weight": "bold"})
    axes[0].set_title("V1 Baseline: Confusion Matrix", fontsize = 13, fontweight = 'bold')
    
    sns.heatmap(cm_v2, annot = True, fmt = 'd', cmap = 'Greens', xticklabels = classes, yticklabels = classes, ax = axes[1], cbar = False, annot_kws = {"size": 11, "weight": "bold"})
    axes[1].set_title("V2 Optimized: Confusion Matrix", fontsize = 13, fontweight = 'bold')

    for ax in axes:
        ax.set_xlabel("Predicted Signal Type", fontsize = 11, fontweight = 'bold')
        ax.set_ylabel("True Signal Type", fontsize = 11, fontweight = 'bold')
        ax.set_xticklabels(classes, rotation = 45, ha = 'right', rotation_mode = 'anchor', fontsize = 10, fontweight = 'bold')
        ax.set_yticklabels(classes, rotation = 0, fontsize = 10, fontweight = 'bold')

    plt.tight_layout()
    plt.savefig("plots/final_results/03_comparison_confusion_matrix.png", dpi = 300, bbox_inches = 'tight')
    plt.close()


def plot_roc_curves(y_true, probs_v1, probs_v2, classes):
    fig, axes = plt.subplots(1, 2, figsize = (16, 7))
    fig.suptitle("Test Set ROC Curves: V1 Baseline vs. V2 Optimized", fontsize = 16, fontweight = 'bold', y = 0.98)

    y_true_oh = np.eye(len(classes))[y_true]

    for i, cls_name in enumerate(classes):
        fpr_v1, tpr_v1, _ = roc_curve(y_true_oh[:, i], probs_v1[:, i])
        axes[0].plot(fpr_v1, tpr_v1, label = f'{cls_name} (AUC = {auc(fpr_v1, tpr_v1):.3f})')
        
        fpr_v2, tpr_v2, _ = roc_curve(y_true_oh[:, i], probs_v2[:, i])
        axes[1].plot(fpr_v2, tpr_v2, label = f'{cls_name} (AUC = {auc(fpr_v2, tpr_v2):.3f})')

    axes[0].set_title("V1 Baseline: ROC Curves", fontsize = 13, fontweight = 'bold')
    axes[1].set_title("V2 Optimized: ROC Curves", fontsize = 13, fontweight = 'bold')

    for ax in axes:
        ax.plot([0, 1], [0, 1], 'k--', alpha = 0.5)
        ax.set_xlabel("False Positive Rate", fontsize = 11, fontweight = 'bold')
        ax.set_ylabel("True Positive Rate", fontsize = 11, fontweight = 'bold')
        ax.grid(True, linestyle = '--', alpha = 0.4)
        ax.legend(fontsize = 9, loc = "lower right")

    plt.tight_layout()
    plt.savefig("plots/final_results/04_comparison_roc.png", dpi = 300, bbox_inches = 'tight')
    plt.close()


def plot_performance_table(y_true, v1_preds, v2_preds, v1_probs, v2_probs, classes, v1_ms, v2_ms):
    # 1. Base Metrics
    acc_v1 = accuracy_score(y_true, v1_preds) * 100
    acc_v2 = accuracy_score(y_true, v2_preds) * 100
    
    report_v1 = classification_report(y_true, v1_preds, target_names = classes, output_dict = True, zero_division = 0)
    report_v2 = classification_report(y_true, v2_preds, target_names = classes, output_dict = True, zero_division = 0)

    # 2. Advanced Metrics
    y_true_oh = np.eye(len(classes))[y_true]
    auc_v1 = roc_auc_score(y_true_oh, v1_probs, average = 'macro', multi_class = 'ovr')
    auc_v2 = roc_auc_score(y_true_oh, v2_probs, average = 'macro', multi_class = 'ovr')

    worst_f1_v1 = min([report_v1[cls]['f1-score'] for cls in classes])
    worst_f1_v2 = min([report_v2[cls]['f1-score'] for cls in classes])

    # 3. Build Table Data
    table_data = [
        ["Metric", "V1 Baseline", "V2 Optimized", "Delta (\u0394)"],

        ["Overall Accuracy", f"{acc_v1:.2f}%", f"{acc_v2:.2f}%", f"{(acc_v2 - acc_v1):+.2f}%"],

        ["Macro Precision", f"{report_v1['macro avg']['precision']:.4f}", f"{report_v2['macro avg']['precision']:.4f}", f"{(report_v2['macro avg']['precision'] - report_v1['macro avg']['precision']):+.4f}"],

        ["Macro Recall", f"{report_v1['macro avg']['recall']:.4f}", f"{report_v2['macro avg']['recall']:.4f}", f"{(report_v2['macro avg']['recall'] - report_v1['macro avg']['recall']):+.4f}"],

        ["Macro F1-Score", f"{report_v1['macro avg']['f1-score']:.4f}", f"{report_v2['macro avg']['f1-score']:.4f}", f"{(report_v2['macro avg']['f1-score'] - report_v1['macro avg']['f1-score']):+.4f}"],

        ["Macro AUC (Separability)", f"{auc_v1:.4f}", f"{auc_v2:.4f}", f"{(auc_v2 - auc_v1):+.4f}"],

        ["Worst-Class F1 (Hardest Signal)", f"{worst_f1_v1:.4f}", f"{worst_f1_v2:.4f}", f"{(worst_f1_v2 - worst_f1_v1):+.4f}"],

        ["Inference Time (ms per image)", f"{v1_ms:.2f} ms", f"{v2_ms:.2f} ms", f"{(v2_ms - v1_ms):+.2f} ms"]
    ]

    fig, ax = plt.subplots(figsize = (12, 4.5))
    ax.axis('off')
    
    table = ax.table(cellText = table_data, cellLoc = 'center', loc = 'center', colWidths = [0.40, 0.20, 0.20, 0.20])
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    table.scale(1.0, 1.9)

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight = 'bold', color = 'white')
            cell.set_facecolor('#4c72b0')
        elif i > 0 and j == 3:
            cell.set_text_props(weight='bold')
            val_str = table_data[i][j].replace('%', '').replace(' ms', '').replace('+', '')
            val = float(val_str)
            
            if "Inference" in table_data[i][0]:
                cell.set_facecolor('#d4edda' if val <= 0 else '#f8d7da')
            else:
                cell.set_facecolor('#d4edda' if val >= 0 else '#f8d7da')
        else:
            cell.set_facecolor('#f2f2f2' if i % 2 == 0 else 'white')
            if j == 0:
                cell.set_text_props(weight='bold')

    plt.title("TEST SET PERFORMANCE COMPARISON", fontsize = 16, fontweight = 'bold', pad = 20)
    plt.tight_layout()
    plt.savefig("plots/final_results/05_comparison_table.png", dpi = 300, bbox_inches = 'tight')
    plt.close()


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    os.makedirs("plots/final_results", exist_ok=True)
    
    print("Parsing logs and running test set evaluation...")
    train_v1, valid_v1, lrs_v1 = parse_log_file("models/training_log.txt")
    train_v2, valid_v2, lrs_v2 = parse_log_file("models/training_log_v2.txt")

    data = get_test_predictions()
    classes, y_true = data["classes"], data["y_true"]
    v1_preds, v1_probs, v1_ms = data["v1"]["preds"], data["v1"]["probs"], data["v1"]["ms_per_img"]
    v2_preds, v2_probs, v2_ms = data["v2"]["preds"], data["v2"]["probs"], data["v2"]["ms_per_img"]

    print("Generating 1/5: Accuracy Curves...")
    plot_accuracy_curves(train_v1, valid_v1, train_v2, valid_v2)

    print("Generating 2/5: Learning Rate Decay...")
    plot_lr_decay(lrs_v1, lrs_v2)

    print("Generating 3/5: Confusion Matrices...")
    plot_confusion_matrices(y_true, v1_preds, v2_preds, classes)

    print("Generating 4/5: ROC Curves...")
    plot_roc_curves(y_true, v1_probs, v2_probs, classes)

    print("Generating 5/5: Performance Table...")
    plot_performance_table(y_true, v1_preds, v2_preds, v1_probs, v2_probs, classes, v1_ms, v2_ms)

    print("\nAll 5 comparison files are generated and saved to the 'plots/final_results/' folder.")