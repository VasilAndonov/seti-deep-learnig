import os

import re

import numpy as np
import matplotlib.pyplot as plt

def parse_and_plot():
    if os.path.exists("models/training_log.txt"):
        log_file = "models/training_log.txt"

    # 1. Extract metrics using Regex
    with open(log_file, "r") as f:
        log_data = f.read()

    # Find all occurrences of "Train Acc: XX.XX%" and "Test Acc: YY.YY%"
    train_accs = [float(x) for x in re.findall(r"Train Acc: (\d+\.\d+)%", log_data)]
    test_accs = [float(x) for x in re.findall(r"Test Acc: (\d+\.\d+)%", log_data)]

    epochs = list(range(1, len(train_accs) + 1))
    
    # Recreate the Cosine Annealing LR Schedule used during training
    lr_schedule = [0.001 * (0.5 * (1 + np.cos(np.pi * e / 100))) for e in range(len(epochs))]

    # 2. Create Side-by-Side Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))
    fig.suptitle("Training Progress & Overfitting Analysis", fontsize = 18, y = 0.98)

    # Panel 1: Accuracy
    ax1.plot(epochs, train_accs, label = "Train Accuracy", color = "#1f77b4", linewidth = 2)
    ax1.plot(epochs, test_accs, label = "Test Accuracy", color = "#ff7f0e", linewidth = 2)

    # Find the best epoch
    best_acc = max(test_accs)
    best_epoch = test_accs.index(best_acc) + 1
    
    ax1.scatter([best_epoch], [best_acc], color = "red", s = 100, zorder = 5, label = f"Best Model ({best_acc}% @ Ep {best_epoch})")
    ax1.annotate(f"Best Weights Saved\n({best_acc}%)", 
                 xy = (best_epoch, best_acc), 
                 xytext = (best_epoch + 10, best_acc - 10),
                 arrowprops = dict(facecolor = 'black', shrink = 0.05, width = 1, headwidth = 6),
                 fontsize = 10, fontweight = 'bold')

    ax1.set_title("Accuracy Curves", fontsize = 14, pad = 10)
    ax1.set_xlabel("Epoch", fontsize = 12, fontweight = "bold")
    ax1.set_ylabel("Accuracy (%)", fontsize = 12, fontweight = "bold")
    ax1.grid(True, linestyle = "--", alpha = 0.5)
    ax1.legend(loc = "lower right", fontsize = 11)
    ax1.set_ylim(70, 102)

    # Panel 2: Learning Rate Schedule
    ax2.plot(epochs, lr_schedule, label = "Cosine LR", color = "#2ca02c", linewidth = 2.5)
    ax2.set_title("Learning Rate Decay Schedule", fontsize = 14, pad = 10)
    ax2.set_xlabel("Epoch", fontsize = 12, fontweight = "bold")
    ax2.set_ylabel("Learning Rate", fontsize = 12, fontweight = "bold")
    ax2.grid(True, linestyle = "--", alpha = 0.5)
    ax2.legend(loc = "upper right", fontsize = 11)

    plt.tight_layout(pad = 2.0)

    # 3. Save Plot
    save_path = "plots/05_learning_curves.png"
    plt.savefig(save_path, dpi = 300, bbox_inches = "tight")
    print(f"Extracted {len(epochs)} epochs. Plot saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    parse_and_plot()