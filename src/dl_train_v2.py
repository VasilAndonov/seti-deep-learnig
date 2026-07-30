"""
Model Training Loop (V2 Optimized).
Features Differential Learning Rates, Linear Warmup + Cosine Scheduler, and Gradient Clipping.
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from dl_data_prep_v2 import get_dataloaders_v2
from dl_model_v2 import SETIEfficientNetV2

def train_model_v2():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training V2 Model on device: {device}")

    EPOCHS = 100 
    BATCH_SIZE = 32
    WARMUP_EPOCHS = 5

    train_loader, valid_loader, classes = get_dataloaders_v2(batch_size = BATCH_SIZE)
    model = SETIEfficientNetV2(num_classes = len(classes)).to(device)

    # 1. Differential Learning Rates: Lower LR for backbone features, higher LR for classifier head
    optimizer = optim.AdamW([
        {'params': model.backbone.features.parameters(), 'lr': 1e-4},
        {'params': model.backbone.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

    # 2. Schedulers: 5-Epoch Warmup followed by Cosine Annealing
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.1, total_iters = WARMUP_EPOCHS)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS - WARMUP_EPOCHS, eta_min = 1e-6)

    criterion = nn.CrossEntropyLoss(label_smoothing = 0.1)
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    print(f"\nStarting V2 100-Epoch Training...")
    best_valid_acc = 0.0

    log_path = "models/training_log_v2.txt"

    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                
                # Gradient Clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
                
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = correct_train / total_train

        # --- Validation Phase ---
        model.eval()
        correct_valid, total_valid = 0, 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                    
                _, predicted = torch.max(outputs, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels).sum().item()

        valid_acc = correct_valid / total_valid

        # Step LR Schedulers
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
            current_lr = warmup_scheduler.get_last_lr()[1]
        else:
            cosine_scheduler.step()
            current_lr = cosine_scheduler.get_last_lr()[1]

        epoch_time = time.time() - start_time
        mins = int(epoch_time // 60)
        secs = int(epoch_time % 60)
        time_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"

        log_line = (f"Epoch [{epoch+1:03d}/{EPOCHS}] | Head LR: {current_lr:.6f} | "
                    f"Time: {time_str} | Train Acc: {train_acc*100:.2f}% | Valid Acc: {valid_acc*100:.2f}%\n")
        
        print(log_line.strip())

        with open(log_path, "a") as f:
            f.write(log_line)
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), "models/seti_efficientnet_v2_best.pth")

    print(f"\nV2 Training Complete. Best Valid Accuracy: {best_valid_acc*100:.2f}%")
    print(f"V2 model saved to models/seti_efficientnet_v2_best.pth")
    print(f"Training log successfully saved to {log_path}")

if __name__ == "__main__":
    train_model_v2()