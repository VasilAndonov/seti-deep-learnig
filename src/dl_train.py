"""
Model Training Loop.
100-epoch training run and hardware acceleration (CUDA or MPS).
"""

import os

import time

import torch
import torch.nn as nn
import torch.optim as optim

from dl_data_prep import get_dataloaders
from dl_model import SETIEfficientNet

def train_model():
    # Hardware Configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")

    # Training Parameters 
    EPOCHS = 100 
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3

    train_loader, test_loader, classes = get_dataloaders(batch_size = BATCH_SIZE)
    model = SETIEfficientNet(num_classes = len(classes)).to(device)

    # Label smoothing helps prevent overconfidence on noisy static background
    criterion = nn.CrossEntropyLoss(label_smoothing = 0.1) 
    
    # AdamW with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay = 1e-4)
    
    # Cosine annealing scaled over 100 epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS, eta_min = 1e-6)

    # CUDA automatic mixed precision
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    print(f"\nStarting 100-Epoch Training...")
    best_test_acc = 0.0

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
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = correct_train / total_train
        train_loss = running_loss / total_train

        # --- Validation Phase ---
        model.eval()
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                    
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_acc = correct_test / total_test

        # Step the learning rate scheduler
        scheduler.step()
        
        # Calculate time in minutes and seconds
        epoch_time = time.time() - start_time
        mins = int(epoch_time // 60)
        secs = int(epoch_time % 60)
        time_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"

        print(f"Epoch [{epoch+1:03d}/{EPOCHS}] | LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Time: {time_str} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
        
        # Save the model whenever a new best validation accuracy is reached
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/seti_efficientnet_best.pth")

    print(f"\nTraining Complete. Best Test Accuracy: {best_test_acc*100:.2f}%")
    print("Best model checkpoint saved to models/seti_efficientnet_best.pth")

if __name__ == "__main__":
    train_model()