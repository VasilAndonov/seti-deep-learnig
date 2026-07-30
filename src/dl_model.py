"""
DL Architecture for SETI Classification.
Uses EfficientNet-B0 backbone adapted for 1-channel spectrograms via 3-channel input replication.
"""

import torch
import torch.nn as nn
from torchvision import models

class SETIEfficientNet(nn.Module):
    def __init__(self, num_classes = 7):
        super(SETIEfficientNet, self).__init__()
        
        # Load EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Custom classification head with dropout
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p = 0.4, inplace = True),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1) 
        return self.backbone(x)

if __name__ == "__main__":
    print("Initializing SETIEfficientNet-B0...")
    model = SETIEfficientNet(num_classes = 7)
    dummy_input = torch.randn(8, 1, 224, 224) 
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")