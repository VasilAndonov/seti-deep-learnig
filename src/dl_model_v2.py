"""
Optimized DL Architecture (V2).
Natively adapts EfficientNet-B0 to 1-channel input by averaging pretrained weights,
and introduces a multi-layer classifier head with BatchNorm and SiLU activation.
"""

import torch
import torch.nn as nn
from torchvision import models

class SETIEfficientNetV2(nn.Module):
    def __init__(self, num_classes = 7):
        super(SETIEfficientNetV2, self).__init__()
        
        # 1. Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # 2. Modify first conv layer to accept 1-channel grayscale directly
        # Average weights across the 3 RGB channels to preserve pretrained feature knowledge
        original_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels = 1,
            out_channels = original_conv.out_channels,
            kernel_size = original_conv.kernel_size,
            stride = original_conv.stride,
            padding = original_conv.padding,
            bias = False
        )
        with torch.no_grad():
            new_conv.weight = nn.Parameter(original_conv.weight.sum(dim = 1, keepdim = True))
            
        self.backbone.features[0][0] = new_conv
        
        # 3. Enhanced Classifier Head
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 512),
            nn.SiLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    model = SETIEfficientNetV2(num_classes=7)
    dummy_input = torch.randn(8, 1, 224, 224)
    output = model(dummy_input)
    print(f"V2 Model Initialized successfully. Output shape: {output.shape}")