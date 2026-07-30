import os

import sys
sys.path.append("src")

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

import cv2

from dl_data_prep import get_dataloaders
from dl_model import SETIEfficientNet

class SimpleGradCAM:
    """Extracts gradients to see exactly what the neural network is 'looking' at."""
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        score = output[0, target_class]
        score.backward()
        
        pooled_gradients = torch.mean(self.gradients, dim = [2, 3], keepdim = True)
        cam = self.activations * pooled_gradients
        heatmap = torch.mean(cam, dim = 1).squeeze()
        
        heatmap = F.relu(heatmap)
        
        heatmap_max = torch.max(heatmap)
        if heatmap_max > 0:
            heatmap /= heatmap_max
            
        return heatmap.cpu().numpy()

def gradcam():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Grad-CAM on device: {device}")
    
    _, test_loader, classes = get_dataloaders(batch_size = 1) 
    model = SETIEfficientNet(num_classes = len(classes)).to(device)
    model.load_state_dict(torch.load("models/seti_efficientnet_best.pth", map_location = device))
    model.eval()

    target_layer = model.backbone.features[-1]
    cam_engine = SimpleGradCAM(model, target_layer)

    print("Searching for signals to analyze...")
    successful_samples = {}
    
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        true_class = classes[labels.item()]
        
        if predicted.item() == labels.item() and true_class not in successful_samples:
            if true_class in ['narrowband', 'squiggle', 'narrowbanddrd', 'squarepulsednarrowband']:
                successful_samples[true_class] = (inputs, labels.item())
                
        if len(successful_samples) >= 4:
            break

    # Horizontal layout
    fig, axes = plt.subplots(2, 4, figsize = (20, 10))
    fig.suptitle("Inside the Black Box: What the Model is Looking At", fontsize = 24, y = 0.98)
    
    # Iterate through our 4 samples
    for i, (class_name, (inputs, class_idx)) in enumerate(successful_samples.items()):
        
        original_img = inputs.cpu().numpy().squeeze()
        p2, p98 = np.percentile(original_img, (2, 98))
        original_img_stretched = np.clip((original_img - p2) / (p98 - p2), 0, 1)
        
        heatmap = cam_engine.generate_heatmap(inputs, class_idx)
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3] 
        
        original_3ch = np.stack((original_img_stretched,) * 3, axis = -1)
        overlay = (0.6 * heatmap_colored) + (0.4 * original_3ch)
        overlay = np.clip(overlay, 0, 1)

        # Top row: original images
        axes[0, i].imshow(original_img_stretched, cmap = 'gray')
        axes[0, i].set_title(f"Original: {class_name}", fontweight = 'bold', fontsize = 14)
        axes[0, i].axis('off')
        
        # Bottom row: Grad-CAM overlays
        axes[1, i].imshow(overlay)
        axes[1, i].set_title("Grad-CAM Focus", fontweight = 'bold', color = 'darkred', fontsize = 14)
        axes[1, i].axis('off')

    plt.tight_layout(pad = 2.0, h_pad = 1.5, w_pad = 1.5)
    
    save_path = "plots/04_gradcam_analysis.png"
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    print(f"Grad-CAM grid saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    gradcam()