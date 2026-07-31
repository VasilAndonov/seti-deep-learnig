"""
Grad-CAM comparison code (V1 vs V2).
Generates a 3-row visual comparison showing what each model is looking at:
  Row 1: Original Test Images
  Row 2: V1 Baseline Grad-CAM Heatmaps
  Row 3: V2 Optimized Grad-CAM Heatmaps
"""

import os

import cv2

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms

from dl_model import SETIEfficientNet as SETIEfficientNetV1
from dl_model_v2 import SETIEfficientNetV2
from dl_data_prep_v2 import SETIDatasetV2


# ==========================================
# 1. CUSTOM GRAD-CAM IMPLEMENTATION
# ==========================================
class GradCAM:
    """Extracts the gradient heatmaps from a target convolutional layer."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to intercept data during forward and backward passes
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate_heatmap(self, input_tensor, class_idx = None):
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim = 1).item()
            
        # Backward pass
        score = output[:, class_idx]
        score.backward(retain_graph = True)
        
        # Pool the gradients across the spatial dimensions
        weights = torch.mean(self.gradients, dim = [2, 3], keepdim = True)
        
        # Weight the activations and apply ReLU
        cam = torch.sum(weights * self.activations, dim = 1, keepdim = True)
        cam = F.relu(cam)
        
        # Normalize between 0 and 1
        cam = cam / (torch.max(cam) + 1e-7)
        return cam.squeeze().cpu().detach().numpy()


# ==========================================
# 2. IMAGE OVERLAY HELPER
# ==========================================
def apply_colormap_on_image(org_img, activation, colormap_name = cv2.COLORMAP_JET):
    """Blends the Grad-CAM heatmap with the original image."""
    # Ensure original image is 3 channels and normalized to 0-255
    if len(org_img.shape) == 2 or org_img.shape[0] == 1:
        org_img = np.squeeze(org_img)
        org_img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2BGR)
    
    org_img = (org_img - org_img.min()) / (org_img.max() - org_img.min() + 1e-7)
    org_img = np.uint8(255 * org_img)
    
    # Resize heatmap to match image size
    activation = cv2.resize(activation, (org_img.shape[1], org_img.shape[0]))
    
    # Convert heatmap to RGB colors
    heatmap = np.uint8(255 * activation)
    heatmap = cv2.applyColorMap(heatmap, colormap_name)
    
    # Blend them together
    superimposed_img = cv2.addWeighted(org_img, 0.4, heatmap, 0.6, 0)
    
    # Convert BGR to RGB
    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)


# ==========================================
# 3. MAIN EXECUTION & PLOTTING
# ==========================================
def generate_gradcam_comparison():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Generating Grad-CAMs on device: {device}")

    # 1. Setup Test Dataset
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5], std = [0.5])
    ])
    dataset = SETIDatasetV2("data/raw", split = "test", transform = test_transform)
    classes = dataset.classes

    # 2. Load Models
    v1_model = SETIEfficientNetV1(num_classes = len(classes)).to(device)
    v1_model.load_state_dict(torch.load("models/seti_efficientnet_best.pth", map_location = device))
    v1_model.eval()

    v2_model = SETIEfficientNetV2(num_classes = len(classes)).to(device)
    v2_model.load_state_dict(torch.load("models/seti_efficientnet_v2_best.pth", map_location = device))
    v2_model.eval()

    # 3. Attach Grad-CAM to the final Convolutional layer of EfficientNet
    v1_cam = GradCAM(v1_model, v1_model.backbone.features[-1])
    v2_cam = GradCAM(v2_model, v2_model.backbone.features[-1])

    # 4. Select 4 signal classes to visualize
    target_classes = ['narrowband', 'narrowbanddrd', 'squarepulsednarrowband', 'squiggle']
    selected_samples = []

    for idx, (img_tensor, label) in enumerate(dataset):
        cls_name = classes[label]
        if cls_name in target_classes:
            selected_samples.append((img_tensor, cls_name))
            target_classes.remove(cls_name)
        if not target_classes:
            break

    # 5. Build the 3x4 Plot Grid
    fig, axes = plt.subplots(3, 4, figsize = (16, 12))
    fig.suptitle("Grad-CAM Feature Attention (V1 vs V2)", fontsize = 20, fontweight = 'bold', y = 0.98)

    for col, (img_tensor, cls_name) in enumerate(selected_samples):
        # Prepare inputs
        inputs_v2 = img_tensor.unsqueeze(0).to(device)
        inputs_v1 = inputs_v2.repeat(1, 3, 1, 1) if getattr(v1_model.backbone.features[0][0], 'in_channels', 1) == 3 else inputs_v2
        
        # Raw Image for display
        raw_img = img_tensor.squeeze().numpy()

        # Generate Heatmaps
        v1_heatmap = v1_cam.generate_heatmap(inputs_v1.requires_grad_(True))
        v2_heatmap = v2_cam.generate_heatmap(inputs_v2.requires_grad_(True))

        # Create Overlays
        v1_overlay = apply_colormap_on_image(raw_img, v1_heatmap)
        v2_overlay = apply_colormap_on_image(raw_img, v2_heatmap)

        # Plot Row 1: Original
        axes[0, col].imshow(raw_img, cmap = 'gray')
        axes[0, col].set_title(f"True Signal: {cls_name}", fontsize = 12, fontweight = 'bold')
        axes[0, col].axis('off')

        # Plot Row 2: V1 Grad-CAM
        axes[1, col].imshow(v1_overlay)
        if col == 0:
            axes[1, col].set_ylabel("V1 Baseline\nAttention", fontsize = 14, fontweight = 'bold')
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

        # Plot Row 3: V2 Grad-CAM
        axes[2, col].imshow(v2_overlay)
        if col == 0:
            axes[2, col].set_ylabel("V2 Optimized\nAttention", fontsize = 14, fontweight = 'bold')
        axes[2, col].set_xticks([])
        axes[2, col].set_yticks([])

    plt.tight_layout(rect = [0, 0, 1, 0.95])
    
    save_path = "plots/final_results/06_gradcam_comparison.png"

    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')

    print(f"\nGrad-CAM comparison is generated and saved to: {save_path}")

    plt.show()

if __name__ == "__main__":
    generate_gradcam_comparison()