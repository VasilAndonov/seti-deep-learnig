# Deep Learning for SETI: Optimizing Convolutional Neural Networks for Spectrogram Signal Classificatio

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![Torchvision](https://img.shields.io/badge/Torchvision-Computer%20Vision-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8.svg)

> **Key Achievement:** By adapting a pre-trained EfficientNet-B0 architecture to use 1-channel spectrograms and applying spatial regularization (Random Erasing), this model closed a big training overfitting gap. It achieved 94.43% overall test accuracy while reducing inference latency by 7.5%, proving its capability to track continuous Doppler drift trajectories without anchoring to isolated background noise.

<img width="2683" height="1543" alt="06_gradcam_comparison" src="https://github.com/user-attachments/assets/3259e3da-8144-4389-8f27-fd747ca1e0d4" />
Figure: Grad-CAM Feature Attention highlighting how V2 optimizations (bottom) forced the neural network to learn continuous signal trajectories, overcoming the localized pixel-collapse seen in the V1 baseline (middle).

---

## 1. Problem Formulation
Radio telescopes generate a lot of data sweeping the electromagnetic spectrum for narrow-band signals that could indicate artificial extraterrestrial sources (technosignatures). Distinguishing these faint signals from surrounding cosmic noise and human-generated radio frequency interference (RFI) is a complex task.

## 2. Project Architecture
* **`PyTorch` & `Torchvision`:** Core deep learning framework handling EfficientNet feature extraction and dynamic data augmentations.
* **`Automatic Mixed Precision (AMP)`:** Hardware acceleration utilizing PyTorch `GradScaler` for efficient gradient calculation on NVIDIA CUDA and Apple Silicon (MPS).
* **`OpenCV`:** High-speed matrix ingestion of raw 1-channel grayscale spectrograms.
* **`Grad-CAM`:** Interpretability engine used to generate feature activation heatmaps, ensuring the model's visual reasoning aligns with signal physics.

## 3. DL Model
We evaluated baseline machine learning approache against deep learning architectures. An optimized EfficientNet-B0 was selected due to its scaling between feature extraction depth and computational parameter count.

* **Base Architecture:** SETIEfficientNetV2 (ImageNet-1k pre-trained weights, summed for 1-channel inputs)
* **Classifier Head:** 1D BatchNorm -> 512-Node Dense Layer -> SiLU Activation -> 40% Dropout
* **Dimensionality:** 1x224x224 2D Time-Frequency Spectrograms
* **Optimization:** AdamW with Differential Learning Rates (1e-4 for backbone, 1e-3 for head) + Linear Warmup and Cosine Annealing.
* **Explainability:** Integrated Grad-CAM mappings confirming the network tracks full spatial continuity (linear drift, curves, pulsing gaps) across the entire matrix.

A comparison between V1 (baseline) and V2 (optimized) confirmed that without Random Erasing and deep Dropout regularization, the V1 model memorized training artifacts, achieving peak validation accuracy at Epoch 25 before overfitting. The V2 optimizations closed this gap, expanding the network's receptive field and improving the F1-Score of the most difficult signal class (from 0.8807 to 0.8959).

---

## 4. Repository Structure

```text
├── data/                  # Directory for raw and processed Kaggle SETI datasets
├── notebooks/             # Jupyter notebooks for EDA and initial model prototyping
├── models/                # Saved weights (*.pth) and training logs
├── plots/                 # Web-optimized diagnostic visualizations and Grad-CAM plots
├── src/                   # Python code (models, pipelines, training, evaluations)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## 5. Execution Guide
To reproduce the experimental pipeline:

1. Install dependencies:
`pip install -r requirements.txt`

2. Pull datasets via DVC:
`dvc pull`

3. Visualize data augmentation:
`python src/visualize_augmentations.py`
`python src/visualize_augmentations_v2.py`

4. Train both models:
`python src/dl_train.py`
`python src/dl_train_v2.py`

5. Models comparison (ROC, Confusion Matrices, Metrics):
`python src/v1_v2_comparison.py`

6. Generate Grad-CAM Plots:
`python src/gradcam_comparison.py`
