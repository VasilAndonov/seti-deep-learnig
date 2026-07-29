"""
ml_data_prep.py
Machine Learning Data Pipeline for SETI Classification

Extracts, flattens, scales, and applies PCA dimensionality reduction 
to the SETI spectrograms for use with classical ML models (SVM, Random Forest).
"""

import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
RAW_DATA_DIR = "data/raw/" 
PROCESSED_ML_DIR = "data/processed/ml"
IMG_SIZE_ML = (170, 128) # (Width, Height) Preserves 3:4 native aspect ratio

def load_ml_split(split_name, limit_per_class = 400):
    """Helper function to load and flatten images for a specific split."""
    split_dir = os.path.join(RAW_DATA_DIR, split_name)
    classes = sorted(os.listdir(split_dir))
    X, y = [], []
    
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir): 
            continue
        
        # Limit the ML samples to prevent RAM explosion during PCA covariance matrix calculation
        images = os.listdir(cls_dir)[:limit_per_class]
        for img_name in images:
            img_path = os.path.join(cls_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, IMG_SIZE_ML)
                X.append(img_resized.flatten())
                y.append(label)
                
    return np.array(X), np.array(y)

def prepare_ml_data():
    """Extracts, scales, and runs PCA on the pre-split ML data."""
    print(">>> Starting ML Data Preparation (128x170)...")
    
    X_train, y_train = load_ml_split('train', limit_per_class = 500)
    X_test, y_test = load_ml_split('test', limit_per_class = 200) 
    
    # Standardize features by removing the mean and scaling to unit variance
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA to retain 95% of spatial variance
    print("Running Principal Component Analysis (PCA)...")
    pca = PCA(n_components = 0.95, random_state = 42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"ML Pipeline: Reduced features from {X_train.shape[1]} to {X_train_pca.shape[1]} dimensions.")
    
    # Save the processed artifacts
    os.makedirs(PROCESSED_ML_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_ML_DIR, 'X_train_pca.npy'), X_train_pca)
    np.save(os.path.join(PROCESSED_ML_DIR, 'X_test_pca.npy'), X_test_pca)
    np.save(os.path.join(PROCESSED_ML_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_ML_DIR, 'y_test.npy'), y_test)
    print(f"ML Data successfully saved to {PROCESSED_ML_DIR}\n")

if __name__ == "__main__":
    prepare_ml_data()