"""
ml_data_prep.py
Machine Learning Data Pipeline for SETI Classification

Extracts, flattens, scales, shuffles, and applies aggressive PCA 
dimensionality reduction to isolate signals from radio noise.
"""

import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

# --- CONFIGURATION ---
RAW_DATA_DIR = "data/raw" 
PROCESSED_ML_DIR = "data/processed/ml"
IMG_SIZE_ML = (170, 128) 

def load_ml_split(split_name, limit_per_class=400):
    """Helper function to load and flatten images for a specific split."""
    split_dir = os.path.join(RAW_DATA_DIR, split_name)
    classes = sorted(os.listdir(split_dir))
    X, y = [], []
    
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir): 
            continue
        
        images = os.listdir(cls_dir)[:limit_per_class]
        for img_name in images:
            img_path = os.path.join(cls_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, IMG_SIZE_ML)
                X.append(img_resized.flatten())
                y.append(label)
                
    # Convert to arrays and SHUFFLE to prevent sequential bias
    X, y = shuffle(np.array(X), np.array(y), random_state=42)
    return X, y

def prepare_ml_data():
    """Extracts, scales, and runs PCA to strip background noise."""
    print(">>> Starting ML Data Preparation (128x170)...")
    
    X_train, y_train = load_ml_split('train', limit_per_class=500)
    X_test, y_test = load_ml_split('test', limit_per_class=200) 
    
    # Scale Data
    print("Scaling features to [0, 1] range...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # FIX: Hardcode PCA to top 60 components to strip radio noise and isolate the signal
    print("Running PCA (Extracting top 60 structural components)...")
    pca = PCA(n_components=60, random_state=42)
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