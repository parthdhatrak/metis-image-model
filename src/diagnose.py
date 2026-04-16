import os
import sys

# Fix for DLL load failed error on Windows
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

DATASET_DIR = "d:/metis-image-detection/dataset"

def diagose_dataset():
    """
    Step 1 & 2: Dataset Analysis and Class Imbalance detection.
    """
    print("\n[DIAGNOSIS] Analyzing Dataset...")
    
    # Load all annotations
    train_csv = os.path.join(DATASET_DIR, "train", "_annotations.csv")
    valid_csv = os.path.join(DATASET_DIR, "valid", "_annotations.csv")
    
    if not os.path.exists(train_csv):
        print("Error: Dataset CSV not found.")
        return
        
    df_train = pd.read_csv(train_csv).drop_duplicates(subset='filename')
    df_valid = pd.read_csv(valid_csv).drop_duplicates(subset='filename')
    
    # Count classes
    train_counts = df_train['class'].value_counts()
    valid_counts = df_valid['class'].value_counts()
    
    print("\n--- Class Distribution (Train) ---")
    print(train_counts)
    print("\n--- Class Distribution (Valid) ---")
    print(valid_counts)
    
    # Check for imbalance
    total = len(df_train)
    imbalance_detected = False
    for label, count in train_counts.items():
        ratio = count / total
        if ratio < 0.2: # Simple threshold for 3 classes
            print(f"⚠️  WARNING: Class '{label}' is underrepresented ({ratio:.1%})")
            imbalance_detected = True
            
    # Compute Class Weights
    unique_classes = np.unique(df_train['class'])
    # Map classes to indices for sklearn compatible labels
    # We use our label_map: Healthy:0, Malnourished/Moderate:1, Severe:2
    y_train = df_train['class'].map({"Healthy": 0, "Malnourished": 1, "Severe": 2}).fillna(1).values
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"\n[OPTIMIZATION] Computed Class Weights: {weight_dict}")
    print("Interpretation: Higher weights for underrepresented classes (e.g. Severe) forces the model to learn them better.")
    
    return weight_dict

def analyze_curves(history_path='training_curves.png'):
    """
    Interpretation Guide for Curves:
    
    1. Overfitting: Train accuracy is high (95%+), Val accuracy is plateauing or decreasing (60-70%).
       Solution: Increase Dropout, Stronger Augmentation, Unfreeze FEWER layers.
       
    2. Underfitting: Both Train and Val accuracy are low (e.g. < 50%).
       Solution: Lower Learning Rate, Train for LONGER epochs, Unfreeze MORE layers.
       
    3. Bias: High overall accuracy but 0% recall for a specific class (check Confusion Matrix).
       Solution: Apply Class Weights, Balanced Sampling.
    """
    print(f"\n[INFO] Please check {history_path} for visual diagnosis.")
    print("- Divergence between Train/Val loss = Overfitting.")
    print("- Flat loss curves = Model not learning (check LR).")

if __name__ == "__main__":
    diagose_dataset()
