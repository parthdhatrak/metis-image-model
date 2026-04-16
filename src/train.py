import os
import sys

# Windows Fix: Force TF initialization before other libraries
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from model import build_malnutrition_model
from preprocessing import MalnutritionDataGenerator
from region_extraction import RegionExtractor

# Fix for DLL loading on Windows (TF first)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Optimization Configs
BATCH_SIZE = 16
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 40 # Increased as requested
IMAGE_SIZE = (224, 224)
DATASET_DIR = "d:/metis-image-detection/dataset"

def load_dataset_metadata(subset="train"):
    csv_path = os.path.join(DATASET_DIR, subset, "_annotations.csv")
    df = pd.read_csv(csv_path).drop_duplicates(subset='filename')
    
    label_map = {"Healthy": 0, "Malnourished": 1, "Severe": 2} # 3-class structure
    
    image_paths = [os.path.join(DATASET_DIR, subset, f) for f in df['filename']]
    labels = [label_map.get(l, 1) for l in df['class']] # Default to 1 (Moderate)
    
    labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=3)
    return image_paths, labels_one_hot, labels

def train_system_optimized():
    """
    Step 5, 6, 8 & 9: Optimized Training Pipeline.
    """
    # 1. Load Data
    train_paths, train_labels, train_idx = load_dataset_metadata("train")
    val_paths, val_labels, val_idx = load_dataset_metadata("valid")
    
    # Compute Class Weights (Step 2)
    cw = compute_class_weight('balanced', classes=np.unique(train_idx), y=train_idx)
    class_weight_dict = {i: weight for i, weight in enumerate(cw)}
    print(f"\n[OPTIMIZATION] Calculated Class Weights: {class_weight_dict}")
    
    extractor = RegionExtractor()
    
    train_gen = MalnutritionDataGenerator(
        train_paths, train_labels, batch_size=BATCH_SIZE, 
        augment=True, region_extractor=extractor
    )
    val_gen = MalnutritionDataGenerator(
        val_paths, val_labels, batch_size=BATCH_SIZE, 
        augment=False, region_extractor=extractor
    )
    
    # 2. Build Model (Step 5: MobileNetV2 + Head + Dropout 0.5)
    model, base_model = build_malnutrition_model(num_classes=3)
    
    # 3. Phase 1: Feature Extraction
    print("\n--- Phase 1: Calibration (Head Only) ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_model_v2_optimized.h5', save_best_only=True)
    ]
    
    h1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE1,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    
    # 4. Phase 2: Deep Fine-Tuning (Step 5: Unfreeze 50 layers, LR 1e-5)
    print("\n--- Phase 2: Deep Optimization (Unfreezing 50 layers) ---")
    base_model.trainable = True
    for layer in base_model.layers[:-50]: # Unfreeze more as requested (50 layers)
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Lower LR for stability
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_ft = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
    ]
    
    h2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE2,
        class_weight=class_weight_dict,
        callbacks=callbacks_ft
    )
    
    # 5. Full Evaluation (Step 8)
    evaluate_optimized(model, val_gen, h1, h2)
    
    model.save('malnutrition_final_v2.h5')
    return model

def evaluate_optimized(model, val_gen, h1, h2):
    # Accuracy Plot
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    
    plt.figure(figsize=(10, 5))
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Optimized Accuracy Curves')
    plt.legend()
    plt.savefig('optimized_curves.png')
    
    # Detailed Metrics (Step 8)
    y_pred, y_true = [], []
    for i in range(len(val_gen)):
        x, y = val_gen[i]
        preds = model.predict(x)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(y, axis=1))
        
    print("\n[OPTIMIZED] Classification Report:")
    # Using explicit labels to handle missing classes in validation
    print(classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=['Healthy', 'Moderate', 'Severe']))
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['H', 'M', 'S'], yticklabels=['H', 'M', 'S'])
    plt.savefig('optimized_confusion_matrix.png')

if __name__ == "__main__":
    train_system_optimized()
