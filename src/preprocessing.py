import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import cv2
import numpy as np

def get_augmentation_pipeline():
    """
    Creates a Keras Sequential model for data augmentation.
    
    Augmentations:
    1. RandomRotation (±90 degrees): Factor 0.25. Critical for children lying down in beds.
    2. RandomZoom (0.6-1.4): Factor 0.4.
    3. RandomFlip ("horizontal"): Since physiological wasting is symmetric.
    4. RandomBrightness (0.5 factor -> 0.5 to 1.5): Handles dim hospital ward lighting.
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.25),
        tf.keras.layers.RandomZoom(height_factor=(-0.4, 0.4), width_factor=(-0.4, 0.4)),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomBrightness(factor=0.5),
    ])
    return data_augmentation

def preprocess_image(image, target_size=(224, 224)):
    """
    Standard preprocessing for inference and training.
    1. Color conversion if necessary (BGR to RGB)
    2. Resize to target size
    3. Normalization [0, 1]
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    return image

class MalnutritionDataGenerator(tf.keras.utils.Sequence):
    """
    Custom generator that integrates Region Extraction into the flow.
    """
    def __init__(self, image_paths, labels, batch_size=32, target_size=(224, 224), augment=False, region_extractor=None):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.region_extractor = region_extractor
        self.augmentation_layer = get_augmentation_pipeline() if augment else None

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x_paths = self.image_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]

        batch_x = []
        for path in batch_x_paths:
            img = cv2.imread(path)
            if img is None:
                # Handle missing image - return black image
                img = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            
            # Apply Region Extraction if provided
            if self.region_extractor:
                img = self.region_extractor.extract_regions(img)
            else:
                img = cv2.resize(img, self.target_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = img.astype(np.float32) / 255.0
            batch_x.append(img)
        
        batch_x = np.array(batch_x)
        
        if self.augment:
            batch_x = self.augmentation_layer(batch_x, training=True)
            
        return batch_x, np.array(batch_y)
