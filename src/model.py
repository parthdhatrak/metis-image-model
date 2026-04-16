import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model

def build_malnutrition_model(num_classes=3, input_shape=(224, 224, 3)):
    """
    Builds the Malnutrition Detection model using MobileNetV2.
    
    1. MobileNetV2: Chosen for its efficiency on mobile/edge devices and strong feature extraction.
    2. GlobalAveragePooling2D: Flattens the 7x7x1280 output of the base model into a 1280 vector by averaging.
    3. Dense(128): Intermediate layer for classification logic.
    4. Dropout(0.5): Regularization to prevent overfitting in the custom head.
    5. Dense(num_classes): Softmax output for Healthy, Moderate, Severe.
    """
    # Load pretrained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the base model by default
    base_model.trainable = False
    
    # Add custom classification head
    inputs = Input(shape=input_shape, name='input_layer_1')
    # Add custom classification head
    inputs = Input(shape=input_shape, name='input_layer_1')
    x = base_model(inputs, training=False) 
    x = GlobalAveragePooling2D(name='gap')(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(0.5, name='dropout')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="metis_malnutrition_model")
    
    return model, base_model

if __name__ == "__main__":
    model, base = build_malnutrition_model()
    model.summary()
