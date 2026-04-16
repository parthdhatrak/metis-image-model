import os
# Fix for DLL load failed error on Windows
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

def convert_to_tflite(h5_model_path='malnutrition_final_v2.h5', tflite_path='malnutrition_model_v2.tflite'):
    """
    Converts a saved Keras model (.h5) to TensorFlow Lite (.tflite).
    
    Why TFLite?
    1. Size: Significantly smaller than .h5 files.
    2. Speed: Optimized for mobile/embedded CPU and GPU (NPU).
    3. Quantization: Supports INT8/Float16 for further acceleration (optional).
    """
    try:
        model = tf.keras.models.load_model(h5_model_path)
        
        # Initialize the converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optional: Apply basic optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the model
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
        print(f"Successfully converted {h5_model_path} to {tflite_path}")
        
    except Exception as e:
        print(f"Failed to convert model: {e}")

if __name__ == "__main__":
    convert_to_tflite()
