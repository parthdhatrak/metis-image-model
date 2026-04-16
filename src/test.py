import os
import sys

# Fixed TF load early (Windows fix)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Dynamic Pathing
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

try:
    from inference import MalnutritionPredictor
    from gradcam import get_gradcam_heatmap, display_gradcam
except ImportError as e:
    print(f"Import Error: {e}")

def run_test(image_path, model_path='malnutrition_final_v2.h5'):
    """
    Step 10: Testing Implementation with Interpretability.
    """
    if not os.path.exists(model_path):
        # Fallback to general model if optimized final is not found
        model_path = 'malnutrition_model.h5'
        
    predictor = MalnutritionPredictor(model_path)
    
    # Run Prediction
    result = predictor.predict(image_path)
    print("\n--- TEST RESULT ---")
    print(f"Class: {result['class']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    
    # Step 7: Grad-CAM Visualization
    img = cv2.imread(image_path)
    # Re-extract region as predictor does internally for visualization
    processed_img = predictor.extractor.extract_regions(img)
    img_array = processed_img.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    heatmap = get_gradcam_heatmap(predictor.model, img_array)
    superimposed = display_gradcam(processed_img, heatmap)
    
    cv2.imwrite('test_result_gradcam.jpg', cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
    print(f"\n[Interpretability] Grad-CAM saved to test_result_gradcam.jpg")
    print("Check if the model is highlighting the face, arms, or torso.")
    print("If it focuses on the background, the model is using irrelevant features.")

if __name__ == "__main__":
    # Change to a valid image path in your dataset for verification
    # Example: d:/metis-image-detection/dataset/valid/image.jpg
    # run_test("path/to/test.jpg")
    print("Usage: python src/test.py <image_path>")
    if len(sys.argv) > 1:
        run_test(sys.argv[1])
