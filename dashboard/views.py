import os
import sys

# Fix for DLL loading on Windows (TF first)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings

# Add src to path to import our modules
sys.path.append(os.path.join(settings.BASE_DIR, 'src'))

try:
    import tensorflow as tf
    from inference import MalnutritionPredictor
    from gradcam import get_gradcam_heatmap, display_gradcam
    import cv2
    import numpy as np
except ImportError as e:
    print(f"Error importing modules: {e}")

MODEL_PATH = os.path.join(settings.BASE_DIR, 'malnutrition_final_v2.h5')
predictor = None

def get_predictor():
    global predictor
    if predictor is None and os.path.exists(MODEL_PATH):
        try:
            predictor = MalnutritionPredictor(MODEL_PATH)
        except Exception as e:
            print(f"Failed to load predictor: {e}")
    return predictor

def index(request):
    return render(request, 'dashboard/index.html')

def predict(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        uploaded_file_url = fs.url(filename)
        full_path = fs.path(filename)
        
        # Load Predictor
        p = get_predictor()
        if not p:
            return render(request, 'dashboard/index.html', {
                'error': 'Model not found or training in progress. Please check back later.'
            })
            
        # 1. Prediction
        result = p.predict(full_path)
        
        # 2. Grad-CAM
        try:
            # We need to preprocess for Grad-CAM
            img = cv2.imread(full_path)
            # Use the predictor's extractor logic to be consistent
            processed_img = p.extractor.extract_regions(img)
            img_array = processed_img.astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            heatmap = get_gradcam_heatmap(p.model, img_array)
            superimposed = display_gradcam(processed_img, heatmap)
            
            # Save Grad-CAM image
            cam_filename = 'cam_' + filename
            cam_path = os.path.join(settings.MEDIA_ROOT, cam_filename)
            cv2.imwrite(cam_path, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
            cam_url = fs.url(cam_filename)
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            cam_url = None

        return render(request, 'dashboard/result.html', {
            'image_url': uploaded_file_url,
            'cam_url': cam_url,
            'class': result.get('class'),
            'confidence': result.get('confidence'),
        })
        
    return render(request, 'dashboard/index.html')
