# 🧒 Malnutrition Detection System

A production-ready deep learning pipeline to detect malnutrition in children using MobileNetV2, MediaPipe region extraction, and Grad-CAM interpretability.

## 🧱 Project Structure

```text
src/
├── region_extraction.py  # MediaPipe face & upper body extraction
├── preprocessing.py      # Augmentation and data generators
├── model.py              # MobileNetV2 architecture with custom head
├── train.py              # Two-phase training & evaluation script
├── gradcam.py            # Grad-CAM interpretability visualization
├── inference.py          # Unified prediction pipeline
├── api.py                # FastAPI backend implementation
└── export_tflite.py     # Conversion to TF Lite for edge deployment
```

## 🚀 Getting Started

### 1. Prerequisites
Install the required dependencies:
```bash
pip install tensorflow mediapipe opencv-python pandas matplotlib seaborn scikit-learn fastapi uvicorn python-multipart
```

### 2. Prepare the Dataset
Ensure your dataset is structured as follows in `d:/metis-image-detection/dataset`:
- `dataset/train/_annotations.csv`
- `dataset/valid/_annotations.csv`
- Image files inside `train/` and `valid/` folders.

### 3. Run Training
This script executes two-phase fine-tuning and generates accuracy/loss graphs.
```bash
python src/train.py
```
- **Phase 1**: Trains only the custom head (Feature extraction).
- **Phase 2**: Unfreezes the top 30 layers for fine-tuning with a low learning rate.

### 4. Run the API
Start the FastAPI server for remote inference:
```bash
python src/api.py
```
- Endpoint: `POST http://localhost:8000/predict`
- Param: `file` (Image file)

### 5. Export for Edge (Optional)
Convert the trained `.h5` model to `.tflite`:
```bash
python src/export_tflite.py
```

## 🧠 System Architecture Logic

### Region Focus (MediaPipe)
Instead of processing the entire image (which might contain background noise), we use MediaPipe to:
1.  **Detect Face**: Focuses on hollow cheeks and sunken eyes.
2.  **Detect Pose**: Finds shoulders and mid-torso to observe narrow shoulders and muscle wasting.
3.  **Combine**: The crops are stacked into a 224x224 input, forcing the model to ignore clothes/background.

### Model interpretability (Grad-CAM)
The `gradcam.py` module allows you to verify *why* the model made a decision. If the heatmap highlights the background instead of the child's arms or face, the model should be considered unreliable.

### Two-Phase Training
1.  **Phase 1**: We freeze the ImageNet weights to preserve general feature extraction (edges, textures) while teaching the new head how to classify malnutrition.
2.  **Phase 2**: We unfreeze the top layers to "align" the high-level features of MobileNetV2 specifically with physiological indicators of malnutrition.
