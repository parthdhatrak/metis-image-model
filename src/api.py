from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import cv2
import numpy as np
from inference import MalnutritionPredictor

app = FastAPI(title="Malnutrition Detection API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Predictor globally for performance (loads model once)
# Note: Ensure malnutrition_model.h5 exists before starting
MODEL_PATH = "malnutrition_model.h5"
predictor = None

@app.on_event("startup")
def load_model():
    global predictor
    try:
        predictor = MalnutritionPredictor(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Warning: Model could not be loaded. Please run train.py first. Error: {e}")

@app.get("/")
def read_root():
    return {"status": "Malnutrition Detection API is running"}

@app.post("/predict")
async def predict_malnutrition(file: UploadFile = File(...)):
    """
    Accepts an image file and returns malnutrition classification results.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded. Try training the model first.")

    # Save temporary file
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Run inference
        result = predictor.predict(temp_path)
        
        # Cleanup
        os.remove(temp_path)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
