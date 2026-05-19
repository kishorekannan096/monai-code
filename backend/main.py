import os
import random
import pickle
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import numpy as np
from utils.helper import (
    load_checkpoint,
    infer_with_gradcam,
    save_heatmap,
    save_image,
    clear_temp_storage
)
from dotenv import load_dotenv
load_dotenv()

# Configuration
CKPT_PATH = os.getenv("CKPT_PATH", "outputs/checkpoints/best_cxr_multilabel_3.pt")
VAL_LOADER_PATH = os.getenv("VAL_LOADER_PATH", "outputs/val_samples.pkl")
TEMP_DIR = os.getenv("TEMP_DIR", "backend/static/heatmaps")
os.makedirs(TEMP_DIR, exist_ok=True)

app = FastAPI(title="MONAI Inference Server")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static File Serving
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# 1. Serve image samples from outputs/val_samples
VAL_SAMPLES_DIR = os.path.join(PROJECT_ROOT, "outputs", "val_samples")
if os.path.exists(VAL_SAMPLES_DIR):
    app.mount("/images", StaticFiles(directory=VAL_SAMPLES_DIR), name="val_samples")
else:
    print(f"Warning: {VAL_SAMPLES_DIR} not found.")

# 2. Serve static files for heatmaps (mounted at /static)
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global state
model = None
labels = None
img_size = None
device = None
val_loader = None
val_items = []

@app.on_event("startup")
async def startup_event():
    global model, labels, img_size, device, val_items
    
    # Load model
    model, labels, img_size, device = load_checkpoint(CKPT_PATH)
    
    # Load pickled list of items
    if os.path.exists(VAL_LOADER_PATH):
        with open(VAL_LOADER_PATH, "rb") as f:
            # Expecting a list of dicts: [{"image": path, "label": array}, ...]
            val_items = pickle.load(f)
            if not isinstance(val_items, list):
                print(f"Warning: Expected list in {VAL_LOADER_PATH}, got {type(val_items)}")
    else:
        print(f"Warning: {VAL_LOADER_PATH} not found. Please pickle the validation list.")

class SampleResponse(BaseModel):
    image_path: str
    labels: List[float]
    class_names: List[str]

@app.get("/sample", response_model=SampleResponse)
async def get_sample():
    if not val_items:
        raise HTTPException(status_code=500, detail="Validation items not loaded.")
    
    item = random.choice(val_items)
    
    lbl = item["label"]
    if hasattr(lbl, "tolist"):
        lbl = lbl.tolist()
    
    return {
        "image_path": item["image"],
        "labels": lbl,
        "class_names": labels
    }

@app.get("/classes")
async def get_classes():
    if labels is None:
        raise HTTPException(status_code=500, detail="Model labels not loaded.")
    return {"class_names": labels}

class InferRequest(BaseModel):
    image_path: str
    class_name: str

class InferResponse(BaseModel):
    probabilities: Dict[str, float]
    heatmap_url: str
    input_image_url: str
    prediction: str

@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    # 1. Clear temp storage
    clear_temp_storage(TEMP_DIR)
    
    # 2. Get class index
    if request.class_name not in labels:
        raise HTTPException(status_code=400, detail=f"Invalid class name. Must be one of {labels}")
    
    label_to_idx = {l: i for i, l in enumerate(labels)}
    class_idx = label_to_idx[request.class_name]
    
    # 3. Perform inference
    try:
        # Resolve path: images are in VAL_SAMPLES_DIR by their filename
        # request.image_path might be "data/images/xxx.png", but we only have "xxx.png" in VAL_SAMPLES_DIR
        abs_image_path = os.path.join(VAL_SAMPLES_DIR, os.path.basename(request.image_path))
        
        probs, cam, img = infer_with_gradcam(
            model=model,
            image_path=abs_image_path,
            class_idx=class_idx,
            img_size=img_size,
            target_layers="features"
        )
        
        # 4. Save heatmap and input image
        suffix = random.randint(1000, 9999)
        heatmap_filename = f"heatmap_{suffix}.jpg"
        input_filename = f"input_{suffix}.jpg"
        
        heatmap_path = os.path.join(TEMP_DIR, heatmap_filename)
        input_path = os.path.join(TEMP_DIR, input_filename)
        
        save_heatmap(img, cam, heatmap_path)
        save_image(img, input_path)
        
        topk = np.argsort(-probs)[:5]
        # 5. Prepare response
        prob_dict = {labels[i]: float(probs[i]) for i in topk}
        # Simple prediction: classes with prob > 0.5
        preds = [labels[i] for i, p in enumerate(probs) if p > 0.5]
        prediction_str = ", ".join(preds) if preds else "No Finding"
        
        return {
            "probabilities": prob_dict,
            "heatmap_url": f"/static/heatmaps/{heatmap_filename}",
            "input_image_url": f"/static/heatmaps/{input_filename}",
            "prediction": prediction_str
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infer_upload", response_model=InferResponse)
async def infer_upload(
    file: UploadFile = File(...),
    class_name: str = Form(...)
):
    # 1. Clear temp storage
    clear_temp_storage(TEMP_DIR)
    
    # 2. Get class index
    if labels is None:
        raise HTTPException(status_code=500, detail="Model labels not loaded.")
    
    if class_name not in labels:
        raise HTTPException(status_code=400, detail=f"Invalid class name. Must be one of {labels}")
    
    label_to_idx = {l: i for i, l in enumerate(labels)}
    class_idx = label_to_idx[class_name]
    
    # 3. Save uploaded file temporarily
    upload_path = os.path.join(TEMP_DIR, f"upload_{file.filename}")
    try:
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        # 4. Perform inference
        probs, cam, img = infer_with_gradcam(
            model=model,
            image_path=upload_path,
            class_idx=class_idx,
            img_size=img_size,
            target_layers="features"
        )
        
        # 5. Save heatmap and preprocessed input
        suffix = random.randint(1000, 9999)
        heatmap_filename = f"heatmap_{suffix}.jpg"
        input_filename = f"input_{suffix}.jpg"
        
        heatmap_path = os.path.join(TEMP_DIR, heatmap_filename)
        input_path = os.path.join(TEMP_DIR, input_filename)
        
        save_heatmap(img, cam, heatmap_path)
        save_image(img, input_path)
        
        topk = np.argsort(-probs)[:5]
        prob_dict = {labels[i]: float(probs[i]) for i in topk}
        preds = [labels[i] for i, p in enumerate(probs) if p > 0.5]
        prediction_str = ", ".join(preds) if preds else "No Finding"
        
        return {
            "probabilities": prob_dict,
            "heatmap_url": f"/static/heatmaps/{heatmap_filename}",
            "input_image_url": f"/static/heatmaps/{input_filename}",
            "prediction": prediction_str
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
