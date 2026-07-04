import os
import pickle
import torch
import torch.nn as nn
from PIL import Image
import io
import yaml
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import base64
import random
from src.data.preprocessing import clean_tweet, get_image_transforms
from src.models.classifier import MultimodalClassifier

app = FastAPI(title="Multimodal Disaster Classification API")

# Enable CORS for React Native Web / mobile clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for loaded models, vectorizers, and mappings
models_cache = {}
configs = None
device = None

def init_resources():
    global configs, device
    # Load configuration
    with open("config.yaml", "r") as f:
        configs = yaml.safe_load(f)
        
    device = torch.device(
        "cuda" if torch.cuda.is_available() and configs["training"]["device"] == "cuda"
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Server backend initialized. Using device: {device}")

def get_task_components(task_name: str):
    """
    Loads and caches components (model, vectorizer, class mapping) for a given task.
    """
    if task_name in models_cache:
        return models_cache[task_name]
        
    # Check if checkpoint files exist
    ckpt_dir = "models/checkpoints"
    model_path = os.path.join(ckpt_dir, f"multimodal_best_{task_name}.pth")
    vec_path = os.path.join(ckpt_dir, f"vectorizer_{task_name}.pkl")
    map_path = os.path.join(ckpt_dir, f"mapping_{task_name}.pkl")
    
    if not (os.path.exists(model_path) and os.path.exists(vec_path) and os.path.exists(map_path)):
        return None
        
    # Load vectorizer
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
        
    # Load class mapping
    with open(map_path, "rb") as f:
        mapping = pickle.load(f)
        
    # Instantiate and load model
    text_in_dim = len(vectorizer.vocabulary_)
    num_classes = len(mapping["unique_labels"])
    
    model = MultimodalClassifier(
        text_in_dim=text_in_dim,
        image_backbone=configs["image_model"]["backbone"],
        pretrained_image=False, # No need to download ImageNet weights for loading custom checkpoints
        freeze_image_backbone=False,
        project_dim=configs["multimodal_model"]["project_dim"],
        fusion_type="attention", # Always use attention for Phase 2
        num_classes=num_classes,
        dropout=0.0
    )
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    components = {
        "model": model,
        "vectorizer": vectorizer,
        "mapping": mapping
    }
    models_cache[task_name] = components
    return components

@app.post("/predict")
async def predict(
    tweet: str = Form(...),
    image: UploadFile = File(...)
):
    # Ensure config and device are loaded
    if configs is None:
        init_resources()
        
    # 1. Preprocess text
    cleaned_text = clean_tweet(tweet)
    
    # 2. Preprocess image
    try:
        # Detect SVG presets and bypass PIL raw parser
        is_svg = (image.filename and image.filename.endswith(".svg")) or (image.headers.get("content-type") == "image/svg+xml")
        
        if is_svg:
            filename_lower = (image.filename or "").lower()
            if "wildfire" in filename_lower or "fire" in filename_lower:
                pil_image = Image.new("RGB", (224, 224), color=(239, 68, 68)) # Red
            elif "flood" in filename_lower or "water" in filename_lower:
                pil_image = Image.new("RGB", (224, 224), color=(6, 182, 212)) # Blue
            else:
                pil_image = Image.new("RGB", (224, 224), color=(120, 53, 15)) # Brown/Earthquake
        else:
            contents = await image.read()
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
    img_size = configs["image_model"]["img_size"]
    transform_fn = get_image_transforms(img_size=img_size, is_train=False)
    img_tensor = transform_fn(pil_image).unsqueeze(0).to(device) # Add batch dimension -> [1, 3, H, W]
    
    results = {}
    tasks = ["informativeness", "disaster_type", "damage_severity"]
    
    for task in tasks:
        comp = get_task_components(task)
        if comp is None:
            results[task] = {
                "error": "Model checkpoint not found. Ensure training has completed for this task."
            }
            continue
            
        model = comp["model"]
        vectorizer = comp["vectorizer"]
        mapping = comp["mapping"]
        
        # Vectorize text and load tensor
        text_vec = vectorizer.transform([cleaned_text]).toarray()
        text_tensor = torch.tensor(text_vec, dtype=torch.float32).to(device)
        
        # Inference
        with torch.no_grad():
            logits = model(text_tensor, img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
        pred_label = mapping["idx_to_label"][pred_idx]
        
        # Format labels nicely for frontend
        formatted_label = pred_label.replace("_", " ").title()
        
        # Build probability distribution map
        prob_dist = {}
        for idx, prob in enumerate(probs.tolist()):
            label_name = mapping["idx_to_label"][idx].replace("_", " ").title()
            prob_dist[label_name] = round(prob * 100, 2)
            
        results[task] = {
            "prediction": formatted_label,
            "confidence": round(confidence * 100, 2),
            "distribution": prob_dist
        }
        
    return results

@app.get("/random-test-sample")
def get_random_test_sample():
    if configs is None:
        init_resources()
        
    tsv_path = configs["data"]["test_tsv"]
    images_dir = configs["data"]["raw_dir"]
    
    if not os.path.exists(tsv_path):
        raise HTTPException(status_code=404, detail=f"Test TSV file not found at {tsv_path}")
        
    try:
        df = pd.read_csv(tsv_path, sep="\t")
        df = df.dropna(subset=["image", "tweet_text"])
        
        # Select random row
        row = df.sample(n=1).iloc[0]
        
        img_rel_path = row["image"]
        img_abs_path = os.path.join(images_dir, img_rel_path)
        
        if not os.path.exists(img_abs_path):
            raise FileNotFoundError(f"Image not found at {img_abs_path}")
            
        with open(img_abs_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        ext = os.path.splitext(img_rel_path)[1].lower().replace(".", "")
        mime = f"image/{ext}" if ext in ["png", "jpg", "jpeg", "webp"] else "image/jpeg"
        data_uri = f"data:{mime};base64,{encoded_string}"
        
        return {
            "tweet": str(row["tweet_text"]),
            "image_data_uri": data_uri,
            "event_name": str(row.get("event_name", "Unknown")),
            "damage_severity": str(row.get("label", "Unknown"))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch test sample: {str(e)}")

# Initialize components on startup if model checkpoints exist
@app.on_event("startup")
def startup_event():
    init_resources()
    # Try loading tasks pre-emptively
    for task in ["informativeness", "disaster_type", "damage_severity"]:
        try:
            get_task_components(task)
        except Exception as e:
            print(f"Could not load task {task} on startup: {e}")

# Mount static folder for serving frontend assets
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    @app.get("/")
    def read_root():
        return {"message": "Server running. Static directory not found. Create src/static folder structure."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
