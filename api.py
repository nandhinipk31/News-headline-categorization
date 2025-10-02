# api.py
import os
import pickle
import logging
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import texts_to_padded, MAX_LEN
from preprocess import clean_text
import torch

# -------------------------------
# Logging setup
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "distilmbert_headline")

# -------------------------------
# Load model, tokenizer, and label encoder
# -------------------------------
def load_model_and_objects():
    try:
        logger.info("📥 Loading Hugging Face PyTorch model...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
        with open(le_path, "rb") as f:
            le = pickle.load(f)

        logger.info(f"✅ Loaded model, tokenizer, and {len(le.classes_)} categories")
        return model, tokenizer, le

    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model: {e}")

model, tokenizer, le = load_model_and_objects()

# Set model to eval mode
model.eval()

# -------------------------------
# Initialize FastAPI app
# -------------------------------
app = FastAPI(
    title="📰 News Headline Categorization API",
    version="1.2",
    description="Multilingual news headline classification with Hugging Face + FastAPI",
    contact={"name": "ML API Maintainer", "email": "support@example.com"},
)

# -------------------------------
# Input schema
# -------------------------------
class HeadlineRequest(BaseModel):
    headline: str

# -------------------------------
# Output schema
# -------------------------------
class PredictionResponse(BaseModel):
    headline: str
    predicted_category: str
    confidence: float

# -------------------------------
# Root endpoint
# -------------------------------
@app.get("/", tags=["Root"])
def home():
    return {
        "message": "✅ News Headline Categorization API is running!",
        "endpoints": ["/predict/"],
        "model_info": {
            "categories": len(le.classes_),
            "max_sequence_length": MAX_LEN,
        },
    }

# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict/", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: HeadlineRequest):
    headline = request.headline.strip()
    if not headline:
        return PredictionResponse(
            headline=headline,
            predicted_category="unknown",
            confidence=0.0,
        )
    try:
        # Clean & preprocess
        clean = clean_text(headline)
        inputs = texts_to_padded(tokenizer, [clean], max_len=MAX_LEN)

        # Predict
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        idx = int(probs.argmax())
        confidence = float(probs[idx])

        return PredictionResponse(
            headline=headline,
            predicted_category=le.classes_[idx],
            confidence=confidence,
        )

    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed. Check logs for details.")

# -------------------------------
# Run FastAPI server (local dev only)
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
