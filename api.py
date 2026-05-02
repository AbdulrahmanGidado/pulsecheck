# api.py — The brain of PulseCheck
# This file receives text, runs it through your models, and sends back a prediction.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import re
from transformers import pipeline

# ── CREATE THE APP ──
# FastAPI() creates your web server.
# Think of iuvicornt as opening a restaurant — now we need to add menu items (routes).
app = FastAPI(title="PulseCheck API", version="1.0.0")

# ── CORS MIDDLEWARE ──
# This allows your HTML dashboard (running in a browser) to talk to this API.
# Without this, browsers block requests between different origins for security reasons.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production you'd lock this to your dashboard URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LOAD MODELS AT STARTUP ──
# These load once when the server starts, not on every request.
# Loading a model on every request would make your API very slow.
print("Loading fast model...")
fast_model = joblib.load("model.pkl")
vectorizer  = joblib.load("tfidf-vectorizer.pkl")

print("Loading smart model (this may take 30s on first run)...")
smart_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)
print("All models loaded. API is ready.")

# ── LABEL MAP ──
# The RoBERTa model returns LABEL_0, LABEL_1, LABEL_2.
# This maps them to human-readable names.
LABEL_MAP = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

# ── TEXT CLEANING ──
# Only used by the fast model. The transformer handles raw text natively.
def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)   # remove URLs
    text = re.sub(r'@\w+', '', text)              # remove @mentions
    text = re.sub(r'[^a-z\s]', ' ', text)         # keep letters only
    return ' '.join(text.split())                  # remove extra spaces

# ── REQUEST SHAPE ──
# This defines exactly what your API expects to receive.
# Pydantic validates it automatically — wrong types get rejected with a clear error.
class AnalyzeRequest(BaseModel):
    text: str
    mode: str = "smart"    # "fast" or "smart"
    source: str = "manual" # where the feedback came from, e.g. "checkout-form"

# ── RESPONSE SHAPE ──
# This defines exactly what your API sends back.
class AnalyzeResponse(BaseModel):
    sentiment: str    # "positive", "negative", or "neutral"
    confidence: float # 0.0 to 1.0
    model: str        # which model was used
    text: str         # echo the input back (useful for logging)

# ── HEALTH CHECK ROUTE ──
# A simple endpoint that confirms your API is alive.
# You'll use this to verify the server is running.
@app.get("/")
def health():
    return {"status": "ok", "service": "PulseCheck API"}

# ── MAIN ANALYZE ROUTE ──
# This is the core of your product.
# POST /analyze → receives text → returns sentiment prediction.
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):

    # Guard: reject empty or very short input
    if not req.text or len(req.text.strip()) < 3:
        return AnalyzeResponse(
            sentiment="neutral", confidence=0.0,
            model=req.mode, text=req.text
        )

    if req.mode == "fast":
        # Clean → vectorize → predict
        cleaned   = clean(req.text)
        vector    = vectorizer.transform([cleaned])
        probs     = fast_model.predict_proba(vector)[0]
        confidence = float(np.max(probs))
        label     = fast_model.classes_[np.argmax(probs)]

        # If confidence is too low, call it neutral
        if confidence < 0.55:
            label = "neutral"

        return AnalyzeResponse(
            sentiment=label.lower(),
            confidence=confidence,
            model="fast",
            text=req.text
        )

    else:  # smart (default)
        # Transformers have a 512 token limit — truncate just in case
        result    = smart_model(req.text[:512])[0]
        sentiment = LABEL_MAP.get(result["label"], "neutral")
        confidence = float(result["score"])

        return AnalyzeResponse(
            sentiment=sentiment,
            confidence=confidence,
            model="smart",
            text=req.text
        )