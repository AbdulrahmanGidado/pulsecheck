# api.py — PulseCheck API with Supabase storage

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import re
import os
from transformers import pipeline
from supabase import create_client, Client

# ── CREATE APP ──
app = FastAPI(title="PulseCheck API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CONNECT TO SUPABASE ──
# These values come from Railway's environment variables — never hardcoded.
# os.environ.get() reads them securely at runtime.
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── LOAD MODELS ──
print("Loading fast model...")
fast_model = joblib.load("model.pkl")
vectorizer  = joblib.load("tfidf-vectorizer.pkl")

print("Loading smart model...")
smart_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)
print("All models loaded. API is ready.")

LABEL_MAP = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

# ── TEXT CLEANING ──
def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return ' '.join(text.split())

# ── REQUEST / RESPONSE SHAPES ──
class AnalyzeRequest(BaseModel):
    text: str
    mode: str = "smart"
    source: str = "manual"

class AnalyzeResponse(BaseModel):
    sentiment: str
    confidence: float
    model: str
    text: str

# ── HEALTH CHECK ──
@app.get("/")
def health():
    return {"status": "ok", "service": "PulseCheck API"}

# ── ANALYZE ──
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):

    if not req.text or len(req.text.strip()) < 3:
        return AnalyzeResponse(
            sentiment="neutral", confidence=0.0,
            model=req.mode, text=req.text
        )

    if req.mode == "fast":
        cleaned    = clean(req.text)
        vector     = vectorizer.transform([cleaned])
        probs      = fast_model.predict_proba(vector)[0]
        confidence = float(np.max(probs))
        label      = fast_model.classes_[np.argmax(probs)]
        if confidence < 0.55:
            label = "neutral"
        result = AnalyzeResponse(
            sentiment=label.lower(),
            confidence=confidence,
            model="fast",
            text=req.text
        )
    else:
        output     = smart_model(req.text[:512])[0]
        sentiment  = LABEL_MAP.get(output["label"], "neutral")
        confidence = float(output["score"])
        result = AnalyzeResponse(
            sentiment=sentiment,
            confidence=confidence,
            model="smart",
            text=req.text
        )

    # ── SAVE TO SUPABASE ──
    # Every analysis gets stored as a row in the feedback table.
    # This is what makes the Reviews page show real persistent data.
    try:
        supabase.table("feedback").insert({
            "text":       result.text,
            "sentiment":  result.sentiment,
            "confidence": result.confidence,
            "model":      result.model,
            "source":     req.source,
        }).execute()
    except Exception as e:
        # Don't crash the API if saving fails — just log it
        print(f"Supabase insert failed: {e}")

    return result

# ── FETCH REVIEWS ──
# This endpoint retrieves all saved analyses from Supabase.
# The dashboard calls this on load to populate the Reviews page.
@app.get("/reviews")
def get_reviews():
    try:
        response = supabase.table("feedback")\
            .select("*")\
            .order("created_at", desc=True)\
            .limit(100)\
            .execute()
        return {"reviews": response.data}
    except Exception as e:
        print(f"Supabase fetch failed: {e}")
        return {"reviews": []}