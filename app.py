"""
SentimentLens — Portfolio-ready Sentiment Analysis App
Author: Your Name
Models: TF-IDF + Logistic Regression (Fast) | RoBERTa Transformer (Smart)
"""

import streamlit as st
import joblib
import numpy as np
import re
from transformers import pipeline

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentLens",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  — Injects a clean, modern look
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Import clean font */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* Page background */
    .stApp {
        background-color: #0f1117;
    }

    /* Main container max-width */
    .block-container {
        max-width: 680px;
        padding: 2.5rem 2rem;
    }

    /* Brand header */
    .brand-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.25rem;
    }
    .brand-dot {
        width: 9px; height: 9px;
        border-radius: 50%;
        background: #22c997;
        display: inline-block;
    }
    .brand-name {
        font-family: 'DM Mono', monospace;
        font-size: 13px;
        color: #8b8fa8;
        letter-spacing: 0.05em;
    }
    .brand-badge {
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        background: rgba(34, 201, 151, 0.12);
        color: #22c997;
        border: 1px solid rgba(34, 201, 151, 0.25);
        border-radius: 4px;
        padding: 2px 7px;
    }

    /* Page title */
    .page-title {
        font-size: 28px;
        font-weight: 600;
        color: #f0f2f5;
        margin: 0.5rem 0 0.3rem;
        line-height: 1.2;
    }
    .page-subtitle {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    /* Section labels */
    .section-label {
        font-size: 11px;
        font-weight: 500;
        color: #555b70;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }

    /* Mode description cards */
    .mode-desc-card {
        background: #181c27;
        border: 1px solid #252b3b;
        border-radius: 10px;
        padding: 12px 16px;
        font-family: 'DM Mono', monospace;
        font-size: 12px;
        color: #6b7280;
        line-height: 1.7;
        margin-bottom: 1.5rem;
    }
    .mode-desc-card strong { color: #c1c8db; font-weight: 500; }

    /* Example pill buttons */
    .stButton > button {
        background: #181c27 !important;
        border: 1px solid #252b3b !important;
        border-radius: 20px !important;
        color: #8b8fa8 !important;
        font-size: 12px !important;
        font-family: 'DM Sans', sans-serif !important;
        padding: 4px 14px !important;
        transition: all 0.15s !important;
        height: auto !important;
    }
    .stButton > button:hover {
        border-color: #3d4558 !important;
        color: #c1c8db !important;
        background: #1e2333 !important;
    }

    /* Main analyze button override */
    .analyze-btn-container .stButton > button {
        background: #f0f2f5 !important;
        color: #0f1117 !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        padding: 10px 24px !important;
        width: 100%;
        height: 48px !important;
    }
    .analyze-btn-container .stButton > button:hover {
        background: #d8dce5 !important;
    }

    /* Text area */
    .stTextArea > div > div > textarea {
        background: #181c27 !important;
        border: 1px solid #252b3b !important;
        border-radius: 10px !important;
        color: #c1c8db !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14px !important;
    }

    /* Selectbox / radio */
    .stRadio > div {
        background: #181c27;
        border-radius: 10px;
        padding: 4px;
        border: 1px solid #252b3b;
        flex-direction: row;
        gap: 0;
    }
    .stRadio label {
        color: #6b7280 !important;
        font-size: 13px !important;
    }

    /* Result card */
    .result-card {
        background: #181c27;
        border: 1px solid #252b3b;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-top: 1.5rem;
    }
    .result-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    .result-eyebrow {
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        color: #555b70;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .model-tag {
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        padding: 3px 9px;
        border-radius: 4px;
        border: 1px solid;
    }
    .tag-fast { background: rgba(251,146,60,0.1); color: #fb923c; border-color: rgba(251,146,60,0.25); }
    .tag-smart { background: rgba(96,165,250,0.1); color: #60a5fa; border-color: rgba(96,165,250,0.25); }

    .sentiment-big {
        font-size: 22px;
        font-weight: 600;
        color: #f0f2f5;
    }
    .sentiment-positive { color: #22c997; }
    .sentiment-negative { color: #f87171; }
    .sentiment-neutral  { color: #8b8fa8; }

    .conf-label {
        display: flex;
        justify-content: space-between;
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        color: #555b70;
        margin: 0.75rem 0 4px;
    }
    .conf-bar-bg {
        height: 5px;
        border-radius: 3px;
        background: #252b3b;
        overflow: hidden;
        margin-bottom: 1rem;
    }
    .conf-bar-pos { background: #22c997; height: 100%; border-radius: 3px; }
    .conf-bar-neg { background: #f87171; height: 100%; border-radius: 3px; }
    .conf-bar-neu { background: #555b70; height: 100%; border-radius: 3px; }

    .reasoning-text {
        font-size: 12px;
        color: #6b7280;
        border-top: 1px solid #252b3b;
        padding-top: 0.875rem;
        line-height: 1.6;
        font-family: 'DM Mono', monospace;
    }

    /* Divider */
    hr { border-color: #252b3b !important; margin: 1.5rem 0 !important; }

    /* Warning / info banners */
    .stAlert { background: #181c27 !important; border-color: #252b3b !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL LOADING
# Cached so they only load once, not on every rerun.
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading fast model...")
def load_fast_model():
    """Load the TF-IDF vectorizer + Logistic Regression model from disk."""
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("tfidf-vectorizer.pkl")
    return model, vectorizer

@st.cache_resource(show_spinner="Loading transformer (first run may take ~30s)...")
def load_transformer():
    """
    Load the pretrained RoBERTa transformer pipeline.
    Only runs once per session thanks to @st.cache_resource.
    """
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
    )

# ─────────────────────────────────────────────
# TEXT PREPROCESSING
# Cleans raw user input for the fast model.
# Transformer handles raw text natively, so we skip this for Smart mode.
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalize text: lowercase, strip URLs, handles, special chars."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)   # remove URLs
    text = re.sub(r'@\w+', '', text)                        # remove @mentions
    text = re.sub(r'[^a-z\s]', ' ', text)                  # keep letters only
    text = ' '.join(text.split())                           # collapse whitespace
    return text

# ─────────────────────────────────────────────
# PREDICTION LOGIC
# Two separate functions — one per mode.
# Returns: (label: str, confidence: float)
# ─────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.55  # Below this → classify as neutral (fast mode only)

def predict_fast(text: str, model, vectorizer) -> tuple[str, float]:
    """
    Fast prediction pipeline:
    1. Clean the text
    2. Vectorize with TF-IDF
    3. Get class probabilities from Logistic Regression
    4. If max probability is too low → return neutral (avoids overconfident wrong guesses)
    """
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    probs = model.predict_proba(vector)[0]
    confidence = float(np.max(probs))
    label = model.classes_[np.argmax(probs)]

    if confidence < CONFIDENCE_THRESHOLD:
        return "neutral", confidence
    return label.lower(), confidence


def predict_smart(text: str, transformer) -> tuple[str, float]:
    """
    Smart (transformer) prediction:
    - Passes raw text directly — no cleaning needed
    - The RoBERTa model understands context, punctuation, and capitalization
    - Maps model's label codes (LABEL_0/1/2) to human-readable names
    """
    result = transformer(text[:512])[0]  # limit to 512 tokens (model max)
    raw_label = result["label"].lower()
    confidence = float(result["score"])

    # cardiffnlp/twitter-roberta uses LABEL_0/1/2 → negative/neutral/positive
    label_map = {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}
    label = label_map.get(raw_label, raw_label)

    return label, confidence

# ─────────────────────────────────────────────
# UI HELPERS
# Small functions that return HTML strings for result cards.
# ─────────────────────────────────────────────

SENTIMENT_EMOJI = {"positive": "😊", "negative": "😠", "neutral": "😐"}
SENTIMENT_COLOR = {"positive": "sentiment-positive", "negative": "sentiment-negative", "neutral": "sentiment-neutral"}
BAR_CLASS      = {"positive": "conf-bar-pos", "negative": "conf-bar-neg", "neutral": "conf-bar-neu"}

def render_result_card(label: str, confidence: float, mode: str, note: str = "") -> str:
    """Build an HTML result card string for st.markdown(... unsafe_allow_html=True)."""
    emoji = SENTIMENT_EMOJI.get(label, "🤔")
    color_class = SENTIMENT_COLOR.get(label, "")
    bar_class = BAR_CLASS.get(label, "conf-bar-neu")
    pct = round(confidence * 100)
    tag_class = "tag-fast" if mode == "fast" else "tag-smart"
    tag_label = "⚡ fast model" if mode == "fast" else "🧠 smart model"
    note_html = f'<div class="reasoning-text">{note}</div>' if note else ""

    return f"""
    <div class="result-card">
        <div class="result-top">
            <span class="result-eyebrow">Result</span>
            <span class="model-tag {tag_class}">{tag_label}</span>
        </div>
        <div>
            <span style="font-size:28px; line-height:1">{emoji}</span>
            <span class="sentiment-big {color_class}">&nbsp; {label.capitalize()}</span>
        </div>
        <div class="conf-label">
            <span>confidence</span><span>{pct}%</span>
        </div>
        <div class="conf-bar-bg">
            <div class="{bar_class}" style="width:{pct}%"></div>
        </div>
        {note_html}
    </div>
    """

# ─────────────────────────────────────────────
# MAIN UI — PAGE HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="brand-row">
    <span class="brand-dot"></span>
    <span class="brand-name">SentimentLens</span>
    <span class="brand-badge">v1.0</span>
</div>
<p class="page-title">Text Sentiment Analysis</p>
<p class="page-subtitle">Analyze tone and emotion using two different AI approaches</p>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODE SELECTOR
# ─────────────────────────────────────────────

MODE_DESCRIPTIONS = {
    "⚡ Fast": (
        "<strong>⚡ Fast Mode</strong> — TF-IDF + Logistic Regression. "
        "Lightweight, low-latency classification with a confidence threshold to catch uncertain predictions."
    ),
    "🧠 Smart": (
        "<strong>🧠 Smart Mode</strong> — RoBERTa Transformer. "
        "Context-aware analysis that understands nuance, sarcasm, and complex phrasing."
    ),
    "↔ Compare": (
        "<strong>↔ Compare Mode</strong> — Runs both models simultaneously. "
        "See how Fast and Smart predictions differ on the same input."
    ),
}

mode = st.radio(
    label="Mode",
    options=["⚡ Fast", "🧠 Smart", "↔ Compare"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown(f'<div class="mode-desc-card">{MODE_DESCRIPTIONS[mode]}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# EXAMPLE BUTTONS
# Clicking an example pre-fills the text area via session state.
# ─────────────────────────────────────────────

EXAMPLES = [
    "I absolutely love this product, it changed my life!",
    "This is the worst experience I've ever had.",
    "The package arrived on Tuesday.",
    "I'm cautiously optimistic but still have concerns.",
]

if "user_text" not in st.session_state:
    st.session_state.user_text = ""

st.markdown('<div class="section-label">Try an example</div>', unsafe_allow_html=True)
cols = st.columns(len(EXAMPLES))
for i, (col, ex) in enumerate(zip(cols, EXAMPLES)):
    short = ex[:30] + "…" if len(ex) > 30 else ex
    if col.button(short, key=f"ex_{i}"):
        st.session_state.user_text = ex

# ─────────────────────────────────────────────
# TEXT INPUT
# ─────────────────────────────────────────────

user_input = st.text_area(
    label="Your text",
    placeholder="Type or paste text here…",
    value=st.session_state.user_text,
    height=110,
    label_visibility="collapsed",
)
st.caption(f"{len(user_input)} characters")

# ─────────────────────────────────────────────
# ANALYZE BUTTON + EDGE CASE HANDLING
# ─────────────────────────────────────────────

st.markdown('<div class="analyze-btn-container">', unsafe_allow_html=True)
run = st.button("Analyze Sentiment", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if run:
    text = user_input.strip()

    # Edge case: empty
    if not text:
        st.warning("Please enter some text before analyzing.")
        st.stop()

    # Edge case: too short to be meaningful
    if len(text) < 5:
        st.warning("Input too short — please enter at least a few words.")
        st.stop()

    # ─── FAST MODE ───
    if mode == "⚡ Fast":
        model, vectorizer = load_fast_model()
        with st.spinner("Running fast model…"):
            label, conf = predict_fast(text, model, vectorizer)

        note = "Confidence was below threshold — classified as neutral." if label == "neutral" and conf < CONFIDENCE_THRESHOLD else ""
        st.markdown(render_result_card(label, conf, "fast", note), unsafe_allow_html=True)

    # ─── SMART MODE ───
    elif mode == "🧠 Smart":
        transformer = load_transformer()
        with st.spinner("Running transformer…"):
            label, conf = predict_smart(text, transformer)

        st.markdown(render_result_card(label, conf, "smart"), unsafe_allow_html=True)

    # ─── COMPARE MODE ───
    else:
        model, vectorizer = load_fast_model()
        transformer = load_transformer()

        with st.spinner("Running both models…"):
            fast_label, fast_conf = predict_fast(text, model, vectorizer)
            smart_label, smart_conf = predict_smart(text, transformer)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(render_result_card(fast_label, fast_conf, "fast"), unsafe_allow_html=True)
        with col_b:
            st.markdown(render_result_card(smart_label, smart_conf, "smart"), unsafe_allow_html=True)

        # Highlight when models disagree
        if fast_label != smart_label:
            st.info(
                f"Models disagree — Fast says **{fast_label}**, Smart says **{smart_label}**. "
                "This often happens with sarcasm, irony, or short ambiguous phrases.",
                icon="💡",
            )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("Built with Streamlit · TF-IDF / Logistic Regression · cardiffnlp/twitter-roberta-base-sentiment")
