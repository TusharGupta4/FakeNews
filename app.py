import streamlit as st
import numpy as np
import pickle
import re
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# ── LOAD MODEL & TOKENIZER ───────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = load_model('fake_news_model.keras')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_artifacts()

# ── PREPROCESSING ────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.discard('not')
stop_words.discard('no')
stop_words.discard('nor')
stop_words.discard('never')
stop_words.discard("n't")

def preprocess(text):
    review = re.sub('[^a-zA-Z]', ' ', str(text))
    review = review.lower().split()
    review = [lemmatizer.lemmatize(w)
            for w in review
            if w not in stop_words]
    return ' '.join(review)

def predict(title, text):
    # Preprocess both separately
    clean_title = preprocess(title)
    clean_text  = preprocess(text)

    # Tokenize
    title_seq = tokenizer.texts_to_sequences([clean_title])
    text_seq  = tokenizer.texts_to_sequences([clean_text])

    # Pad — same maxlen as training
    title_pad = pad_sequences(title_seq,
                            maxlen=20,
                            padding='post',
                            truncating='post')
    text_pad  = pad_sequences(text_seq,
                            maxlen=200,
                            padding='post',
                            truncating='pre')

    prob = float(model.predict(
        [title_pad, text_pad], verbose=0)[0][0])

    # Threshold 0.70 — tuned from real world testing
    if prob > 0.70:
        label = 'FAKE'
    elif prob < 0.40:
        label = 'REAL'
    else:
        label = 'UNCERTAIN'

    confidence = prob if prob > 0.5 else 1 - prob
    return label, confidence, prob

# ── HEADER ───────────────────────────────────────────────────
st.title("🔍 Fake News Detector")
st.markdown("Powered by **Dual-Branch LSTM** · "
            "WELFake dataset · **95.16% accuracy**")
st.markdown("---")

# ── METRICS ──────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Model Accuracy", "95.16%")
col2.metric("ROC-AUC Score",  "0.9887")
col3.metric("Training Articles", "71,537")

st.markdown("---")

# ── MAIN LAYOUT ──────────────────────────────────────────────
left, right = st.columns([2, 1])

with left:
    st.subheader("Analyse an Article")

    title_input = st.text_input(
        "📰 News Headline / Title:",
        placeholder="e.g. Government raises interest rates..."
    )

    text_input = st.text_area(
        "📄 Article Body (add for better accuracy):",
        height=180,
        placeholder="Paste the full article text here..."
    )

    # Word count
    combined = (title_input + ' ' + text_input).strip()
    if combined:
        wc = len(combined.split())
        if wc < 20:
            st.caption(f"⚠️ Only {wc} words. "
                    f"Add article body for better accuracy.")
        else:
            st.caption(f"✅ {wc} words detected.")

    if st.button("🔍 Analyse", type="primary",
                use_container_width=True):

        if not title_input.strip():
            st.warning("Please enter a headline first.")
        else:
            with st.spinner("Analysing..."):
                label, conf, prob = predict(
                    title_input,
                    text_input if text_input else ""
                )

            st.markdown("### Result")

            if label == "FAKE":
                st.error(
                    f"🚨 FAKE NEWS — Confidence: {conf*100:.1f}%"
                )
                st.progress(prob)
                st.markdown("""
                **Patterns detected:**
                - Sensational or emotionally charged language
                - Vague or anonymous sources
                - Conspiracy-style framing
                - Designed to trigger anger or fear
                """)

            elif label == "REAL":
                st.success(
                    f"✅ REAL NEWS — Confidence: {conf*100:.1f}%"
                )
                st.progress(1 - prob)
                st.markdown("""
                **Patterns detected:**
                - Neutral, measured language
                - References to named officials or research
                - Factual, specific tone
                - No emotional manipulation
                """)

            else:
                st.warning(
                    f"⚠️ UNCERTAIN — Confidence: {conf*100:.1f}%"
                )
                st.progress(prob)
                st.info(
                    "💡 Model is not confident enough. "
                    "Try adding more article body text."
                )

            st.markdown("---")
            st.caption(
                f"Raw probability: {prob:.4f} · "
                f">0.70 = Fake · <0.40 = Real · else = Uncertain"
            )

with right:
    st.subheader("How it works")
    st.markdown("""
    This model uses a **Dual-Branch LSTM** architecture:

    1. **Title Branch** — reads the headline separately
    2. **Text Branch** — reads the article body separately
    3. Both outputs are **merged** and passed to final layer
    4. Sigmoid gives probability of fake vs real

    This prevents the article body from drowning 
    out the headline signal.
    """)

    st.markdown("---")
    st.subheader("Architecture")
    st.code("""
Title → Embedding → LSTM(64) ──┐
                                ├─ Concat
Text  → Embedding → LSTM(64) ──┘
                ↓
        Dense(64) + Dropout
                ↓
        Dense(1) + Sigmoid
    """)

    st.markdown("---")
    st.subheader("Tips")
    st.markdown("""
    - ✅ Always add article body text
    - ✅ Works best on English news
    - ✅ 20+ words gives reliable results
    - ⚠️ Detects writing style, not facts
    - ⚠️ Always verify from trusted sources
    """)

    st.markdown("---")
    st.caption(
        "Built with TensorFlow, Keras, NLTK & Streamlit"
    )