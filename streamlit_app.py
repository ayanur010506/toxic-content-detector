import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import os

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Toxic Content Detector",
    page_icon="🛡️",
    layout="centered"
)

# ─── Constants ──────────────────────────────────────────────────────────────
MODEL_NAME  = "DeepPavlov/rubert-base-cased"
MODEL_PATH  = "rubert_best.pt"          # должен лежать рядом с этим файлом
DEVICE      = torch.device("cpu")       # Windows без GPU → всегда CPU
MAX_LEN     = 128

# ─── Load model (cached — грузится только один раз) ─────────────────────────
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        model_source = "✅ rubert_best.pt loaded (fine-tuned model)"
    else:
        model_source = "⚠️ rubert_best.pt not found — using base ruBERT (not fine-tuned)"
    model.to(DEVICE)
    model.eval()
    return tokenizer, model, model_source

# ─── Preprocessing ──────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^а-яёА-ЯЁa-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─── Prediction ─────────────────────────────────────────────────────────────
def predict(text: str, tokenizer, model):
    cleaned = clean_text(text)
    enc = tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE)
        ).logits
    probs = torch.softmax(logits, dim=1)[0].tolist()
    label = int(torch.argmax(logits, dim=1).item())
    return label, probs[1], probs[0]   # label, toxic_prob, clean_prob

# ─── UI ─────────────────────────────────────────────────────────────────────
st.title("🛡️ Toxic Content Detector")
st.markdown(
    "**Diploma project · Astana IT University**  \n"
    "Multilingual detection (RU + EN) · ruBERT fine-tuned"
)
st.divider()

# Load model
with st.spinner("Loading ruBERT model... (first launch ~30 sec)"):
    tokenizer, model, model_source = load_model()
st.caption(model_source)

st.divider()

# ── Input ────────────────────────────────────────────────────────────────────
st.subheader("Enter text")

col1, col2 = st.columns(2)
with col1:
    if st.button("🟢 Example: clean (EN)"):
        st.session_state["input_text"] = "I really enjoyed spending time with my family today. It was a wonderful evening."
with col1:
    if st.button("🟢 Example: clean (RU)"):
        st.session_state["input_text"] = "Сегодня прекрасный день, надеюсь всё у вас хорошо!"
with col2:
    if st.button("🔴 Example: toxic (EN)"):
        st.session_state["input_text"] = "You are absolutely worthless and should disappear forever."
with col2:
    if st.button("🔴 Example: toxic (RU)"):
        st.session_state["input_text"] = "Ты идиот, убирайся отсюда, никто тебя не хочет видеть!"

user_input = st.text_area(
    label="Text (Russian or English):",
    value=st.session_state.get("input_text", ""),
    height=150,
    placeholder="Paste or type text here..."
)

analyze = st.button("🔍 Analyze", use_container_width=True, type="primary")

# ── Result ───────────────────────────────────────────────────────────────────
if analyze:
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            label, toxic_prob, clean_prob = predict(user_input, tokenizer, model)

        st.divider()
        st.subheader("Result")

        if label == 1:
            st.error(f"🚨 **TOXIC** — confidence {toxic_prob*100:.1f}%")
        else:
            st.success(f"✅ **NON-TOXIC** — confidence {clean_prob*100:.1f}%")

        st.markdown("**Probability scores:**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("🔴 Toxic", f"{toxic_prob*100:.1f}%")
            st.progress(toxic_prob)
        with col_b:
            st.metric("🟢 Non-toxic", f"{clean_prob*100:.1f}%")
            st.progress(clean_prob)

        with st.expander("🔎 Preprocessed text (what the model sees)"):
            st.code(clean_text(user_input))

# ── Sidebar — model info ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Model info")
    st.markdown("""
| Parameter | Value |
|-----------|-------|
| Base model | DeepPavlov/rubert-base-cased |
| Fine-tuning | 3 epochs |
| Batch size | 32 |
| Learning rate | 2e-5 |
| Max length | 128 tokens |
| **F1-Score** | **0.950** |
| **ROC-AUC** | **0.986** |
""")
    st.divider()
    st.markdown("""
**Datasets used:**
- 🇷🇺 `s-nlp/ru_paradetox` — 22 180 examples
- 🇬🇧 Jigsaw Toxic Comments — 16 000 examples
- **Total: ~38 000 examples**
""")
    st.divider()
    st.markdown("""
**Baseline comparison:**

| Model | F1 | AUC |
|-------|----|-----|
| Logistic Regression | 0.804 | 0.889 |
| SVM | 0.845 | 0.912 |
| Random Forest | 0.875 | 0.921 |
| **ruBERT** | **0.950** | **0.986** |
""")
    st.divider()
    st.caption("Astana IT University · 2025")
