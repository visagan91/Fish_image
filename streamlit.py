# streamlit.py
import os
import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Fish Classifier", page_icon="🐟", layout="centered")
st.title("🐟 Fish Species Classifier")
st.write("Upload an image of a fish to predict its category and see confidence scores.")

MODEL_PATH = "models/best_model.keras"
LABELS_PATH = "models/class_indices.json"
IMG_SIZE = (224, 224)
TOP_K_DEFAULT = 3

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at `{MODEL_PATH}`. Run training/export first.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_labels():
    if not os.path.exists(LABELS_PATH):
        st.error(f"Label map not found at `{LABELS_PATH}`.")
        st.stop()
    with open(LABELS_PATH) as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
    # Ensure labels are ordered by class index
    labels = [id2label[i] for i in range(len(id2label))]
    return labels

def preprocess_image(img: Image.Image, size=IMG_SIZE):
    """No extra rescale here — model already includes preprocessing."""
    img = img.convert("RGB").resize(size)
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr

model = load_model()
labels = load_labels()

uploaded = st.file_uploader("Upload a fish image (jpg/png)", type=["jpg","jpeg","png"])
top_k = st.slider("Top-K predictions", 1, min(5, len(labels)), TOP_K_DEFAULT)

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_container_width=True)

    with st.spinner("Predicting..."):
        x = preprocess_image(img, IMG_SIZE)
        probs = model.predict(x, verbose=0)[0]  # softmax outputs

    # Top-K
    idxs = np.argsort(probs)[::-1][:top_k]
    top = [(labels[i], float(probs[i])) for i in idxs]

    st.subheader("Prediction")
    top_label, top_conf = top[0]
    st.markdown(f"**{top_label}** — confidence: **{top_conf:.2%}**")

    st.subheader(f"Top-{top_k} Confidence")
    for lbl, p in top:
        st.write(f"{lbl}: {p:.2%}")
        # Streamlit progress bars expect 0–100 ints; map prob -> %
        st.progress(int(p * 100))
