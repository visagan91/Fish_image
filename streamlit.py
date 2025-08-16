import streamlit as st
from infer_utils import Predictor

st.set_page_config(page_title="Fish Classifier", page_icon="🐟", layout="centered")
st.title("🐟 Fish Species Classifier")
st.write("Upload an image of a fish to predict its category and see confidence scores.")

@st.cache_resource
def load_predictor():
    return Predictor("models/best_model.h5", "models/class_indices.json")

predictor = load_predictor()

uploaded = st.file_uploader("Upload a fish image (jpg/png)", type=["jpg","jpeg","png"])
if uploaded is not None:
    temp_path = f"/tmp/{uploaded.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.image(temp_path, caption="Uploaded image", use_container_width=True)

    with st.spinner("Predicting..."):
        preds = predictor.predict(temp_path, top_k=3)

    st.subheader("Prediction")
    top_label, top_conf = preds[0]
    st.markdown(f"**{top_label}** — confidence: **{top_conf:.2%}**")

    st.subheader("Top-3 Confidence")
    for lbl, p in preds:
        st.progress(p)
        st.write(f"{lbl}: {p:.2%}")
