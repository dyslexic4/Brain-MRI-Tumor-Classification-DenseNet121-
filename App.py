"""
Brain MRI Classifier: Streamlit frontend
Run:  streamlit run app.py
Needs brain_mri_densenet.joblib (from block 1) in the same folder.
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

JOBLIB_PATH = "brain_mri_densenet.joblib"
LOW_CONFIDENCE = 0.70

INFO = {
    "notumor":    "No tumor pattern detected in this scan.",
    "glioma":     "Tumor arising from glial cells in the brain or spinal cord.",
    "meningioma": "Tumor arising from the meninges, the membranes covering the brain.",
    "pituitary":  "Tumor in the pituitary gland at the base of the brain.",
}

st.set_page_config(page_title="Brain MRI Classifier", page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner="Loading model…")
def load_bundle():
    b = joblib.load(JOBLIB_PATH)
    model = tf.keras.models.model_from_json(b["architecture"])
    model.set_weights(b["weights"])
    return model, b["class_names"], tuple(b["img_size"]), b.get("history", {})


def predict(model, img, img_size):
    arr = np.array(img.convert("RGB").resize(img_size), dtype=np.float32)
    return model.predict(np.expand_dims(arr, 0), verbose=0)[0]


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🧠 Brain MRI Classifier")
    st.caption("DenseNet121 · 4-class transfer learning")
    st.markdown("**Classes**")
    for c in ["notumor", "glioma", "meningioma", "pituitary"]:
        st.markdown(f"- {c}")
    st.divider()
    st.warning(
        "Research and educational demo only. This is **not** a medical device "
        "and must not be used for diagnosis. Consult a qualified radiologist."
    )

if not os.path.exists(JOBLIB_PATH):
    st.error(f"`{JOBLIB_PATH}` not found. Run block 1 first and keep it next to this file.")
    st.stop()

model, class_names, img_size, history = load_bundle()

st.title("Brain Tumor MRI Classification")
tab_predict, tab_history = st.tabs(["🔍 Predict", "📈 Training history"])

# ---------------- Predict tab ----------------
with tab_predict:
    uploaded = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Upload a brain MRI slice (JPG or PNG) to get a prediction.")
    else:
        img = Image.open(uploaded)
        proba = predict(model, img, img_size)
        top = int(np.argmax(proba))
        label, conf = class_names[top], float(proba[top])

        left, right = st.columns([1, 1.2], gap="large")
        with left:
            st.image(img, caption=uploaded.name, use_container_width=True)

        with right:
            st.subheader("Result")
            c1, c2 = st.columns(2)
            c1.metric("Prediction", label.upper())
            c2.metric("Confidence", f"{conf*100:.1f}%")
            st.write(INFO[label])

            if conf < LOW_CONFIDENCE:
                st.warning("Low confidence. Treat this prediction with caution.")
            elif label != "notumor":
                st.error(f"Model suggests a possible {label}. Needs clinical review.")
            else:
                st.success("No tumor pattern detected by the model.")

            st.markdown("**Class probabilities**")
            for name, p in sorted(zip(class_names, proba), key=lambda t: -t[1]):
                st.progress(float(p), text=f"{name}: {p*100:.1f}%")

# ---------------- History tab ----------------
with tab_history:
    if not history:
        st.info("No training history stored in the model file.")
    else:
        h1, h2 = history.get("phase1", {}), history.get("phase2", {})

        def merged(key):
            return list(h1.get(key, [])) + list(h2.get(key, []))

        n1 = len(h1.get("accuracy", []))
        acc = pd.DataFrame({"Train": merged("accuracy"), "Validation": merged("val_accuracy")})
        loss = pd.DataFrame({"Train": merged("loss"), "Validation": merged("val_loss")})
        acc.index = loss.index = range(1, len(acc) + 1)

        st.caption(f"Epochs 1–{n1}: head only · epochs {n1+1}–{len(acc)}: fine-tuning")
        a, l = st.columns(2)
        with a:
            st.subheader("Accuracy")
            st.line_chart(acc)
        with l:
            st.subheader("Loss")
            st.line_chart(loss)

        if len(acc):
            st.metric("Final validation accuracy", f"{acc['Validation'].iloc[-1]*100:.2f}%")