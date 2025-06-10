# streamlit_app.py

import streamlit as st
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image
import os
import requests

os.environ["STREAMLIT_SUPPRESS_ST_EMBEDDED_ERROR"] = "1"

# Load ImageNet labels
@st.cache_data
def load_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    return response.text.strip().split('\n')

# Preprocess input image for MobileNetV2
def preprocess_image(image: np.ndarray) -> np.ndarray:
    img = cv2.resize(image, (224, 224))
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Run inference using ONNX
def predict(image: np.ndarray, session, input_name, output_name, labels) -> str:
    input_tensor = preprocess_image(image)
    outputs = session.run([output_name], {input_name: input_tensor})
    preds = outputs[0]
    top_idx = np.argmax(preds)
    return f"{labels[top_idx]} (index: {top_idx})"

# Download model if not already present
def download_model_if_needed(url: str, path: str):
    if os.path.exists(path):
        return
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        raise RuntimeError(f"Failed to download model: {response.status_code}")

# Load model
model_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
model_path = "mobilenetv2-7.onnx"
download_model_if_needed(model_url, model_path)

session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
labels = load_labels()

st.title("🧠 Image Classification from Webcam using ONNX + MobileNetV2")

# Initialize image history in session state
if "history" not in st.session_state:
    st.session_state.history = []

# Webcam capture
picture = st.camera_input("Take a picture")

if picture:
    img = Image.open(picture).convert("RGB")
    image_np = np.array(img)

    pred = predict(image_np, session, input_name, output_name, labels)

    # Add current image and prediction to history
    st.session_state.history.append({
        "image": img,
        "prediction": pred
    })

    st.success(f"🧠 Predicted: **{pred}**")
    st.image(img, caption="Captured Image", use_column_width=True)

# Show prediction history
if st.session_state.history:
    st.markdown("### 📸 Prediction History")
    # Display in reverse so newest first
    for i in range(len(st.session_state.history) - 1, -1, -1):
        item = st.session_state.history[i]
        col1, col2 = st.columns([5, 1])
        with col1:
            st.image(item["image"], caption=f"{len(st.session_state.history) - i}. {item['prediction']}", use_column_width=True)
        with col2:
            if st.button("❌ Remove", key=f"remove_{i}"):
                st.session_state.history.pop(i)
                st.rerun()  # Refresh the UI after deletion
