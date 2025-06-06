# streamlit_app.py

import streamlit as st
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image
import tempfile
import os
os.environ["STREAMLIT_SUPPRESS_ST_EMBEDDED_ERROR"] = "1"


# Load ImageNet labels
@st.cache_data
def load_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    import requests
    response = requests.get(url)
    return response.text.strip().split('\n')

# Preprocess input image for MobileNetV2
def preprocess_image(image: np.ndarray) -> np.ndarray:
    img = cv2.resize(image, (224, 224))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
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

# Streamlit UI
st.title("🧠 Object Recognition from Webcam using ONNX + MobileNetV2")

# Upload ONNX model
import os
import requests

model_url = "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
model_path = "mobilenetv2-7.onnx"

# Download model if not already present
if not os.path.exists(model_path):
    with open(model_path, "wb") as f:
        f.write(requests.get(model_url).content)

session = ort.InferenceSession(model_path)
# model_path = "mobilenetv2-7.onnx"
# session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
labels = load_labels()

# Webcam capture
picture = st.camera_input("Take a picture")

if picture:
    img = Image.open(picture).convert("RGB")
    image_np = np.array(img)

    st.image(image_np, caption="Captured Image", use_column_width=True)

    pred = predict(image_np, session, input_name, output_name, labels)
    st.success(f"🧠 Predicted: **{pred}**")
