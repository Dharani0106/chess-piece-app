import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# PAGE SETTINGS
st.set_page_config(
    page_title="Chess Recognization",
    layout="wide"
)

# HEADER
st.markdown(
    "<h1 style='text-align: center;'>♞ CHESS PIECE RECOGNIZATION SYSTEM</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center;'>Upload a chess piece image for recognition</h3>",
    unsafe_allow_html=True
)
st.markdown("---")

# LOAD MODEL
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "chess_model_320.h5"
    )

model = load_model()
IMG_SIZE = 320

# CLASS LABELS
class_labels = [
    'Bishop',
    'King',
    'Knight',
    'Pawn',
    'Queen',
    'Rook'
]

# FILE UPLOADER
uploaded_file = st.file_uploader(
    "Upload Chess Image",
    type=["jpg","png","jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # PREPROCESSING
    img_resized = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    # PREDICTION
    prediction = model.predict(img_resized)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # LAYOUT
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown(
            "<h2><b>PREDICTION RESULT</b></h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h1 style='color:green;'><b>{predicted_class}</b></h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h2><b>Confidence : {round(confidence,2)} %</b></h2>",
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.markdown(
            "<b>Model :</b> MobileNetV2 CNN",
            unsafe_allow_html=True
        )
        st.markdown(
            "<b>Input Size :</b> 320 × 320",
            unsafe_allow_html=True
        )

    with col2:
        st.image(
            image,
            width=450
        )