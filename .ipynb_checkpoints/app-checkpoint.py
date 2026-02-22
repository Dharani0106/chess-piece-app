import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Title
st.title("Chess Piece Recognition System")

st.write("Upload a chess piece image to classify the piece.")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("chess_model_final_optimized.h5")

model = load_model()

# Classes
class_labels = [
    'Bishop',
    'King',
    'Knight',
    'Pawn',
    'Queen',
    'Rook'
]

# Upload file
uploaded_file = st.file_uploader("Upload Chess Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)

    img = cv2.resize(img,(224,224))

    img = img/255.0

    img = np.reshape(img,(1,224,224,3))

    prediction = model.predict(img)

    predicted_class = class_labels[np.argmax(prediction)]

    confidence = np.max(prediction)*100

    st.subheader("Prediction")

    st.write("Piece:", predicted_class)

    st.write("Confidence:", round(confidence,2),"%")