# Chess Piece Recognization System ♞

This project is a Convolutional Neural Network (CNN) based system that recognizes chess pieces from images.
The system uses MobileNetV2 transfer learning and can classify the following chess pieces:
- Bishop
- King
- Knight
- Pawn
- Queen
- Rook

The model was trained using image classification techniques and deployed using Streamlit so it can run on both PC and mobile browsers.

---

## Live Application

After deployment, the application will be available at:

Streamlit Link:


---

## Dataset

Dataset used for training:

https://www.kaggle.com/datasets/niteshfre/chessman-image-dataset

Download the dataset and place it in:

Chessman-image-dataset/Chess/

---

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- Streamlit
- OpenCV
- NumPy

---

## How to Run Locally

1. Install dependencies:

pip install -r requirements.txt

2. Run the application:

streamlit run app.py

---

## Model Information

Model: MobileNetV2 Transfer Learning  
Input Size: 320 × 320  
Classes: 6 Chess Pieces

---

## Author

Chess Piece Recognization System Project