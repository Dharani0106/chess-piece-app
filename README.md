# ♞ Chess Piece Recognization System

This project is a deep learning-based chess piece recognition system that classifies chess pieces from images using a Convolutional Neural Network (CNN). 
The model uses MobileNetV2 transfer learning and is deployed as a web application accessible from both PC and mobile devices.

## Live Application
https://chess-piece-recognition.streamlit.app

Users can upload a chess piece image and instantly get the predicted piece along with confidence score.

## Features
• Recognizes 6 chess pieces: King, Queen, Knight, Bishop, Rook, Pawn
• Real-time image prediction  
• Shows confidence score  
• Works on PC and mobile browsers  
• Online AI web application

## Model Information
Model: MobileNetV2 CNN  
Input Size: 320 × 320  
Classes: 6 Chess Pieces  
Training Accuracy: ~96%  
Validation Accuracy: ~90%  
Framework: TensorFlow / Keras

## Dataset
Dataset used for training:
https://www.kaggle.com/datasets/niteshfre/chessman-image-dataset
Dataset is not included in this repository due to size limitations.

## Screenshots
### Home Page
![Home](https://github.com/Dharani0106/chess-piece-app/blob/ba7f323bdc233f2b989f4c59ed50c88c5b22cdcc/Home.png)

### Prediction Example
![Prediction](https://github.com/Dharani0106/chess-piece-app/blob/ba7f323bdc233f2b989f4c59ed50c88c5b22cdcc/Prediction.png)

## Technologies Used
Python, TensorFlow, Keras, MobileNetV2, Streamlit, OpenCV, NumPy, Scikit-learn, Matplotlib

## How to Run Locally
Install libraries:
pip install -r requirements.txt

Run application:
streamlit run app.py

## Project Structure
app.py  
requirements.txt  
runtime.txt  
chess_model_320.h5  
Chess Piece Prediction Model.ipynb  
README.md

## Author
Developed by Dharanidharan B
