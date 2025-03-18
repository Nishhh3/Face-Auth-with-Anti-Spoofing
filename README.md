# Face-Auth-with-Anti-Spoofing
An advanced face authentication system with anti-spoofing capabilities using shape_predictor_68_face_landmarks. This model detects facial landmarks to differentiate between real faces and spoofing attempts, ensuring secure authentication.

## 📌 Overview

This project implements face authentication with anti-spoofing to ensure secure user verification. The system detects real vs. spoofed faces using:

- Pretrained deep learning models for spoof detection
- Eye blink detection as an additional anti-spoofing measure

## 🚀 Features

1. Real-time face authentication using a webcam
2. Spoof detection via deep learning
3. Eye blink detection for liveness verification

## 📥 Installation
1. Clone the Repository
```
git clone https://github.com/Nishhh3/Face-Auth-with-Anti-Spoofing.git
cd Face-Auth-with-Anti-Spoofing
```
2. Install Dependencies
```
pip install flask flask-cors opencv-python dlib numpy pandas speechrecognition pydub torch transformers fuzzywuzzy python-levenshtein scikit-learn mysql-connector-python
```
3. Download Required Models
**Shape Predictor for Facial Landmarks**
Download the shape predictor model from the following link:
[Download Shape Predictor Model](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat)
Move the file into your project directory.
4. Run the Application
```
python anti-spoof.py
```

## 🎯 How It Works

- Face Detection: Detects faces using OpenCV/Dlib.

- Spoof Detection: Classifies input as real or spoof using a deep learning model.

- Eye Blink Detection: Ensures liveness by detecting blinks.


