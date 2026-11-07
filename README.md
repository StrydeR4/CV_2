# Laptop vs Not Laptop Image Classification

## Project Overview

This project is a binary image classification system that identifies whether an image contains a laptop.

The project compares three CNN-based approaches:
- Baseline CNN
- MobileNetV2 with Transfer Learning
- EfficientNetB0 with Transfer Learning

## Dataset

The dataset contains 200 images:
- Laptop: 100 images
- Not Laptop: 100 images

Dataset split:
- Training: 140 images
- Validation: 30 images
- Testing: 30 images

## Models and Results

| Model | Test Accuracy |
|---|---:|
| Baseline CNN | 76.67% |
| MobileNetV2 Frozen | 96.67% |
| MobileNetV2 Fine-Tuned | 96.67% |
| EfficientNetB0 | 96.67% |

## Explainability

Grad-CAM was used to visualize image regions that contributed to model predictions.

## Web Application

The application was developed using Gradio.
Users can upload an image and receive:
- Predicted class
- Prediction confidence
- Grad-CAM visualization

## Technologies

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Gradio
- Google Colab

## How to Run

1. Install the required dependencies.
2. Download the trained MobileNetV2 model.
3. Run the application:

   python app/app.py

## Limitations

The dataset is relatively small and contains only 200 images.
The test set contains 30 images, so each individual prediction has a noticeable effect on the reported accuracy.
