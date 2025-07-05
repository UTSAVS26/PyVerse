# Model Information

This project uses a Convolutional Neural Network (CNN) trained to classify waste images into two categories: Organic and Recyclable. The model was trained on a curated dataset containing labeled images of various waste items. Key details:

- **Architecture:** Custom CNN with multiple convolutional and pooling layers, followed by dense layers for classification.
- **Input Size:** Images are resized to 128x128 pixels before being fed to the model.
- **Classes:**
  - Organic: Includes food scraps, leaves, and other biodegradable materials.
  - Recyclable: Includes plastics, metals, paper, and other materials suitable for recycling.
- **Training:**
  - Data augmentation techniques were used to improve generalization.
  - The model was trained using categorical cross-entropy loss and Adam optimizer.
  - Achieved high accuracy on a held-out validation set.
- **Deployment:**
  - The trained model is saved in HDF5 format (`model/model.h5`) and loaded by the Streamlit app for inference.

# Waste Classification App

A Streamlit web application that classifies waste images as either Organic or Recyclable using a deep learning model.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Demo](#demo)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)

## Overview
Waste management is a critical issue for environmental sustainability. This project aims to assist users in correctly classifying waste as either organic or recyclable using a simple web interface powered by deep learning. The app is designed for educational, research, and practical use cases.

## Features
- Upload image functionality (drag-and-drop or file picker)
- Real-time waste classification (Organic or Recyclable)
- Support for JPG, JPEG, and PNG formats
- Simple and intuitive user interface
- Fast and lightweight model for quick predictions

## Dataset

This model was trained on dataset available for download from here : https://www.kaggle.com/datasets/techsash/waste-classification-data

## How It Works
1. User uploads a waste image.
2. The app preprocesses the image and feeds it to a trained deep learning model.
3. The model predicts whether the waste is organic or recyclable.
4. The result is displayed instantly on the web interface.

## Demo
The app is deployed on Streamlit Community Cloud:
[Waste Classification App Demo](https://wasteclassification2645.streamlit.app/)

## Requirements
- Python 3.7+
- TensorFlow
- Streamlit
- Pillow
- NumPy

## Installation
1. **Clone this repository:**
   ```bash
   git clone <repository-url>
   cd Waste-Classification
   ```
2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
   Or install manually:
   ```bash
   pip install tensorflow streamlit pillow numpy
   ```

## Usage
1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
2. **Open your browser and go to:**
   [http://localhost:8501](http://localhost:8501)
3. **Upload an image** and view the classification result.

## Model Details
- The model is a convolutional neural network (CNN) trained on a dataset of waste images.
- It distinguishes between organic (e.g., food scraps, leaves) and recyclable (e.g., plastic, metal, paper) waste.
- For more details on the dataset and training process, see the `model/` and `notebooks/` directories (if available).





