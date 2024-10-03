# Plant Disease Detection using CNN

This project implements a **Convolutional Neural Network (CNN)** to detect plant diseases using images of plant leaves. The model is built using **TensorFlow** and **Keras**, and the dataset used is sourced from **Kaggle**.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)


## Introduction

Plant diseases can have a devastating effect on agricultural productivity. This project aims to detect plant diseases from images of plant leaves using CNNs, which are well-suited for image classification tasks. By identifying diseases early, we can potentially help farmers take corrective action sooner and minimize crop damage.

## Dataset

The dataset used for this project is sourced from **[Kaggle](https://www.kaggle.com/)**. It contains labeled images of healthy and diseased plant leaves from various plant species, such as:

- Apple
- Potato
- Tomato
- Grape
- And more...

Each image is categorized into one of several classes, including both healthy and various diseased categories.

## Installation

To set up the project environment, first clone the repository, then install the required dependencies listed in `requirements.txt`.

### Clone the Repository
```bash
git clone https://github.com/your-username/ML-Nexus/tree/main/Neural%20Networks/Plant%20Disease%20Detection.git
cd plant-disease-detection
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

Dependencies include:
- TensorFlow
- Keras
- Matplotlib


## Model Architecture

We employ a **Convolutional Neural Network (CNN)** to process the images and classify them into their respective categories. The architecture consists of:

- **Input Layer:** Input size matching the image dimensions.
- **Convolutional Layers:** For feature extraction (with filters for edges, textures, etc.).
- **Pooling Layers:** To reduce spatial dimensions.
- **Fully Connected Layers:** For classification.
- **Output Layer:** Softmax for classification into plant disease categories.

### Example CNN Layer Structure:
```text
1. Conv2D(32 filters, kernel_size=3x3, activation='relu')
2. MaxPooling2D(pool_size=2x2)
3. Conv2D(64 filters, kernel_size=3x3, activation='relu')
4. MaxPooling2D(pool_size=2x2)
5. Flatten()
6. Dense(128, activation='relu')
7. Dense(number_of_classes, activation='softmax')
```

## Training

The model is trained on the Kaggle dataset, which is split into training and validation sets. We use **categorical cross-entropy** as the loss function and **Adam** optimizer for the training process.

To train the model, simply run:

```bash
python train_model.py
```

Key training details:
- **Epochs:** 50 (adjust based on performance)
- **Batch Size:** 32
- **Validation Split:** 10% of the dataset
- **test Split:** 10% of the dataset
- **train Split:** 80% of the dataset

## Results

After training, the model achieves good accuracy in classifying the plant leaves as healthy or diseased. Below are some key metrics from the model:

- **Test Accuracy:** X%
- **Validation Accuracy:** Y%
- **Loss:** Z%

You can view the training process with graphs of accuracy and loss:

```python
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
```

## Usage

Once the model is trained, you can use it to predict plant diseases by passing images of leaves to the trained model.

```python
from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing import image

# Load model
model = load_model('plant_disease_model.h5')

# Load and preprocess image
img = image.load_img('path_to_image.jpg', target_size=(150, 150))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

# Predict
result = model.predict(img)
```
