# 🖊️ Digit Recognizer Neural Network

Welcome to the **Digit Recognizer Neural Network** repository! This project implements a simple two-layer neural network to recognize handwritten digits from the MNIST dataset. The dataset is available on [Kaggle](https://www.kaggle.com/c/digit-recognizer).

---

## 📦 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Code Walkthrough](#code-walkthrough)
- [Example](#example)

---

## 📖 Overview

The model architecture consists of:
- **Input Layer**: 784 units (one for each pixel in a 28x28 image).
- **Hidden Layer**: 10 units with **ReLU** activation.
- **Output Layer**: 10 units with **softmax** activation for digit classification (0-9).

The network employs **forward propagation** and **backward propagation** for training and updates its parameters using **gradient descent**.

---

## ✨ Features

- **Data Handling**: Loads and preprocesses the MNIST dataset.
- **Model Training**: Implements forward and backward propagation for training.
- **Prediction**: Generates predictions for test images with visualization.
- **Accuracy Evaluation**: Computes the model's accuracy on the development set.

---

## 📋 Requirements

Ensure you have the following libraries installed:

```bash
pip install numpy pandas matplotlib
```

# ⚙️ Usage

## Data Preparation
- Load the training dataset from a CSV file.
- Shuffle the data and split it into training and development sets.

## Model Implementation
- Neural network functions are defined for:
  - Initialization
  - Forward propagation
  - Backward propagation
  - Parameter updates
  - Prediction
- Train the model using gradient descent over a specified number of iterations.

## Prediction
- Make predictions on test images and evaluate accuracy on the development set.
- Visualize predictions alongside actual labels.

---

# 🛠️ Code Walkthrough

## Data Loading
- Loads the dataset and prepares training and validation datasets.

## Model Functions
- **`init_params()`**: Initializes weights and biases.
- **`forward_prop()`**: Computes forward propagation to get predictions.
- **`backward_prop()`**: Computes gradients for backward propagation.
- **`update_params()`**: Updates weights and biases based on gradients.
- **`get_predictions()`**: Returns the index of the highest predicted probability.
- **`get_accuracy()`**: Computes the accuracy of predictions.

## Training
- Calls `gradient_descent()` to train the model.

## Testing
- Allows for visual testing of predictions on individual images.

---

# 🔍 Example
To test the model's predictions on the training set, you can use the following code snippet:

```python
test_prediction(0, W1, b1, W2, b2)
