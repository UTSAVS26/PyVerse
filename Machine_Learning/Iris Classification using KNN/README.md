# Iris Dataset Classification Using K-Nearest Neighbors (KNN)

This project is designed to help beginners understand how the K-Nearest Neighbors (KNN) algorithm works by applying it to the famous Iris dataset. The Iris dataset is often used for learning machine learning algorithms because of its simplicity and well-defined structure.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Algorithm](#algorithm)
- [Installation](#installation)
- [Usage](#usage)

---

## Overview
In this project, we classify the Iris dataset into one of three species: **Setosa**, **Versicolor**, and **Virginica**. K-Nearest Neighbors (KNN) is used for classification, which is a simple and effective machine learning algorithm. We will explore the dataset, preprocess the data, and evaluate the model’s performance.

---

## Dataset
The Iris dataset consists of 150 samples, each having four features:
- **Sepal Length**
- **Sepal Width**
- **Petal Length**
- **Petal Width**

Each sample belongs to one of three classes:
1. Setosa
2. Versicolor
3. Virginica

You can directly import the dataset from sklearn library 

---

## Algorithm
We use the **K-Nearest Neighbors (KNN)** algorithm, which classifies a sample based on the majority class among its K nearest neighbors.

### Steps:
1. **Data Loading**: Load the Iris dataset and inspect the structure.
2. **Data Preprocessing**: Split the data into training and testing sets.
3. **Model Training**: Apply KNN to the training data.
4. **Model Evaluation**: Use accuracy, precision, recall, and F1-score to evaluate the model.
5. **Hyperparameter Tuning**: Optimize the value of K to improve classification accuracy.

---

## Installation
To get started with the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/UppuluriKalyani/ML-Nexus/tree/main/Supervised%20Learning/K%20Nearest%20Neighbors/Iris%20Dataset%20Classification
2. **Install dependencies**:
   Ensure you have Python 3 installed. Then, install the necessary Python libraries using pip:
   ```bash
   pip install -r requirements.txt
## Usage
After installing the required dependencies, you can run the project and see the results:


irisClassifier.ipynb

 - Change the value of K in the code to see how it affects the classification accuracy:

```python
knn = KNeighborsClassifier(n_neighbors=3)  # Change 3 to any value of K you want to try


