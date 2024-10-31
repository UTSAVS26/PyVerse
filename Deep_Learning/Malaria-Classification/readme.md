# Malaria Cell Classification

This project involves classifying malaria-infected and uninfected cells using different machine learning and deep learning approaches. The primary goal is to automate the identification of parasitized and uninfected cells using microscopic images.

## Overview

Malaria is a critical disease caused by parasites, and detecting it in cells is a time-consuming process for healthcare workers. This project aims to automate the process by using machine learning and deep learning models to classify cells as parasitized or uninfected from microscope images.

## Dataset

- [Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
## Models

### MLP
- **Performance**: 
  - Accuracy on training data: ~65%
  - Accuracy on test data: ~65%
  
  While the model performs consistently on both train and test sets, the accuracy is too low to depend on for classification.

### CNN
- **Performance**: 
  - Accuracy on training data: 96%
  - Accuracy on test data: 94%
  
  The CNN model performs well with high accuracy on both training and test datasets.

### CNN with Regularization
- **Performance**: 
  - Accuracy on training data: 96%
  - Accuracy on test data: 94%
  
  Adding regularization to the CNN model had no significant effect on the performance, resulting in similar accuracy to the base CNN model.

### Hyperparameter Tuning
- **Performance**: 
  - Improved model performance, but hyperparameter tuning was time-consuming due to the large number of permutations tried. It was challenging to achieve faster training times with exhaustive tuning.

### Transfer Learning (VGG19)
- **Performance**: 
  - Using transfer learning with the VGG19 architecture was applied for 1 epoch due to high computational cost.
  - The model shows promise, but training for more epochs is needed for better performance.

## Signature

Vivek Prakash

[Linkedin](https://www.linkedin.com/in/vivek-prakash-b46830283/)
