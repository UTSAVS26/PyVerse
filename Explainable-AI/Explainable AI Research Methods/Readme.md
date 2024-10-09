# README.md

# Project Title: **Iris Flower Classification with Explainable AI (XAI) Techniques**

## Goal:
The aim of this project is to classify the Iris flower species using machine learning models, specifically Random Forest, and apply various Explainable AI (XAI) techniques such as LIME, SHAP, Counterfactuals, and model visualization to interpret and explain the model's predictions.

## Dataset:
- **Dataset:** Iris Flower Dataset (Scikit-learn)
- **Features:** 4 features - Sepal Length, Sepal Width, Petal Length, Petal Width
- **Target:** 3 classes - Setosa, Versicolor, Virginica

## Description:
This project focuses on training a Random Forest classifier on the famous Iris dataset and employing different XAI techniques to explain the model's predictions. These explanations help in understanding how the model makes decisions, what factors influence the predictions, and how changes in input features can affect outcomes.

## What I Have Done
## XAI Methods Used:

### 1. LIME (Local Interpretable Model-Agnostic Explanations):
- LIME explains individual predictions by perturbing input data and observing how the model's predictions change.
- A Random Forest classifier was trained on the Iris dataset.
- LIME was applied to explain the prediction for a specific test instance, highlighting important features that influenced the model's decision.

### 2. SHAP (SHapley Additive exPlanations):
- SHAP values provide a unified measure of feature importance by showing the contribution of each feature to the final prediction.
- SHAP was used to explain the overall model behavior and visualize how each feature contributes to the prediction for a specific instance.

### 3. Counterfactual Explanations (DiCE):
- Counterfactuals offer alternate instances where the prediction outcome changes, helping to understand how the input features should be altered to achieve a different prediction.
- DiCE was used to generate counterfactual examples for a test instance in the Iris dataset, showing how changes in features could alter the class prediction.

### 4. Model Visualization:
- The Random Forest model was visualized using the first decision tree in the ensemble.
- This allows for better interpretation of how the model splits features to make decisions about the target class.

## Steps Undertaken:

### 1. Data Loading & Preparation:
- Loaded the Iris dataset from Scikit-learn.
- Split the dataset into training and test sets using an 80/20 split.

### 2. Model Training:
- Trained a **Random Forest Classifier** on the training data using 100 estimators.
- Evaluated the model on the test set.

### 3. Applying XAI Methods:
- **LIME:** Applied to the test set to explain individual predictions.
- **SHAP:** Visualized feature contributions for individual instances.
- **Counterfactuals (DiCE):** Generated alternative feature sets that lead to different predictions.
- **Model Visualization:** Displayed a decision tree from the trained Random Forest model.

## Libraries Required:
- **Scikit-learn**: For data loading, preprocessing, and model training.
- **LIME**: For local explanation of model predictions.
- **SHAP**: For global and local feature importance explanations.
- **DiCE**: For generating counterfactual explanations.
- **Matplotlib**: For data and model visualization.
- **Graphviz**: For visualizing the decision tree from the Random Forest model.

## Results:
- LIME provided insight into the most influential features for specific predictions.
- SHAP values allowed us to understand how individual features contributed to the overall model predictions.
- Counterfactuals helped demonstrate how changing feature values could lead to different predictions.
- The visualization of the decision tree made the model's decision-making process more transparent.

## Insights:
- XAI methods like LIME, SHAP, and DiCE provide valuable insights into how machine learning models make predictions, helping in model validation and ensuring trust in the decisions made by the model.
- Visualizing a model's decision tree adds transparency to complex models like Random Forest, making it easier to understand.

