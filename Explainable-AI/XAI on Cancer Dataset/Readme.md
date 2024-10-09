# Breast Cancer Diagnosis Prediction Using LIME Model

## Goal
The main objective of this project is to predict whether a tumor is benign or malignant using machine learning models and explain the model’s decisions using LIME (Local Interpretable Model-Agnostic Explanations). The dataset used in this project is the **Breast Cancer Dataset** available from Kaggle.

## Dataset
- **Source:** Kaggle
- **Dataset URL:** [Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)
- The dataset contains features related to the physical characteristics of the tumor and the target variable is **diagnosis**, where:
  - 1 represents Malignant
  - 0 represents Benign

## Description
The dataset consists of 30 features, each representing a different attribute of the tumor, such as the radius, texture, perimeter, area, and smoothness. These features are used to predict whether a tumor is malignant or benign. In addition to building a predictive model, we employ **LIME** (Local Interpretable Model-agnostic Explanations) to explain individual predictions, enhancing the interpretability of the model.

## What I Have Done
### 1. Configured Kaggle API
- Set up the Kaggle API for dataset download using the following code:

### 2. Performed Exploratory Data Analysis (EDA)
- Loaded the dataset into a pandas dataframe and analyzed its structure.
- Checked for missing values, data types, and duplicated rows.
- Visualized the distribution of diagnosis values.

### 3. Data Preprocessing
- Converted categorical values (`diagnosis`) to numerical values, where 1 indicates Malignant and 0 indicates Benign.
- Applied one-hot encoding for categorical columns.
- Split the data into training and testing sets using an 80-20 ratio.

### 4. Built and Trained a Decision Tree Model
- Implemented a **Decision Tree Classifier** for the initial classification task.
- Trained the model using the training set (`X_train`, `y_train`) and evaluated its accuracy on the test set (`X_test`, `y_test`).

### 5. Applied the LIME Model
- Applied **LIME** to explain individual predictions of the decision tree model.
- Generated explanations for test instances, which highlight how each feature contributes to the prediction of whether a tumor is benign or malignant.

### 6. Evaluated the Model
- Evaluated the model’s performance using accuracy, confusion matrix, and classification report metrics.
- Used LIME to interpret and explain the model’s predictions for individual cases.

## Models Used
- **Decision Tree Classifier:** A tree-based model used for classification of tumors based on their features.
- **LIME (Local Interpretable Model-Agnostic Explanations):** Used to explain predictions of the machine learning model by generating interpretable explanations for individual instances.

## Libraries Needed
- **Pandas:** For data manipulation.
- **Numpy:** For numerical computations.
- **Scikit-learn:** For building and evaluating the machine learning model.
- **LIME:** For generating explanations for the machine learning model’s predictions.
- **Matplotlib & Seaborn:** For data visualization.

## Insights
This project provides insights into:
- How to build a decision tree model to predict whether a tumor is benign or malignant.
- The application of **LIME** to explain individual predictions and understand the decision-making process of the model.
- Visualizing and interpreting the effect of different features on the predictions.
