# Wine Quality Prediction

This project implements various machine learning models to predict wine quality based on physicochemical properties. The models classify wines into binary categories (high quality vs. low quality) using features like acidity, pH, alcohol content, and more.

## Project Overview

The project uses a dataset containing various chemical properties of wines and their quality ratings. The quality ratings are binarized into two categories:

- 0: Lower quality (rating ≤ 5)
- 1: Higher quality (rating > 5)

## Features

The following features are used for prediction:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Wine type (red/white)

## Models Implemented

The project implements and compares six different machine learning models:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN) Classifier
5. Support Vector Classifier (SVC)
6. Gradient Boosting Classifier

## Technical Implementation

### Data Preprocessing

- Handling missing values
- Feature scaling using MinMaxScaler
- Dimensionality reduction using PCA
- One-hot encoding for categorical variables

### Model Pipeline

Each model uses a consistent pipeline that includes:

1. Feature scaling
2. PCA transformation
3. Model training and prediction

## Project Structure

```
project/
│
├── winequalityN.csv          # Input dataset
├── trained_models/           # Directory containing saved models
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   ├── knn_model.pkl
│   ├── svc_model.pkl
│   └── gradient_boosting_model.pkl
└── wine_quality_prediction.py # Main script
```

## Usage

### Loading a Saved Model

```python
def load_model(model_name):
    """
    Load a saved model from the trained_models directory

    Parameters:
    model_name (str): Name of the model to load (without '_model.pkl')

    Returns:
    object: The loaded model pipeline
    """
    filename = f'trained_models/{model_name}_model.pkl'
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model
```

### Making Predictions

```python
# Example usage
model = load_model('random_forest')
predictions = model.predict(X_test)
```

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Model Performance

The project includes various evaluation metrics for each model:

- Accuracy scores
- Classification reports (precision, recall, F1-score)
- Visual comparisons of model performance

## Visualization

The project includes several visualization components:

- Feature distribution plots
- Correlation heatmaps
- Bivariate analysis plots
- Model performance comparison plots

## Future Improvements

Potential areas for enhancement:

1. Hyperparameter tuning for each model
2. Feature selection optimization
3. Ensemble method exploration
4. Cross-validation implementation
5. Addition of more advanced models
6. API development for model deployment
