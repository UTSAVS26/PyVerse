# Logistic Regression using Gradient Descent

This project demonstrates a simple implementation of logistic regression using gradient descent in Python. The code fits a logistic function to transform the linear combination of inputs into probabilities for binary classification by minimizing the logistic loss function.

## Overview

Logistic regression is a fundamental classification algorithm that estimates probabilities of a binary dependent variable based on independent variables. Instead of fitting a straight line, the logistic function (sigmoid) models the output as probabilities constrained between 0 and 1. The model parameters (slope and intercept) are optimized via gradient descent by minimizing the cross-entropy loss.

## Dataset

The original continuous dataset is transformed into binary labels for classification:

```python
x = [7, 5, 8, 7, 2, 12, 5, 9, 4, 11, 19, 9, 10]  
y = [90, 86, 87, 88, 100, 86, 103, 97, 94, 78, 77, 55, 86]  
```

Labels are created by thresholding y at its median value, assigning 1 if y is less than or equal to the median, else 0.

## Gradient Descent Algorithm

The algorithm optimizes model parameters by minimizing the logistic loss (cross-entropy) using gradient descent.

**Functions used:**

```python
# Sigmoid function to predict probabilities
def sigmoid(z):
    ...

# Prediction function using slope and intercept
def y_pred(slope, intercept, x):
    # Applies sigmoid to linear combination of inputs
    ...

# Cost function (cross-entropy loss)
def cost(x, y, slope, intercept):
    # Calculates average cross-entropy over dataset
    ...

# Gradient functions for intercept and slope
def err_intercept(x, y, slope, intercept):
    ...
    
def err_slope(x, y, slope, intercept):
    ...

# Gradient Descent function
def gradient_descent(x, y, lr, epochs=500):
    # Iteratively updates slope and intercept to minimize cost
    ...
```

## Requirements

- Python 3.x
- numpy
- matplotlib

Installation:

```bash
pip install numpy matplotlib
```

## Usage

Clone the repository:

```bash
git clone https://github.com/your-username/repo-name.git
```

Navigate to the project directory:

```bash
cd Logistic-Regression
```

Run the Python script or Jupyter notebook to train the model and visualize the results:

```bash
jupyter notebook
```

OR

```bash
python logistic_regression.py
```

## Results

After training, the output includes:

- Optimized slope and intercept values.

- Plot of data points as binary labels along with predicted probabilities.

- Observed decrease in the cross-entropy cost during training.

Example output snapshot for first 20 epochs:

```text
Epoch 0: slope=0.0234, intercept=0.0301, cost=0.6931  
Epoch 1: slope=0.0417, intercept=0.0547, cost=0.6860  
...  
```

The final model can be used to predict probabilities for the binary classification task based on new input values.