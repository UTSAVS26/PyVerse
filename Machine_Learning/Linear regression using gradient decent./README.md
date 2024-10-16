# Linear Regression using Gradient Descent

This project demonstrates a simple implementation of linear regression using gradient descent in Python. The code aims to fit a straight line to a given dataset by minimizing the cost function.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Gradient Descent Algorithm](#gradient-descent-algorithm)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)

## Overview

Linear regression is a basic machine learning technique to predict a dependent variable (y) from an independent variable (x). We achieve this by fitting a straight line (y = mx + c) to the data points, where we find the best-fit slope and intercept by minimizing the error using gradient descent.

## Dataset

We use the following dataset (stored directly in the code) for training:

```python
x = [7, 5, 8, 7, 2, 12, 5, 9, 4, 11, 19, 9, 10]
y = [90, 86, 87, 88, 100, 86, 103, 97, 94, 78, 77, 55, 86]
```

## Gradient Descent Algorithm
The code uses a gradient descent optimization technique to minimize the sum of squared errors (SSE) between the predicted and actual values.

###Functions used:
```python
# Cost function
def cost(x, y, slope, intercept):
    # Calculates sum of squared errors (SSE)
    ...
    
# Gradient functions to compute the errors for slope and intercept
def errIntercept(x, y, slope, intercept):
    ...
    
def errSlope(x, y, slope, intercept):
    ...
    
# Gradient Descent function
def gd(x, y, lr):
    # Iteratively adjusts slope and intercept to minimize cost
    ...
```


## Requirements

- `Python 3.x`
- `numpy`
- `matplotlib`
  *Installation -*
  ```python
  pip install numpy matplotlib
  ```

## Usage
1) Clone the repository:

```
git clone https://github.com/your-username/linear-regression-gradient-descent.git
```
2) Navigate to the project directory:

```
cd linear-regression-gradient-descent
```
3) Run the Jupyter Notebook or Python script to train the model and visualize the results:

```
jupyter notebook
```
OR

```
python linear_regression.py
```
## Results
After running the algorithm, the output will include:

- Optimized slope and intercept for the regression line.
- The R-squared value to measure the goodness of fit.
- A visualization of the data points along with the fitted line.
The final slope and intercept obtained through gradient descent are as follows:

```
Slope: -1.47
Intercept: 99.01
```
