# Explainable AI: Using Local Interpretable Model-agnostic Explanations (LIME) & SHapley Additive exPlanations (SHAP)

As reliance on Artificial Intelligence (AI) increases, the explainability of AI systems remains a significant concern. Often described as "black boxes," AI solutions can be challenging to interpret. To address this, Local Interpretable Model-agnostic Explanations (LIME) and SHapley Additive exPlanations (SHAP) are two frameworks designed to enhance our understanding of model behavior. This notebook explores these frameworks and their potential use cases.

## Setting Up

We begin by importing the necessary libraries:

- **NumPy**: for numerical data manipulation
- **Pandas**: for data handling and analysis
- **Matplotlib**: for creating static visualizations
- **Seaborn**: for enhanced data visualizations

```python
# Importing the libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
```

Next, we import the Wine Quality Dataset, a widely recognized classification dataset, using the `read_csv()` function from Pandas.

```python
# Import the dataset
df = pd.read_csv('../input/wine-dataset/wine.csv')

# Viewing the first few rows
df.head()
```

## Preprocessing the Dataset

To start, we check the dimensions of the dataset using the `shape` attribute.

```python
# Checking the dimensions of the dataframe
df.shape
```

This reveals that the dataset consists of 178 rows and 14 columns. 

Next, we check for missing values using the `isnull()` and `sum()` functions.

```python
# Checking for missing values
df.isnull().sum()
```

We find that there are no missing values.

We then examine the data types to determine if any categorical features require encoding.

```python
# Checking for encoding categorical features
df.dtypes
```

With no categorical values present, no encoding is necessary.

Next, we investigate outliers by creating a boxplot of all attributes using the `boxplot()` function.

```python
# Checking for outliers
plt.figure(figsize=(15, 8))
df.boxplot()
plt.title('Boxplot of the Dataset')
plt.xlabel('Attributes')
plt.ylabel('Values')
```

This boxplot indicates a negligible number of outliers.

We then plot the distribution of each column using `kdeplot()` from the Seaborn library to visualize the distribution of the data.

```python
# Plotting the distribution of Alcohol content
plt.figure()
sns.kdeplot(df['Alcohol'])
plt.title('Distribution of variable - Alcohol')
plt.xlabel('Values of Alcohol Content')
```

Subsequent distribution plots for other attributes reveal various shapes and skewnesses, confirming that while some distributions are normal, others exhibit peaks or skewness. Given the limited size of the dataset, we will not perform transformations and will not scale the data, as we intend to build a non-parametric model.

## Model Building

In this notebook, we will build a Random Forest model for classification. We start by splitting the data into independent variables (X) and the dependent variable (y) using the `train_test_split()` function from the `sklearn` library.

```python
# Splitting the data into independent and dependent variables
X = df.drop(columns=['Class'])
y = df['Class']

# Dividing the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=105)
```

Next, we build the model using the `RandomForestClassifier()` from `sklearn`.

```python
# Building the model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=105)
```

To optimize the model, we create a dictionary of parameters and their possible values for optimization. We then employ a grid search to identify the best parameter combination with 10-fold cross-validation.

```python
# Importing the necessary libraries
from sklearn.model_selection import GridSearchCV

# Creating a dictionary and list of their values to optimize the model
params = {
    'n_estimators': [100, 500, 1000],
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
}

# Initiating a grid search to find the most optimum parameters
grid_search = GridSearchCV(model, params, cv=10)

# Fitting the training data
grid_search.fit(X_train, y_train)
```

Next, we retrieve the best estimator from the grid search and fit the training data to this optimized model. We then generate a classification report to evaluate the model's performance.

```python
# Obtaining the best model
model = grid_search.best_estimator_

# Fitting the training data
model.fit(X_train, y_train)

# Obtaining the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, model.predict(X_test)))
```

With our model built and evaluated, we can now analyze its performance.

## Explainability via LIME

Local Interpretable Model-agnostic Explanations (LIME) allows us to interpret individual predictions made by the model. To start, we import the `lime` library and its `lime_tabular` module.

```python
# Importing LIME
import lime
from lime import lime_tabular
```

We then create an instance of the `LimeTabularExplainer` class, passing in our training data, feature names, class names, and setting the mode to 'classification'.

```python
# Creating an instance of the lime tabular explainer
lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train), 
    feature_names=X_train.columns, 
    class_names=['1', '2', '3'], 
    mode='classification'
)
```

Next, we derive the explanation for a specific instance using the `explain_instance` method. The parameters include the data row to explain, the prediction function, the number of top labels to display, and the number of features to consider.

```python
# Obtaining the explanation
explanation = lime_explainer.explain_instance(
    data_row=X_test.iloc[1], 
    predict_fn=model.predict_proba, 
    top_labels=6, 
    num_features=13
)

# Printing out the explanation
explanation.show_in_notebook()
```

The output reveals how the model predicts class membership with associated confidence levels for each class. Each attribute's weight indicates its contribution to the prediction, providing valuable insight into the model's decision-making process.

## Explainability via SHAP

SHapley Additive exPlanations (SHAP) offers another approach for interpreting model predictions by analyzing the contribution of each feature to the overall prediction. This framework uses game theory to assess the impact of features systematically. 

### Next Steps with SHAP

To implement SHAP, we would:

1. Install the SHAP library.
2. Import the necessary modules.
3. Create SHAP values using the trained model.
4. Visualize the SHAP values to understand feature importance and model behavior.

This approach allows for a deeper understanding of how each feature influences the model's predictions, making AI systems more interpretable and trustworthy.

By employing both LIME and SHAP, we can significantly reduce the "black box" nature of AI models, allowing stakeholders to understand, trust, and utilize these powerful tools more effectively.
