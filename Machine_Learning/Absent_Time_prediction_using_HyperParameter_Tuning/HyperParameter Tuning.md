# Hyperparameter Tuning

Hyperparameter tuning is a crucial step in machine learning model development. It involves finding the optimal combination of hyperparameters that results in the best performance of the model. In this resource, we will explore two popular methods for hyperparameter tuning: **GridSearchCV** and **RandomizedSearchCV**.

## GridSearchCV

GridSearchCV is a method for hyperparameter tuning that involves exhaustively searching through a grid of possible hyperparameter combinations. It is a brute-force approach that tries all possible combinations of hyperparameters and evaluates the model's performance on each combination.

### How GridSearchCV Works

GridSearchCV works by defining a grid of possible hyperparameter combinations, splitting the data into training and validation sets, performing a grid search over the hyperparameter grid, and selecting the hyperparameters that result in the best performance on the validation data.

### Advantages of GridSearchCV

- Exhaustive search ensures that the optimal hyperparameters are found.
- Easy to implement and interpret.

### Disadvantages of GridSearchCV

- Computationally expensive, especially for large hyperparameter spaces.
- May not be feasible for high-dimensional hyperparameter spaces.

## RandomizedSearchCV

RandomizedSearchCV is a method for hyperparameter tuning that involves randomly sampling the hyperparameter space. It is a more efficient approach than GridSearchCV, especially when the hyperparameter space is large.

### How RandomizedSearchCV Works

RandomizedSearchCV works by defining a distribution for each hyperparameter, splitting the data into training and validation sets, performing a randomized search over the hyperparameter space, and selecting the hyperparameters that result in the best performance on the validation data.

### Advantages of RandomizedSearchCV

- More efficient than GridSearchCV, especially for large hyperparameter spaces.
- Can handle high-dimensional hyperparameter spaces.

### Disadvantages of RandomizedSearchCV

- May not find the optimal hyperparameters due to the random nature of the search.
- Requires careful tuning of the hyperparameter distributions.

## Comparison of GridSearchCV and RandomizedSearchCV

| Feature                | GridSearchCV        | RandomizedSearchCV   |
|------------------------|---------------------|----------------------|
| Search strategy         | Exhaustive search    | Random search        |
| Computational complexity | High                | Low                  |
| Hyperparameter space     | Discrete            | Continuous or discrete|
| Number of iterations     | Fixed               | Variable             |

### When to use each method

**Use GridSearchCV when:**
- The hyperparameter space is small.
- You want to exhaustively search the hyperparameter space.

**Use RandomizedSearchCV when:**
- The hyperparameter space is large.
- You want to perform a more efficient search.

## Best Practices for Hyperparameter Tuning

- Use a combination of GridSearchCV and RandomizedSearchCV to leverage the strengths of both methods.
- Use cross-validation to evaluate the model's performance on unseen data.
- Use a robust evaluation metric to avoid overfitting.
- Perform hyperparameter tuning on a subset of the data to reduce computational complexity.
- Use techniques such as early stopping and learning rate scheduling to improve the efficiency of the search.

## Hyperparameter Tuning Strategies

### Grid Search Strategies

- **Full Grid Search:** Exhaustively search the entire hyperparameter space.
- **Random Grid Search:** Randomly sample the hyperparameter space and perform a grid search on the sampled points.
- **Grid Search with Bayesian Optimization:** Use Bayesian optimization to guide the grid search and focus on the most promising regions of the hyperparameter space.

### Random Search Strategies

- **Uniform Random Search:** Randomly sample the hyperparameter space using a uniform distribution.
- **Non-Uniform Random Search:** Randomly sample the hyperparameter space using a non-uniform distribution (e.g., normal, lognormal, etc.).
- **Random Search with Bayesian Optimization:** Use Bayesian optimization to guide the random search and focus on the most promising regions of the hyperparameter space.

### Hybrid Strategies

- **Grid Search with Random Search:** Perform a grid search on a subset of the hyperparameter space and then use random search to explore the remaining space.
- **Random Search with Grid Search:** Perform a random search on the entire hyperparameter space and then use grid search to refine the search in the most promising regions.

## Hyperparameter Tuning for Deep Learning Models

### Challenges of Hyperparameter Tuning for Deep Learning Models

- **High-dimensional hyperparameter space:** Deep learning models have a large number of hyperparameters, making it challenging to perform hyperparameter tuning.
- **Computational complexity:** Training deep learning models is computationally expensive, making it challenging to perform hyperparameter tuning using traditional methods.

### Strategies for Hyperparameter Tuning for Deep Learning Models

- **Transfer learning:** Use pre-trained models as a starting point for hyperparameter tuning.
- **Hyperparameter tuning using proxy tasks:** Use proxy tasks (e.g., image classification on a smaller dataset) to perform hyperparameter tuning and then transfer the learned hyperparameters to the target task.
- **Hyperparameter tuning using Bayesian optimization:** Use Bayesian optimization to guide the hyperparameter tuning process and focus on the most promising regions of the hyperparameter space.

## Conclusion

Hyperparameter tuning is a crucial step in machine learning model development. GridSearchCV and RandomizedSearchCV are two popular methods for hyperparameter tuning, each with its own strengths and weaknesses. By understanding the theoretical aspects of these methods and using best practices for hyperparameter tuning, machine learning practitioners can develop more accurate and efficient models.

## Use Case: GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, VotingRegressor
)
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

X_train, X_test, Y_train, Y_test = train_test_split(
    df.drop(columns='Absenteeism time in hours'),
    df['Absenteeism time in hours'],
    test_size=0.2,
    random_state=4
)

r2_scorer = make_scorer(r2_score, greater_is_better=True)
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

models = {
    'ridge': Ridge(),
    'decision_tree': DecisionTreeRegressor(),
    'random_forest': RandomForestRegressor(),
    'extra_trees': ExtraTreesRegressor(),
    'gradient_boosting': GradientBoostingRegressor(),
    'mlp': MLPRegressor(),
    'bagging': BaggingRegressor(),
    'adaboost': AdaBoostRegressor(),
    'svr': SVR(),
    'vote': VotingRegressor(estimators=[
        ('ridge', Ridge()),
        ('decision_tree', DecisionTreeRegressor()),
        ('random_forest', RandomForestRegressor()),
        ('extra_trees', ExtraTreesRegressor()),
        ('gradient_boosting', GradientBoostingRegressor()),
        ('mlp', MLPRegressor()),
        ('bagging', BaggingRegressor()),
        ('adaboost', AdaBoostRegressor()),
        ('svr', SVR())
    ])
}

from sklearn.model_selection import cross_val_score

def cross_val_scores(model_name, model):
    print(f"Running cross-validation for {model_name}...")
    scores_r2 = cross_val_score(model, X_train, Y_train, cv=5, scoring=r2_scorer)
    scores_mse = cross_val_score(model, X_train, Y_train, cv=5, scoring=mse_scorer)
    print(f"Cross-validation R2 scores for {model_name}: {scores_r2}")
    print(f"Cross-validation MSE scores for {model_name}: {scores_mse}")
    print(f"Average cross-validation R2 score for {model_name}: {scores_r2.mean()}")
    print(f"Average cross-validation MSE score for {model_name}: {scores_mse.mean()}")

for model_name, model in models.items():
    cross_val_scores(model_name, model)
    Y_pred = model.predict(X_test)
    r2 = r2_score(Y_test, Y_pred)
    print(f"R2 Score for {model_name} on test set: {r2}")

    best_model_name = max(best_models, key=lambda x: r2_score(Y_test, best_models[x].predict(X_test)))
    print(f"Best model: {best_model_name}")

    Y_pred = best_models[best_model_name].predict(X_test)
    r2 = r2_score(Y_test, Y_pred)
    print(f"R2 Score for best model on test set: {r2}")
```


## Use Case: RandomizedSearchCV

```python

from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

X_train, X_test, Y_train, Y_test = train_test_split(
    df.drop(columns='Absenteeism time in hours'),
    df['Absenteeism time in hours'],
    test_size=0.2,
    random_state=4
)

scorer = make_scorer(r2_score)

param_distributions = {
    'ridge': {
        'alpha': [0.05, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        'solver': ['svd', 'cholesky']
    },
    'decision_tree': {
        'max_depth': [3, 5, 7, 10, 15, 20, 25],
        'min_samples_split': [2, 5, 7, 10],
        'min_samples_leaf': [1, 2
, 4, 5],
        'criterion': ['absolute_error', 'poisson', 'squared_error', 'friedman_mse']
    },
    'random_forest': {
        'n_estimators': [100, 250, 500, 1000, 2000, 5000, 10000],
        'max_depth': [5, 7, 10, 15, 20, 25],
        'min_samples_split': [2, 5, 7, 10],
        'min_samples_leaf': [1, 2, 4, 5],
        'criterion': ['absolute_error', 'poisson', 'squared_error', 'friedman_mse']
    },
    'extra_trees': {
        'n_estimators': [100, 250, 500, 1000, 2000, 5000, 10000],
        'max_depth': [5, 7, 10, 15, 20, 25],
        'min_samples_split': [2, 5, 7, 10],
        'min_samples_leaf': [1, 2, 4, 5],
        'criterion': ['absolute_error', 'poisson', 'squared_error', 'friedman_mse']
    },
    'gradient_boosting': {
        'n_estimators': [100, 250, 500, 1000, 2000, 5000, 10000],
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7, 10, 15, 20, 25],
        'subsample': [0.8, 0.9, 1.0]
    },
    'svr': {
        'kernel': ['poly', 'sigmoid', 'rbf', 'linear'],
        'C': [1e0, 1e2, 1e4, 1e6, 1e8, 1e10],
        'epsilon': [0.1, 0.5, 1.0, 5.0, 10.0]
    },
    'mlp': {
        'hidden_layer_sizes': [(50, 50), (100, 100), (200, 200), (500, 500), (1000, 1000)],
        'activation': ['relu', 'tanh', 'sigmoid'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    },
    'bagging': {
        'n_estimators': [100, 250, 500, 1000, 2000, 5000, 10000],
        'max_samples': [0.2, 0.5, 0.8, 1.0],
        'max_features': [0.2, 0.5, 0.8, 1.0]
    },
    'adaboost': {
        'n_estimators': [100, 250, 500, 1000, 2000, 5000, 10000],
        'learning_rate': [0.001, 0.01, 0.05, 0.1]
    }
}

models = {
    'ridge': Ridge(),
    'decision_tree': DecisionTreeRegressor(),
    'random_forest': RandomForestRegressor(),
    'extra_trees': ExtraTreesRegressor(),
    'gradient_boosting': GradientBoostingRegressor(),
    'mlp': MLPRegressor(),
    'bagging': BaggingRegressor(),
    'adaboost': AdaBoostRegressor(),
    'svr': SVR(),
}

best_models = {}

def random_search_model(model_name, model, param_distribution):
    print(f"Running RandomizedSearchCV for {model_name}...")
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distribution, cv=5, n_iter=10, random_state=42, scoring=scorer)
    random_search.fit(X_train, Y_train)
    print(f"Best parameters for {model_name}: {random_search.best_params_}")
    print(f"Best R2 Score for {model_name}: {random_search.best_score_}")
    return random_search.best_estimator_

for model_name in models:
    best_models[model_name] = random_search_model(model_name, models[model_name], param_distributions[model_name])

for model_name, model in best_models.items():
    Y_pred = model.predict(X_test)
    r2 = r2_score(Y_test, Y_pred)
    print(f"R2 Score for {model_name} on test set: {r2}")

best_model_name = max(best_models, key=lambda x: r2_score(Y_test, best_models[x].predict(X_test)))
print(f"Best model: {best_model_name}")

Y_pred = best_models[best_model_name].predict(X_test)
r2 = r2_score(Y_test, Y_pred)
print(f"R2 Score for best model on test set: {r2}")
```