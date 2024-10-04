# Predicting Exam Scores Using Linear Regression

This project demonstrates the application of linear regression to predict exam scores based on the number of study hours.

## Dataset

The dataset contains two columns:
- `Hours`: Number of study hours.
- `Scores`: Exam scores.

## Steps

1. **Import Libraries**:
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    ```

2. **Load Dataset**:
    ```python
    data = pd.read_csv('path/to/dataset.csv')
    ```

3. **Data Visualization**:
    ```python
    plt.scatter(data['Hours'], data['Scores'])
    plt.title('Hours vs Scores')
    plt.xlabel('Hours Studied')
    plt.ylabel('Score')
    plt.show()
    ```

4. **Prepare Data**:
    ```python
    X = data['Hours'].values.reshape(-1, 1)
    y = data['Scores'].values
    ```

5. **Split Data into Training and Test Sets**:
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

6. **Train the Model**:
    ```python
    model = LinearRegression()
    model.fit(X_train, y_train)
    ```

7. **Make Predictions**:
    ```python
    y_pred = model.predict(X_test)
    ```

8. **Evaluate the Model**:
    ```python
    print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
    ```

## Conclusion

This model can be used to predict exam scores based on the number of hours studied. Further improvements can be made by exploring more complex models and additional features.
