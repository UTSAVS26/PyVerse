# Smartphone Price Prediction Using Random Forest Regressor

## 🎯 Goal
The goal of this project is to build a predictive model that estimates smartphone prices based on their ratings and the number of reviews. The project uses a **Random Forest Regressor** model to predict prices from these features, aiming to deliver accurate pricing insights.

## 🧵 Dataset
The dataset contains various attributes of smartphones, including:
- **Price** (Target variable)
- **Rating** (Independent variable)
- **Reviews** (Independent variable)

The dataset is loaded from a CSV file, which needs to be provided in the file path specified during execution.

## 🧾 Description
This project involves the following steps:
1. **Data Preprocessing**:
   - Handling missing values in the price column.
   - Cleaning the rating column by replacing non-numeric values and imputing missing values with the mean.
   - Converting the price and reviews columns into numeric format.
   - Dropping any remaining rows with missing data.
   
2. **Model Building**:
   - Features: The model uses smartphone **Rating** and **Reviews** to predict **Price**.
   - A **Random Forest Regressor** model is built to predict the price using 80% of the data for training and 20% for testing.

3. **Model Evaluation**:
   - **RMSE (Root Mean Squared Error)** and **R² Score** are calculated to evaluate the model's performance.

4. **Data Visualization**:
   - Visualization of the price distribution.
   - Feature importance plot to show how each feature contributes to the model.
   - Actual vs. predicted prices scatter plot.

## 📊 Features and Workflow

1. **Data Preprocessing**:
   - Remove rows with missing prices.
   - Convert ratings and reviews into numeric form.
   - Remove commas from the price column and convert it to a numeric format.
   - Drop any remaining rows with missing values.

2. **Random Forest Regressor**:
   - Built using the `sklearn` library.
   - `n_estimators=100` for building 100 trees in the forest.
   - Data split into training (80%) and testing (20%) sets.

3. **Model Performance**:
   - **RMSE**: Measures the average difference between the actual and predicted prices.
   - **R² Score**: Indicates how well the independent variables explain the variance in the price.

4. **Visualizations**:
   - **Price Distribution**: Histogram and KDE plot showing the distribution of smartphone prices.
   - **Feature Importance**: Bar plot indicating which features contribute the most to price prediction.
   - **Actual vs. Predicted Prices**: Scatter plot comparing the predicted and actual prices for the test set.

## 🧮 Performance Metrics
- **RMSE (Root Mean Squared Error)**: A measure of the model’s error in predicting the price.
- **R² Score**: A metric that shows the proportion of variance in the dependent variable explained by the independent variables.

## 📚 Libraries Needed
- `pandas`: For data manipulation.
- `numpy`: For numerical operations.
- `matplotlib` & `seaborn`: For data visualization.
- `sklearn`: For model building and evaluation.

## 🔧 How to Run
1. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Load the dataset and run the script:
   ```python
   df = pd.read_csv('path_to_your_dataset.csv')  # Update with the actual path
   ```
3. The script will preprocess the data, train the Random Forest model, and provide predictions along with visualizations.

## 📈 Results
- **RMSE** and **R² Score** will be printed in the console after running the script.
- Visualizations (Price distribution, feature importance, actual vs predicted prices) will be shown to help analyze the results.

## 📢 Conclusion
This project demonstrates how **Random Forest Regressor** can be used to predict smartphone prices based on features like ratings and reviews. The visualizations provide a deeper understanding of the model's performance and feature contributions.

## ✒️ Author
**Benak Deepak**

- LinkedIn: [www.linkedin.com/in/benak-deepak-210918254](www.linkedin.com/in/benak-deepak-210918254)
```
