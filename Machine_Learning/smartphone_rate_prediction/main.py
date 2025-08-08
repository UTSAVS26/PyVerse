import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Load the dataset
df = pd.read_csv('C://Users//Deepak//Desktop//dataset.csv')  # Update with your actual file path


# Data Preprocessing

# Convert columns to numeric types and handle 'No Rating' as NaN
df['Price'] = pd.to_numeric(df['Price'].str.replace(',', '',regex=True), errors='coerce')
df['Rating'] = pd.to_numeric(df['Rating'].replace('No Rating', np.nan), errors='coerce')
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')

# Use median imputation to fill missing values (safer than dropping them)
imputer = SimpleImputer(strategy='median')
df_clean = df.copy()
df_clean[['Price', 'Rating', 'Reviews']] = imputer.fit_transform(df_clean[['Price', 'Rating', 'Reviews']])


# Apply log1p to reduce skewness in price (log1p handles zero values safely)
df_clean['Price'] = np.log1p(df_clean['Price'])


# Drop any remaining NaN values
df_clean = df_clean.dropna()

# Features: We'll use 'Rating' and 'Reviews' to predict 'Price'
X = df_clean[['Rating', 'Reviews']]
y = df_clean['Price']

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Set up parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5)
}

# Run randomized search with 5-fold cross-validation
base_model = RandomForestRegressor(random_state=42)
search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=20,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)
model = search.best_estimator_

# Predict using the best model
y_pred = model.predict(X_test)

# Convert log predictions back to original price scale for evaluation
y_test_exp = np.expm1(y_test)
y_pred_exp = np.expm1(y_pred)

mse = mean_squared_error(y_test_exp, y_pred_exp)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_exp, y_pred_exp)


# Print the results
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")

# Visualization 1: Distribution of Prices
plt.figure(figsize=(10, 6))
sns.histplot(df_clean['Price'], bins=30, kde=True)
plt.title('Distribution of Smartphone Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Visualization 2: Feature Importance
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), features[indices], rotation=45)
plt.xlim([-1, X.shape[1]])
plt.ylabel('Importance')
plt.grid(True)
plt.show()

# Visualization 3: Actual vs Predicted Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test_exp, y_pred_exp, color='blue', alpha=0.7)
plt.plot([y_test_exp.min(), y_test_exp.max()], [y_test_exp.min(), y_test_exp.max()], 'k--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.grid(True)
plt.show()  