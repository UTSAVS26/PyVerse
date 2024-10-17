import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('C://Users//Deepak//Desktop//dataset.csv')  # Update with your actual file path

# Data Preprocessing
# Drop rows with missing price
df_clean = df.dropna(subset=['Price'])

# Clean the 'Rating' column
df_clean['Rating'] = pd.to_numeric(df_clean['Rating'].replace('No Rating', np.nan), errors='coerce')
df_clean['Rating'].fillna(df_clean['Rating'].mean(), inplace=True)  # Replace NaN with the mean rating

# Convert 'Price' to numeric after removing commas
df_clean['Price'] = pd.to_numeric(df_clean['Price'].str.replace(',', ''), errors='coerce')

# Convert 'Reviews' to numeric
df_clean['Reviews'] = pd.to_numeric(df_clean['Reviews'], errors='coerce')

# Drop any remaining NaN values
df_clean = df_clean.dropna()

# Features: We'll use 'Rating' and 'Reviews' to predict 'Price'
X = df_clean[['Rating', 'Reviews']]
y = df_clean['Price']

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build RandomForest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

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
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.grid(True)
plt.show()
