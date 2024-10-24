import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('C:\\Users\\Deepak\\Desktop\\merge\\merged_output.csv') # Update with your dataset path

# Feature engineering and data preparation
# Assuming the dataset has the following columns: 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target'
# You can change this according to your actual dataset
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Selecting features and target
X = df[['Open', 'High', 'Low', 'Volume']]  # Example features, adjust based on your dataset
y_classification = (df['Close'].shift(-1) > df['Close']).astype(int)  # Binary classification (increase/decrease)
y_regression = df['Close']  # Continuous target for regression (stock prices)

# Handling missing values using SimpleImputer (mean strategy)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Splitting dataset into training and testing sets
X_train, X_test, y_train_class, y_test_class = train_test_split(X_imputed, y_classification, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_imputed, y_regression, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

# Classification Models
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "k-NN": KNeighborsClassifier(),
    "Neural Networks": MLPClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

print("Classification Results:")
for name, model in classifiers.items():
    model.fit(X_train_scaled, y_train_class)
    y_pred_class = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test_class, y_pred_class)
    precision = precision_score(y_test_class, y_pred_class)
    f1 = f1_score(y_test_class, y_pred_class)
    print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}")
