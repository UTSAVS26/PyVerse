import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(filepath='data/bitcoin_dataset.csv'):
    """Load dataset and perform initial inspection"""
    data = pd.read_csv(filepath)
    print("Data loaded successfully. Shape:", data.shape)
    return data

def preprocess_data(data):
    """Handle missing values and clean data"""
    print("\nMissing values before treatment:")
    print(data.isnull().sum())
    
    # Convert Date to datetime if exists
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
        data = data.drop('Date', axis=1)
    
    # Forward fill missing values
    data_clean = data.ffill()
    
    print("\nMissing values after forward fill:")
    print(data_clean.isnull().sum())
    
    return data_clean

def prepare_training_data(data, target_col='btc_market_price', test_size=0.2, random_state=20):
    """Prepare data for machine learning"""
    # Verify target column exists
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    X = data.drop([target_col], axis=1)
    y = data[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler