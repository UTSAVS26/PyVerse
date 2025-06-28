import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, mean_squared_error

def train_model(model, X_train, X_test, y_train, y_test):
    """Train and evaluate a model"""
    model.fit(X_train, y_train)
    
    # Calculate scores
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_score = r2_score(y_train, train_pred)
    test_score = r2_score(y_test, test_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    # Get feature importance if available
    feature_importance = None
    if hasattr(model, 'coef_'):
        feature_importance = np.abs(model.coef_)
    
    return {
        'model': model,
        'train_score': train_score,
        'test_score': test_score,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'feature_importance': feature_importance
    }

def get_feature_importance_df(model, feature_names):
    """Create feature importance dataframe"""
    if hasattr(model, 'coef_'):
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(model.coef_)
        }).sort_values('Importance', ascending=False)
    return None

def save_model(model, filepath):
    """Save trained model to file"""
    joblib.dump(model, filepath)

def load_model(filepath):
    """Load trained model from file"""
    return joblib.load(filepath)

class BitcoinPredictor:
    def __init__(self, model: BaseEstimator, feature_names, scaler=None):
        self.model = model
        self.feature_names = feature_names
        self.scaler = scaler if scaler else MinMaxScaler()
        
    def predict(self, X):
        """Make predictions on new data"""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names]
        elif isinstance(X, dict):
            X = pd.DataFrame([X])[self.feature_names]
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)