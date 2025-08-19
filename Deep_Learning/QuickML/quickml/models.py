"""
Model definitions and training utilities for QuickML.
Contains various ML algorithms and training pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from typing import Dict, List, Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Handles training of multiple machine learning models and model selection.
    """
    
    def __init__(self, task_type: str = 'classification'):
        """
        Initialize the model trainer.
        
        Args:
            task_type: 'classification' or 'regression'
        """
        self.task_type = task_type
        self.models = {}
        self.trained_models = {}
        self.cv_scores = {}
        self.best_model = None
        self.best_score = -np.inf
        self.best_model_name = None
        
    def get_models(self) -> Dict[str, BaseEstimator]:
        """
        Get the dictionary of models for the current task type.
        
        Returns:
            Dictionary of model name to model instance
        """
        if self.task_type == 'classification':
            return {
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'SVM': SVC(random_state=42, probability=True),
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }
        else:  # regression
            return {
                'LinearRegression': LinearRegression(),
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'SVM': SVR(),
                'KNN': KNeighborsRegressor(n_neighbors=5),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
    
    def train_models(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, float]:
        """
        Train all models and perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary of model names to their CV scores
        """
        self.models = self.get_models()
        self.cv_scores = {}
        
        # Choose appropriate cross-validation strategy
        if self.task_type == 'classification':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = 'accuracy'
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = 'r2'
        
        print(f"Training {len(self.models)} models for {self.task_type} task...")
        
        for name, model in self.models.items():
            try:
                # Perform cross-validation
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                mean_score = scores.mean()
                std_score = scores.std()
                
                self.cv_scores[name] = {
                    'mean': mean_score,
                    'std': std_score,
                    'scores': scores
                }
                
                # Train the model on full dataset
                model.fit(X, y)
                self.trained_models[name] = model
                
                print(f"  {name}: {mean_score:.4f} (+/- {std_score * 2:.4f})")
                
                # Update best model
                if mean_score > self.best_score:
                    self.best_score = mean_score
                    self.best_model = model
                    self.best_model_name = name
                    
            except Exception as e:
                print(f"  {name}: Error during training - {str(e)}")
                self.cv_scores[name] = {
                    'mean': -np.inf,
                    'std': 0,
                    'scores': []
                }
        
        return {name: scores['mean'] for name, scores in self.cv_scores.items()}
    
    def get_best_model(self) -> Tuple[str, BaseEstimator, float]:
        """
        Get the best performing model.
        
        Returns:
            Tuple of (model_name, model_instance, score)
        """
        if self.best_model is None:
            raise ValueError("No models have been trained yet")
        
        return self.best_model_name, self.best_model, self.best_score
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Get a summary of all trained models and their performance.
        
        Returns:
            DataFrame with model performance summary
        """
        if not self.cv_scores:
            return pd.DataFrame()
        
        summary_data = []
        for name, scores in self.cv_scores.items():
            summary_data.append({
                'Model': name,
                'Mean Score': f"{scores['mean']:.4f}",
                'Std Score': f"{scores['std']:.4f}",
                'CV Range': f"{scores['mean'] - scores['std']:.4f} - {scores['mean'] + scores['std']:.4f}"
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Mean Score', ascending=False)
        
        return df
    
    def predict(self, X: np.ndarray, model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using a specific model or the best model.
        
        Args:
            X: Feature matrix
            model_name: Name of the model to use (if None, uses best model)
            
        Returns:
            Predictions
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No models have been trained yet")
            model = self.best_model
        else:
            if model_name not in self.trained_models:
                raise ValueError(f"Model '{model_name}' not found in trained models")
            model = self.trained_models[model_name]
        
        return model.predict(X)
    
    def predict_proba(self, X: np.ndarray, model_name: Optional[str] = None) -> np.ndarray:
        """
        Get prediction probabilities (for classification only).
        
        Args:
            X: Feature matrix
            model_name: Name of the model to use (if None, uses best model)
            
        Returns:
            Prediction probabilities
        """
        if self.task_type != 'classification':
            raise ValueError("Probability predictions only available for classification tasks")
        
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No models have been trained yet")
            model = self.best_model
        else:
            if model_name not in self.trained_models:
                raise ValueError(f"Model '{model_name}' not found in trained models")
            model = self.trained_models[model_name]
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            raise ValueError(f"Model '{model_name}' does not support probability predictions")
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Get feature importance from the model (if available).
        
        Args:
            model_name: Name of the model to use (if None, uses best model)
            
        Returns:
            Feature importance array or None if not available
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No models have been trained yet")
            model = self.best_model
        else:
            if model_name not in self.trained_models:
                raise ValueError(f"Model '{model_name}' not found in trained models")
            model = self.trained_models[model_name]
        
        # Check if model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        
        # Check if model has coef_ attribute (linear models)
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_)
        
        return None
