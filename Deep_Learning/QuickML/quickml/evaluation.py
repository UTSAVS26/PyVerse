"""
Model evaluation utilities for QuickML.
Handles metrics calculation and model performance assessment.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score
)
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Handles model evaluation and metrics calculation for both classification and regression tasks.
    """
    
    def __init__(self, task_type: str = 'classification'):
        """
        Initialize the model evaluator.
        
        Args:
            task_type: 'classification' or 'regression'
        """
        self.task_type = task_type
        self.metrics = {}
        
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Handle multi-class vs binary classification
        if len(np.unique(y_true)) == 2:
            # Binary classification
            metrics['precision'] = precision_score(y_true, y_pred, average='binary')
            metrics['recall'] = recall_score(y_true, y_pred, average='binary')
            metrics['f1'] = f1_score(y_true, y_pred, average='binary')
            
            if y_proba is not None:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                except:
                    metrics['roc_auc'] = np.nan
        else:
            # Multi-class classification
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
            
            if y_proba is not None:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                except:
                    metrics['roc_auc'] = np.nan
        
        return metrics
    
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
        
        return metrics
    
    def evaluate_model(self, model, X: np.ndarray, y: np.ndarray, 
                      test_size: float = 0.2, random_state: int = 42) -> Dict[str, Union[float, str]]:
        """
        Evaluate a model using train-test split.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary of evaluation results
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if self.task_type == 'classification' else None
        )
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if self.task_type == 'classification':
            metrics = self.calculate_classification_metrics(y_test, y_pred)
            
            # Try to get probabilities for ROC AUC
            try:
                y_proba = model.predict_proba(X_test)
                metrics.update(self.calculate_classification_metrics(y_test, y_pred, y_proba))
            except:
                pass
                
        else:  # regression
            metrics = self.calculate_regression_metrics(y_test, y_pred)
        
        # Add additional information
        metrics['test_size'] = len(X_test)
        metrics['train_size'] = len(X_train)
        
        return metrics
    
    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Get confusion matrix for classification.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        if self.task_type != 'classification':
            raise ValueError("Confusion matrix only available for classification tasks")
        
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Get detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report string
        """
        if self.task_type != 'classification':
            raise ValueError("Classification report only available for classification tasks")
        
        return classification_report(y_true, y_pred)
    
    def compare_models(self, models: Dict[str, object], X: np.ndarray, y: np.ndarray,
                      test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
        """
        Compare multiple models and return a comparison DataFrame.
        
        Args:
            models: Dictionary of model name to model instance
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame with model comparison results
        """
        results = []
        
        for name, model in models.items():
            try:
                metrics = self.evaluate_model(model, X, y, test_size, random_state)
                metrics['model'] = name
                results.append(metrics)
            except Exception as e:
                print(f"Error evaluating model {name}: {str(e)}")
                continue
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Reorder columns to put model name first
        cols = ['model'] + [col for col in df.columns if col != 'model']
        df = df[cols]
        
        return df
    
    def get_best_model_from_comparison(self, comparison_df: pd.DataFrame) -> Tuple[str, float]:
        """
        Get the best model from a comparison DataFrame.
        
        Args:
            comparison_df: DataFrame with model comparison results
            
        Returns:
            Tuple of (best_model_name, best_score)
        """
        if comparison_df.empty:
            raise ValueError("Comparison DataFrame is empty")
        
        # Determine the primary metric based on task type
        if self.task_type == 'classification':
            primary_metric = 'accuracy'
        else:  # regression
            primary_metric = 'r2'
        
        if primary_metric not in comparison_df.columns:
            # Fallback to first available metric
            available_metrics = [col for col in comparison_df.columns 
                               if col not in ['model', 'test_size', 'train_size']]
            if not available_metrics:
                raise ValueError("No metrics available in comparison DataFrame")
            primary_metric = available_metrics[0]
        
        # Find the best model
        best_idx = comparison_df[primary_metric].idxmax()
        best_model = comparison_df.loc[best_idx, 'model']
        best_score = comparison_df.loc[best_idx, primary_metric]
        
        return best_model, best_score
    
    def format_metrics_for_display(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Format metrics for display purposes.
        
        Args:
            metrics: Dictionary of raw metrics
            
        Returns:
            Dictionary of formatted metrics
        """
        formatted = {}
        
        for key, value in metrics.items():
            if key in ['test_size', 'train_size']:
                formatted[key] = str(int(value))
            elif isinstance(value, float):
                if np.isnan(value):
                    formatted[key] = 'N/A'
                else:
                    formatted[key] = f"{value:.4f}"
            else:
                formatted[key] = str(value)
        
        return formatted
