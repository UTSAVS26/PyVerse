"""
Core QuickML AutoML engine.
Main orchestrator that coordinates preprocessing, model training, evaluation, and visualization.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import os
from datetime import datetime

from .preprocessing import DataPreprocessor
from .models import ModelTrainer
from .evaluation import ModelEvaluator
from .visualization import Visualizer

warnings.filterwarnings('ignore')


class QuickML:
    """
    Main QuickML AutoML engine that orchestrates the entire pipeline.
    """
    
    def __init__(self, target_column: Optional[str] = None, task_type: Optional[str] = None):
        """
        Initialize the QuickML engine.
        
        Args:
            target_column: Name of the target column (if known)
            task_type: 'classification' or 'regression' (if known)
        """
        self.target_column = target_column
        self.task_type = task_type
        self.preprocessor = None
        self.model_trainer = None
        self.evaluator = None
        self.visualizer = None
        self.best_model = None
        self.best_model_name = None
        self.best_score = None
        self.feature_names = []
        self.results = {}
        
    def fit(self, df: pd.DataFrame, target_column: Optional[str] = None, 
            cv_folds: int = 5) -> Dict[str, Any]:
        """
        Fit the QuickML pipeline on the provided data.
        
        Args:
            df: Input DataFrame
            target_column: Target column name (if not already set)
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing results and metadata
        """
        print("🚀 Starting QuickML AutoML Pipeline...")
        print(f"📊 Dataset shape: {df.shape}")
        
        # Step 1: Data Preprocessing
        print("\n🔧 Step 1: Data Preprocessing")
        self.preprocessor = DataPreprocessor(target_column or self.target_column)
        
        # Detect target column and task type
        if target_column:
            self.target_column = target_column
        elif not self.target_column:
            self.target_column = self.preprocessor.detect_target_column(df)
        
        print(f"🎯 Target column: {self.target_column}")
        
        # Transform data
        X_transformed, _, detected_task_type = self.preprocessor.fit_transform(df, self.target_column)
        
        if not self.task_type:
            self.task_type = detected_task_type
        
        print(f"📈 Task type: {self.task_type}")
        print(f"🔢 Transformed features shape: {X_transformed.shape}")
        
        # Get target values
        y = df[self.target_column].values
        
        # Step 2: Model Training
        print(f"\n🤖 Step 2: Model Training ({self.task_type})")
        self.model_trainer = ModelTrainer(task_type=self.task_type)
        
        # Train all models
        cv_scores = self.model_trainer.train_models(X_transformed, y, cv_folds)
        
        # Get best model
        self.best_model_name, self.best_model, self.best_score = self.model_trainer.get_best_model()
        
        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"📊 Best Score: {self.best_score:.4f}")
        
        # Step 3: Model Evaluation
        print(f"\n📊 Step 3: Model Evaluation")
        self.evaluator = ModelEvaluator(task_type=self.task_type)
        
        # Evaluate best model
        evaluation_metrics = self.evaluator.evaluate_model(
            self.best_model, X_transformed, y
        )
        
        # Step 4: Visualization Setup
        self.visualizer = Visualizer(task_type=self.task_type)
        
        # Get feature names
        try:
            self.feature_names = self.preprocessor.get_feature_names()
        except:
            self.feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
        
        # Compile results
        self.results = {
            'target_column': self.target_column,
            'task_type': self.task_type,
            'best_model_name': self.best_model_name,
            'best_score': self.best_score,
            'cv_scores': cv_scores,
            'evaluation_metrics': evaluation_metrics,
            'model_summary': self.model_trainer.get_model_summary(),
            'feature_names': self.feature_names,
            'data_shape': df.shape,
            'transformed_shape': X_transformed.shape,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n✅ QuickML Pipeline Complete!")
        print(f"📈 Best Model: {self.best_model_name}")
        print(f"🎯 Score: {self.best_score:.4f}")
        
        return self.results
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Predictions
        """
        if self.best_model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Transform the data
        X_transformed = self.preprocessor.transform(df)
        
        # Make predictions
        predictions = self.best_model.predict(X_transformed)
        
        return predictions
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities (for classification only).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Prediction probabilities
        """
        if self.task_type != 'classification':
            raise ValueError("Probability predictions only available for classification tasks")
        
        if self.best_model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Transform the data
        X_transformed = self.preprocessor.transform(df)
        
        # Get probabilities
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(X_transformed)
        else:
            raise ValueError("Best model does not support probability predictions")
        
        return probabilities
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance from the best model.
        
        Returns:
            Feature importance array or None if not available
        """
        if self.best_model is None:
            raise ValueError("Model must be fitted before getting feature importance")
        
        return self.model_trainer.get_feature_importance()
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Get a summary of all trained models.
        
        Returns:
            DataFrame with model performance summary
        """
        if self.model_trainer is None:
            raise ValueError("Models must be trained before getting summary")
        
        return self.model_trainer.get_model_summary()
    
    def save_model(self, filepath: str = 'best_model.pkl') -> str:
        """
        Save the best model and preprocessor to a file.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            Path where model was saved
        """
        if self.best_model is None:
            raise ValueError("Model must be fitted before saving")
        
        # Create a model package with both model and preprocessor
        model_package = {
            'model': self.best_model,
            'preprocessor': self.preprocessor,
            'model_name': self.best_model_name,
            'task_type': self.task_type,
            'target_column': self.target_column,
            'feature_names': self.feature_names,
            'results': self.results
        }
        
        # Save the model package
        joblib.dump(model_package, filepath)
        
        print(f"💾 Model saved to: {filepath}")
        return filepath
    
    def load_model(self, filepath: str) -> None:
        """
        Load a saved model and preprocessor.
        
        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load the model package
        model_package = joblib.load(filepath)
        
        # Restore all components
        self.best_model = model_package['model']
        self.preprocessor = model_package['preprocessor']
        self.best_model_name = model_package['model_name']
        self.task_type = model_package['task_type']
        self.target_column = model_package['target_column']
        self.feature_names = model_package['feature_names']
        self.results = model_package['results']
        
        # Reinitialize other components
        self.model_trainer = ModelTrainer(task_type=self.task_type)
        self.evaluator = ModelEvaluator(task_type=self.task_type)
        self.visualizer = Visualizer(task_type=self.task_type)
        
        print(f"📂 Model loaded from: {filepath}")
    
    def create_visualizations(self, df: pd.DataFrame, save_plots: bool = False, 
                            output_dir: str = 'plots') -> Dict[str, Any]:
        """
        Create comprehensive visualizations for the model and data.
        
        Args:
            df: Original input DataFrame
            save_plots: Whether to save plots to files
            output_dir: Directory to save plots
            
        Returns:
            Dictionary containing all plots
        """
        if self.visualizer is None:
            raise ValueError("Model must be fitted before creating visualizations")
        
        plots = {}
        
        # 1. Model comparison plot
        if hasattr(self, 'model_trainer') and self.model_trainer:
            model_summary = self.model_trainer.get_model_summary()
            if not model_summary.empty:
                plots['model_comparison'] = self.visualizer.plot_model_comparison(
                    model_summary, metric='Mean Score'
                )
        
        # 2. Feature importance plot
        feature_importance = self.get_feature_importance()
        if feature_importance is not None:
            plots['feature_importance'] = self.visualizer.plot_feature_importance(
                self.feature_names, feature_importance
            )
        
        # 3. Data distribution plots
        plots['data_distribution'] = self.visualizer.plot_data_distribution(
            df, self.target_column
        )
        
        # 4. Model-specific plots
        if self.task_type == 'classification':
            # Confusion matrix
            y_pred = self.predict(df)
            plots['confusion_matrix'] = self.visualizer.plot_confusion_matrix(
                df[self.target_column].values, y_pred
            )
            
            # ROC curve (if probabilities available)
            try:
                y_proba = self.predict_proba(df)
                plots['roc_curve'] = self.visualizer.plot_roc_curve(
                    df[self.target_column].values, y_proba
                )
            except:
                pass
        else:
            # Prediction vs actual plot
            y_pred = self.predict(df)
            plots['prediction_vs_actual'] = self.visualizer.plot_prediction_vs_actual(
                df[self.target_column].values, y_pred
            )
        
        # Save plots if requested
        if save_plots:
            self.visualizer.save_all_plots(plots, output_dir)
        
        return plots
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the QuickML results.
        
        Returns:
            Dictionary with results summary
        """
        if not self.results:
            raise ValueError("Model must be fitted before getting results summary")
        
        summary = {
            'quickml_version': '1.0.0',
            'timestamp': self.results.get('timestamp', ''),
            'data_info': {
                'shape': self.results.get('data_shape', ''),
                'target_column': self.results.get('target_column', ''),
                'task_type': self.results.get('task_type', ''),
                'transformed_features': self.results.get('transformed_shape', '')
            },
            'best_model': {
                'name': self.results.get('best_model_name', ''),
                'score': self.results.get('best_score', ''),
                'evaluation_metrics': self.results.get('evaluation_metrics', {})
            },
            'all_models': self.results.get('cv_scores', {}),
            'feature_importance_available': self.get_feature_importance() is not None
        }
        
        return summary
    
    def __repr__(self) -> str:
        """String representation of the QuickML object."""
        if self.best_model is None:
            return "QuickML(not fitted)"
        
        return f"QuickML(best_model={self.best_model_name}, score={self.best_score:.4f}, task_type={self.task_type})"
