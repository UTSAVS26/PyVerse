"""
Comprehensive test suite for QuickML AutoML Engine.
Tests all modules and functionality.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

from quickml import QuickML
from quickml.preprocessing import DataPreprocessor
from quickml.models import ModelTrainer
from quickml.evaluation import ModelEvaluator
from quickml.visualization import Visualizer


class TestDataPreprocessor:
    """Test cases for DataPreprocessor."""
    
    def setup_method(self):
        """Set up test data."""
        # Create test classification data
        np.random.seed(42)
        n_samples = 100
        
        self.classification_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'categorical': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.choice([0, 1], n_samples)
        })
        
        # Create test regression data
        self.regression_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'categorical': np.random.choice(['X', 'Y', 'Z'], n_samples),
            'target': np.random.normal(0, 1, n_samples)
        })
        
        # Create data with missing values
        self.missing_data = self.classification_data.copy()
        self.missing_data.loc[0:10, 'feature1'] = np.nan
        self.missing_data.loc[5:15, 'categorical'] = np.nan
    
    def test_init(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor.target_column is None
        assert preprocessor.is_fitted is False
        
        preprocessor = DataPreprocessor(target_column='target')
        assert preprocessor.target_column == 'target'
    
    def test_detect_column_types(self):
        """Test column type detection."""
        preprocessor = DataPreprocessor(target_column='target')
        numerical, categorical = preprocessor.detect_column_types(self.classification_data)
        
        assert 'feature1' in numerical
        assert 'feature2' in numerical
        assert 'categorical' in categorical
        assert 'target' not in numerical
        assert 'target' not in categorical
    
    def test_detect_target_column(self):
        """Test target column detection."""
        preprocessor = DataPreprocessor()
        target_col = preprocessor.detect_target_column(self.classification_data)
        assert target_col == 'target'
        
        preprocessor = DataPreprocessor(target_column='feature1')
        target_col = preprocessor.detect_target_column(self.classification_data)
        assert target_col == 'feature1'
    
    def test_detect_task_type(self):
        """Test task type detection."""
        preprocessor = DataPreprocessor()
        
        # Classification task
        task_type = preprocessor.detect_task_type(self.classification_data, 'target')
        assert task_type == 'classification'
        
        # Regression task
        task_type = preprocessor.detect_task_type(self.regression_data, 'target')
        assert task_type == 'regression'
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        preprocessor = DataPreprocessor()
        cleaned_data = preprocessor.handle_missing_values(self.missing_data)
        
        assert cleaned_data.isnull().sum().sum() == 0
        assert cleaned_data.shape == self.missing_data.shape
    
    def test_encode_categorical_features(self):
        """Test categorical feature encoding."""
        preprocessor = DataPreprocessor()
        categorical_cols = ['categorical']
        encoded_data = preprocessor.encode_categorical_features(
            self.classification_data, categorical_cols
        )
        
        assert encoded_data['categorical'].dtype in [np.int64, np.int32]
        assert len(preprocessor.label_encoders) == 1
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        preprocessor = DataPreprocessor()
        X_transformed, target_col, task_type = preprocessor.fit_transform(
            self.classification_data
        )
        
        assert isinstance(X_transformed, np.ndarray)
        assert target_col == 'target'
        assert task_type == 'classification'
        assert preprocessor.is_fitted is True
        assert X_transformed.shape[0] == len(self.classification_data)
    
    def test_transform(self):
        """Test transform method."""
        preprocessor = DataPreprocessor()
        preprocessor.fit_transform(self.classification_data)
        
        # Test on new data
        new_data = self.classification_data.head(10)
        X_transformed = preprocessor.transform(new_data)
        
        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape[0] == 10


class TestModelTrainer:
    """Test cases for ModelTrainer."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        self.X = np.random.normal(0, 1, (n_samples, n_features))
        self.y_classification = np.random.choice([0, 1], n_samples)
        self.y_regression = np.random.normal(0, 1, n_samples)
    
    def test_init(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(task_type='classification')
        assert trainer.task_type == 'classification'
        assert trainer.best_model is None
        
        trainer = ModelTrainer(task_type='regression')
        assert trainer.task_type == 'regression'
    
    def test_get_models_classification(self):
        """Test model dictionary for classification."""
        trainer = ModelTrainer(task_type='classification')
        models = trainer.get_models()
        
        expected_models = [
            'LogisticRegression', 'RandomForest', 'SVM', 
            'KNN', 'GradientBoosting'
        ]
        
        for model_name in expected_models:
            assert model_name in models
    
    def test_get_models_regression(self):
        """Test model dictionary for regression."""
        trainer = ModelTrainer(task_type='regression')
        models = trainer.get_models()
        
        expected_models = [
            'LinearRegression', 'RandomForest', 'SVM', 
            'KNN', 'GradientBoosting'
        ]
        
        for model_name in expected_models:
            assert model_name in models
    
    def test_train_models_classification(self):
        """Test model training for classification."""
        trainer = ModelTrainer(task_type='classification')
        cv_scores = trainer.train_models(self.X, self.y_classification, cv_folds=3)
        
        assert isinstance(cv_scores, dict)
        assert len(cv_scores) > 0
        assert trainer.best_model is not None
        assert trainer.best_model_name is not None
        assert trainer.best_score > 0
    
    def test_train_models_regression(self):
        """Test model training for regression."""
        trainer = ModelTrainer(task_type='regression')
        cv_scores = trainer.train_models(self.X, self.y_regression, cv_folds=3)
        
        assert isinstance(cv_scores, dict)
        assert len(cv_scores) > 0
        assert trainer.best_model is not None
        assert trainer.best_model_name is not None
    
    def test_get_best_model(self):
        """Test getting best model."""
        trainer = ModelTrainer(task_type='classification')
        trainer.train_models(self.X, self.y_classification, cv_folds=3)
        
        model_name, model, score = trainer.get_best_model()
        assert isinstance(model_name, str)
        assert model is not None
        assert isinstance(score, float)
    
    def test_get_model_summary(self):
        """Test model summary generation."""
        trainer = ModelTrainer(task_type='classification')
        trainer.train_models(self.X, self.y_classification, cv_folds=3)
        
        summary = trainer.get_model_summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) > 0
        assert 'Model' in summary.columns
        assert 'Mean Score' in summary.columns
    
    def test_predict(self):
        """Test prediction functionality."""
        trainer = ModelTrainer(task_type='classification')
        trainer.train_models(self.X, self.y_classification, cv_folds=3)
        
        predictions = trainer.predict(self.X)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(self.X)
    
    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        trainer = ModelTrainer(task_type='classification')
        trainer.train_models(self.X, self.y_classification, cv_folds=3)
        
        importance = trainer.get_feature_importance()
        # Feature importance might not be available for all models
        if importance is not None:
            assert isinstance(importance, np.ndarray)
            assert len(importance) == self.X.shape[1]


class TestModelEvaluator:
    """Test cases for ModelEvaluator."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        self.X = np.random.normal(0, 1, (n_samples, n_features))
        self.y_classification = np.random.choice([0, 1], n_samples)
        self.y_regression = np.random.normal(0, 1, n_samples)
    
    def test_init(self):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator(task_type='classification')
        assert evaluator.task_type == 'classification'
        
        evaluator = ModelEvaluator(task_type='regression')
        assert evaluator.task_type == 'regression'
    
    def test_calculate_classification_metrics(self):
        """Test classification metrics calculation."""
        evaluator = ModelEvaluator(task_type='classification')
        
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        metrics = evaluator.calculate_classification_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert all(isinstance(v, float) for v in metrics.values())
    
    def test_calculate_regression_metrics(self):
        """Test regression metrics calculation."""
        evaluator = ModelEvaluator(task_type='regression')
        
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = evaluator.calculate_regression_metrics(y_true, y_pred)
        
        assert 'r2' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert all(isinstance(v, float) for v in metrics.values())
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        from sklearn.linear_model import LogisticRegression
        
        evaluator = ModelEvaluator(task_type='classification')
        model = LogisticRegression(random_state=42)
        
        metrics = evaluator.evaluate_model(model, self.X, self.y_classification)
        
        assert isinstance(metrics, dict)
        assert 'test_size' in metrics
        assert 'train_size' in metrics
    
    def test_get_confusion_matrix(self):
        """Test confusion matrix generation."""
        evaluator = ModelEvaluator(task_type='classification')
        
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        cm = evaluator.get_confusion_matrix(y_true, y_pred)
        assert isinstance(cm, np.ndarray)
        assert cm.shape == (2, 2)
    
    def test_compare_models(self):
        """Test model comparison."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        
        evaluator = ModelEvaluator(task_type='classification')
        
        models = {
            'LogisticRegression': LogisticRegression(random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42)
        }
        
        comparison = evaluator.compare_models(models, self.X, self.y_classification)
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert 'model' in comparison.columns


class TestVisualizer:
    """Test cases for Visualizer."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100
        
        self.comparison_df = pd.DataFrame({
            'Model': ['Model1', 'Model2', 'Model3'],
            'Mean Score': [0.85, 0.90, 0.88],
            'Std Score': [0.02, 0.03, 0.01]
        })
        
        self.feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
        self.importance_scores = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        
        self.y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
    
    def test_init(self):
        """Test Visualizer initialization."""
        visualizer = Visualizer(task_type='classification')
        assert visualizer.task_type == 'classification'
        
        visualizer = Visualizer(task_type='regression')
        assert visualizer.task_type == 'regression'
    
    def test_plot_model_comparison(self):
        """Test model comparison plot."""
        visualizer = Visualizer(task_type='classification')
        fig = visualizer.plot_model_comparison(self.comparison_df)
        
        assert fig is not None
        # Add more specific assertions about the figure if needed
    
    def test_plot_feature_importance(self):
        """Test feature importance plot."""
        visualizer = Visualizer(task_type='classification')
        fig = visualizer.plot_feature_importance(
            self.feature_names, self.importance_scores
        )
        
        assert fig is not None
    
    def test_plot_confusion_matrix(self):
        """Test confusion matrix plot."""
        visualizer = Visualizer(task_type='classification')
        fig = visualizer.plot_confusion_matrix(self.y_true, self.y_pred)
        
        assert fig is not None
    
    def test_plot_confusion_matrix_regression_error(self):
        """Test that confusion matrix raises error for regression."""
        visualizer = Visualizer(task_type='regression')
        
        with pytest.raises(ValueError, match="Confusion matrix only available for classification tasks"):
            visualizer.plot_confusion_matrix(self.y_true, self.y_pred)
    
    def test_create_interactive_model_comparison(self):
        """Test interactive model comparison."""
        visualizer = Visualizer(task_type='classification')
        fig = visualizer.create_interactive_model_comparison(self.comparison_df)
        
        assert fig is not None
    
    def test_create_interactive_feature_importance(self):
        """Test interactive feature importance."""
        visualizer = Visualizer(task_type='classification')
        fig = visualizer.create_interactive_feature_importance(
            self.feature_names, self.importance_scores
        )
        
        assert fig is not None


class TestQuickML:
    """Test cases for the main QuickML class."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100
        
        self.classification_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'categorical': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.choice([0, 1], n_samples)
        })
        
        self.regression_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'categorical': np.random.choice(['X', 'Y', 'Z'], n_samples),
            'target': np.random.normal(0, 1, n_samples)
        })
    
    def test_init(self):
        """Test QuickML initialization."""
        quickml = QuickML()
        assert quickml.target_column is None
        assert quickml.task_type is None
        assert quickml.best_model is None
        
        quickml = QuickML(target_column='target', task_type='classification')
        assert quickml.target_column == 'target'
        assert quickml.task_type == 'classification'
    
    def test_fit_classification(self):
        """Test QuickML fit for classification."""
        quickml = QuickML()
        results = quickml.fit(self.classification_data, cv_folds=3)
        
        assert isinstance(results, dict)
        assert 'best_model_name' in results
        assert 'best_score' in results
        assert 'task_type' in results
        assert results['task_type'] == 'classification'
        assert quickml.best_model is not None
        assert quickml.best_model_name is not None
        assert quickml.best_score > 0
    
    def test_fit_regression(self):
        """Test QuickML fit for regression."""
        quickml = QuickML()
        results = quickml.fit(self.regression_data, cv_folds=3)
        
        assert isinstance(results, dict)
        assert 'best_model_name' in results
        assert 'best_score' in results
        assert 'task_type' in results
        assert results['task_type'] == 'regression'
        assert quickml.best_model is not None
        assert quickml.best_model_name is not None
    
    def test_predict(self):
        """Test prediction functionality."""
        quickml = QuickML()
        quickml.fit(self.classification_data, cv_folds=3)
        
        predictions = quickml.predict(self.classification_data)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(self.classification_data)
    
    def test_predict_proba(self):
        """Test probability prediction."""
        quickml = QuickML()
        quickml.fit(self.classification_data, cv_folds=3)
        
        probabilities = quickml.predict_proba(self.classification_data)
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape[0] == len(self.classification_data)
        assert probabilities.shape[1] == 2  # Binary classification
    
    def test_predict_proba_regression_error(self):
        """Test that predict_proba raises error for regression."""
        quickml = QuickML()
        quickml.fit(self.regression_data, cv_folds=3)
        
        with pytest.raises(ValueError, match="Probability predictions only available for classification tasks"):
            quickml.predict_proba(self.regression_data)
    
    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        quickml = QuickML()
        quickml.fit(self.classification_data, cv_folds=3)
        
        importance = quickml.get_feature_importance()
        if importance is not None:
            assert isinstance(importance, np.ndarray)
            assert len(importance) > 0
    
    def test_get_model_summary(self):
        """Test model summary generation."""
        quickml = QuickML()
        quickml.fit(self.classification_data, cv_folds=3)
        
        summary = quickml.get_model_summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) > 0
    
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        quickml = QuickML()
        quickml.fit(self.classification_data, cv_folds=3)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            quickml.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Load model
            new_quickml = QuickML()
            new_quickml.load_model(model_path)
            
            assert new_quickml.best_model is not None
            assert new_quickml.best_model_name == quickml.best_model_name
            assert new_quickml.task_type == quickml.task_type
            assert new_quickml.target_column == quickml.target_column
            
            # Test prediction consistency
            original_pred = quickml.predict(self.classification_data)
            loaded_pred = new_quickml.predict(self.classification_data)
            np.testing.assert_array_equal(original_pred, loaded_pred)
            
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_create_visualizations(self):
        """Test visualization creation."""
        quickml = QuickML()
        quickml.fit(self.classification_data, cv_folds=3)
        
        plots = quickml.create_visualizations(self.classification_data)
        assert isinstance(plots, dict)
        assert len(plots) > 0
    
    def test_get_results_summary(self):
        """Test results summary generation."""
        quickml = QuickML()
        quickml.fit(self.classification_data, cv_folds=3)
        
        summary = quickml.get_results_summary()
        assert isinstance(summary, dict)
        assert 'quickml_version' in summary
        assert 'data_info' in summary
        assert 'best_model' in summary
    
    def test_repr(self):
        """Test string representation."""
        quickml = QuickML()
        assert str(quickml) == "QuickML(not fitted)"
        
        quickml.fit(self.classification_data, cv_folds=3)
        repr_str = str(quickml)
        assert "QuickML(best_model=" in repr_str
        assert "score=" in repr_str
        assert "task_type=" in repr_str


class TestIntegration:
    """Integration tests for the complete QuickML pipeline."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 200
        
        # Create more realistic test data
        self.classification_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples),
            'categorical1': np.random.choice(['A', 'B', 'C'], n_samples),
            'categorical2': np.random.choice(['X', 'Y'], n_samples),
            'target': np.random.choice([0, 1], n_samples)
        })
        
        # Add some missing values
        self.classification_data.loc[0:10, 'feature1'] = np.nan
        self.classification_data.loc[5:15, 'categorical1'] = np.nan
    
    def test_complete_pipeline_classification(self):
        """Test complete QuickML pipeline for classification."""
        quickml = QuickML()
        
        # Fit the model
        results = quickml.fit(self.classification_data, cv_folds=3)
        
        # Verify results
        assert results['task_type'] == 'classification'
        assert results['best_score'] > 0
        assert results['best_model_name'] is not None
        
        # Test predictions
        predictions = quickml.predict(self.classification_data)
        assert len(predictions) == len(self.classification_data)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Test probabilities
        probabilities = quickml.predict_proba(self.classification_data)
        assert probabilities.shape == (len(self.classification_data), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        
        # Test feature importance
        importance = quickml.get_feature_importance()
        if importance is not None:
            assert len(importance) > 0
        
        # Test model summary
        summary = quickml.get_model_summary()
        assert len(summary) > 0
    
    def test_model_persistence(self):
        """Test complete model save/load cycle."""
        quickml = QuickML()
        quickml.fit(self.classification_data, cv_folds=3)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            quickml.save_model(model_path)
            
            # Load model
            new_quickml = QuickML()
            new_quickml.load_model(model_path)
            
            # Test predictions are identical
            original_pred = quickml.predict(self.classification_data)
            loaded_pred = new_quickml.predict(self.classification_data)
            np.testing.assert_array_equal(original_pred, loaded_pred)
            
            # Test probabilities are identical
            original_proba = quickml.predict_proba(self.classification_data)
            loaded_proba = new_quickml.predict_proba(self.classification_data)
            np.testing.assert_array_almost_equal(original_proba, loaded_proba)
            
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        quickml = QuickML()
        
        # Test prediction without fitting
        with pytest.raises(ValueError, match="Model must be fitted before making predictions"):
            quickml.predict(self.classification_data)
        
        # Test probability prediction without fitting
        with pytest.raises(ValueError, match="Probability predictions only available for classification tasks"):
            quickml.predict_proba(self.classification_data)
        
        # Test feature importance without fitting
        with pytest.raises(ValueError, match="Model must be fitted before getting feature importance"):
            quickml.get_feature_importance()
        
        # Test model summary without fitting
        with pytest.raises(ValueError, match="Models must be trained before getting summary"):
            quickml.get_model_summary()
        
        # Test results summary without fitting
        with pytest.raises(ValueError, match="Model must be fitted before getting results summary"):
            quickml.get_results_summary()
        
        # Test save without fitting
        with pytest.raises(ValueError, match="Model must be fitted before saving"):
            quickml.save_model("test.pkl")
        
        # Test load non-existent file
        with pytest.raises(FileNotFoundError):
            quickml.load_model("non_existent_file.pkl")


if __name__ == "__main__":
    pytest.main([__file__])
