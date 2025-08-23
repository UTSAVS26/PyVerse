"""
QuickML - Mini AutoML Engine

A powerful yet simple AutoML engine that automatically processes any CSV dataset
and finds the best machine learning model.
"""

__version__ = "1.0.0"
__author__ = "QuickML Team"

from .core import QuickML
from .preprocessing import DataPreprocessor
from .models import ModelTrainer
from .evaluation import ModelEvaluator
from .visualization import Visualizer

__all__ = [
    "QuickML",
    "DataPreprocessor", 
    "ModelTrainer",
    "ModelEvaluator",
    "Visualizer"
]
