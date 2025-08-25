"""
CodeSage - AI-Based Code Complexity Estimator

A powerful tool for analyzing code complexity using AST analysis and static metrics.
"""

__version__ = "0.1.0"
__author__ = "CodeSage Team"

from .analyzer import CodeAnalyzer
from .metrics import ComplexityMetrics
from .reporter import ReportGenerator

__all__ = ["CodeAnalyzer", "ComplexityMetrics", "ReportGenerator"]
