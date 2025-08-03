"""
Behavior-Based Script Detector - Analyzer Package

This package contains the core analysis components for detecting
suspicious patterns in Python scripts using AST analysis.
"""

from .pattern_rules import PatternRules
from .score_calculator import ScoreCalculator
from .report_generator import ReportGenerator

__all__ = ['PatternRules', 'ScoreCalculator', 'ReportGenerator'] 