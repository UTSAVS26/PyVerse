"""
Tests for the metrics module.
"""

import pytest
import ast
import tempfile
import os
from unittest.mock import patch, MagicMock
import numpy as np

from codesage.metrics import ComplexityMetrics, FunctionMetrics, FileMetrics


class TestComplexityMetrics:
    """Test cases for ComplexityMetrics class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = ComplexityMetrics()
    
    def test_calculate_cyclomatic_complexity_simple(self):
        """Test cyclomatic complexity calculation for simple function."""
        code = """
def simple_function():
    return 42
"""
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        complexity = self.metrics.calculate_cyclomatic_complexity(func_node)
        assert complexity == 1  # Base complexity only
    
    def test_calculate_cyclomatic_complexity_with_conditionals(self):
        """Test cyclomatic complexity with conditionals."""
        code = """
def complex_function(x, y):
    if x > 0:
        if y > 0:
            return x + y
        else:
            return x - y
    else:
        return 0
"""
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        complexity = self.metrics.calculate_cyclomatic_complexity(func_node)
        assert complexity == 3  # Base + 2 if statements
    
    def test_calculate_cyclomatic_complexity_with_loops(self):
        """Test cyclomatic complexity with loops."""
        code = """
def loop_function(items):
    result = 0
    for item in items:
        if item > 0:
            result += item
        while item > 10:
            item -= 1
    return result
"""
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        complexity = self.metrics.calculate_cyclomatic_complexity(func_node)
        assert complexity == 4  # Base + for + if + while
    
    def test_calculate_nesting_depth_simple(self):
        """Test nesting depth calculation for simple function."""
        code = """
def simple_function():
    return 42
"""
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        depth = self.metrics.calculate_nesting_depth(func_node)
        assert depth == 0  # No nesting
    
    def test_calculate_nesting_depth_deep(self):
        """Test nesting depth calculation for deeply nested function."""
        code = """
def deep_function(x):
    if x > 0:
        if x > 10:
            if x > 100:
                for i in range(x):
                    if i % 2 == 0:
                        while i > 0:
                            i -= 1
    return x
"""
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        depth = self.metrics.calculate_nesting_depth(func_node)
        assert depth == 6  # if -> if -> if -> for -> if -> while (corrected)
    
    def test_calculate_maintainability_index(self):
        """Test maintainability index calculation."""
        metrics = FunctionMetrics(
            name="test_function",
            cyclomatic_complexity=5,
            lines_of_code=20,
            nesting_depth=2,
            parameters=3,
            return_statements=1,
            comments_ratio=0.1,
            maintainability_index=0.0,
            ai_risk_score=0.0
        )
        
        mi = self.metrics.calculate_maintainability_index(metrics)
        assert 0 <= mi <= 100
        # Note: mi can be int or float, both are valid
    
    def test_train_anomaly_detector(self):
        """Test ML model training."""
        training_data = [
            {'cyclomatic_complexity': 5, 'lines_of_code': 20, 'nesting_depth': 2, 'parameters': 3, 'comments_ratio': 0.1},
            {'cyclomatic_complexity': 10, 'lines_of_code': 50, 'nesting_depth': 3, 'parameters': 5, 'comments_ratio': 0.05},
            {'cyclomatic_complexity': 15, 'lines_of_code': 100, 'nesting_depth': 4, 'parameters': 7, 'comments_ratio': 0.02},
        ]
        
        self.metrics.train_anomaly_detector(training_data)
        assert self.metrics.is_trained == True
    
    def test_calculate_ai_risk_score_trained(self):
        """Test AI risk score calculation with trained model."""
        # Train the model first
        training_data = [
            {'cyclomatic_complexity': 5, 'lines_of_code': 20, 'nesting_depth': 2, 'parameters': 3, 'comments_ratio': 0.1},
            {'cyclomatic_complexity': 10, 'lines_of_code': 50, 'nesting_depth': 3, 'parameters': 5, 'comments_ratio': 0.05},
        ]
        self.metrics.train_anomaly_detector(training_data)
        
        metrics = FunctionMetrics(
            name="test_function",
            cyclomatic_complexity=15,
            lines_of_code=80,
            nesting_depth=4,
            parameters=6,
            return_statements=2,
            comments_ratio=0.02,
            maintainability_index=0.0,
            ai_risk_score=0.0
        )
        
        risk_score = self.metrics.calculate_ai_risk_score(metrics)
        assert 0 <= risk_score <= 100
        # Note: risk_score can be int or float, both are valid
    
    def test_calculate_ai_risk_score_untrained(self):
        """Test AI risk score calculation without trained model."""
        metrics = FunctionMetrics(
            name="test_function",
            cyclomatic_complexity=20,
            lines_of_code=100,
            nesting_depth=5,
            parameters=8,
            return_statements=3,
            comments_ratio=0.01,
            maintainability_index=0.0,
            ai_risk_score=0.0
        )
        
        risk_score = self.metrics.calculate_ai_risk_score(metrics)
        assert 0 <= risk_score <= 100
        # Note: risk_score can be int or float, both are valid
        # Should be high due to high complexity
        assert risk_score > 50
    
    def test_analyze_function(self):
        """Test complete function analysis."""
        code = """
def test_function(x, y):
    # This is a test function
    if x > 0:
        result = x + y
        if y > 0:
            result *= 2
        return result
    return 0
"""
        tree = ast.parse(code)
        func_node = tree.body[0]
        source_lines = code.splitlines()
        
        metrics = self.metrics.analyze_function(func_node, source_lines)
        
        assert metrics.name == "test_function"
        assert metrics.cyclomatic_complexity == 3  # Base + if + if (corrected)
        assert metrics.lines_of_code > 0
        assert metrics.nesting_depth == 2  # Two levels of nesting (corrected)
        assert metrics.parameters == 2  # x, y
        assert metrics.return_statements == 2
        assert metrics.comments_ratio > 0
        assert 0 <= metrics.maintainability_index <= 100
        assert 0 <= metrics.ai_risk_score <= 100
    
    def test_analyze_file(self):
        """Test complete file analysis."""
        code = """
def function1():
    return 1

def function2(x):
    if x > 0:
        return x * 2
    return 0

# Main function
def main():
    result = function1()
    print(result)
"""
        tree = ast.parse(code)
        source_lines = code.splitlines()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            filename = f.name
        
        try:
            file_metrics = self.metrics.analyze_file(tree, filename, source_lines)
            
            assert file_metrics.filename == filename
            assert file_metrics.total_lines == len(source_lines)
            assert file_metrics.total_functions == 3
            assert file_metrics.average_complexity > 0
            assert 0 <= file_metrics.maintainability_index <= 100
            assert 0 <= file_metrics.ai_anomaly_score <= 100
            
            # Check individual functions
            assert len(file_metrics.functions) == 3
            function_names = {f.name for f in file_metrics.functions}
            assert function_names == {'function1', 'function2', 'main'}
            
        finally:
            os.unlink(filename)
    
    def test_analyze_file_no_functions(self):
        """Test file analysis with no functions."""
        code = """
# This is a comment
import os
import sys

CONSTANT = 42
"""
        tree = ast.parse(code)
        source_lines = code.splitlines()
        
        file_metrics = self.metrics.analyze_file(tree, "test.py", source_lines)
        
        assert file_metrics.filename == "test.py"
        assert file_metrics.total_lines == len(source_lines)
        assert file_metrics.total_functions == 0
        assert file_metrics.average_complexity == 0.0
        assert file_metrics.maintainability_index == 100.0
        assert file_metrics.ai_anomaly_score == 0.0
        assert len(file_metrics.functions) == 0
    
    def test_complexity_thresholds(self):
        """Test complexity threshold configuration."""
        assert self.metrics.complexity_thresholds['low'] == 5
        assert self.metrics.complexity_thresholds['medium'] == 10
        assert self.metrics.complexity_thresholds['high'] == 15
        assert self.metrics.complexity_thresholds['critical'] == 25
    
    def test_rule_based_risk_score(self):
        """Test rule-based risk scoring."""
        # Low risk function
        low_risk = FunctionMetrics(
            name="low_risk",
            cyclomatic_complexity=3,
            lines_of_code=20,
            nesting_depth=1,
            parameters=2,
            return_statements=1,
            comments_ratio=0.2,
            maintainability_index=0.0,
            ai_risk_score=0.0
        )
        
        low_score = self.metrics._rule_based_risk_score(low_risk)
        assert low_score < 30
        
        # High risk function
        high_risk = FunctionMetrics(
            name="high_risk",
            cyclomatic_complexity=30,
            lines_of_code=150,
            nesting_depth=7,
            parameters=10,
            return_statements=5,
            comments_ratio=0.01,
            maintainability_index=0.0,
            ai_risk_score=0.0
        )
        
        high_score = self.metrics._rule_based_risk_score(high_risk)
        assert high_score > 70
    
    def test_ml_model_integration(self):
        """Test ML model integration with mocked dependencies."""
        # Test with real data to ensure the model trains
        training_data = [
            {'cyclomatic_complexity': 5, 'lines_of_code': 20, 'nesting_depth': 2, 'parameters': 3, 'comments_ratio': 0.1},
            {'cyclomatic_complexity': 10, 'lines_of_code': 50, 'nesting_depth': 3, 'parameters': 5, 'comments_ratio': 0.05}
        ]
        
        self.metrics.train_anomaly_detector(training_data)
        
        # Verify model was trained
        assert self.metrics.is_trained == True


class TestFunctionMetrics:
    """Test cases for FunctionMetrics dataclass."""
    
    def test_function_metrics_creation(self):
        """Test FunctionMetrics object creation."""
        metrics = FunctionMetrics(
            name="test_function",
            cyclomatic_complexity=5,
            lines_of_code=20,
            nesting_depth=2,
            parameters=3,
            return_statements=1,
            comments_ratio=0.1,
            maintainability_index=75.5,
            ai_risk_score=25.0
        )
        
        assert metrics.name == "test_function"
        assert metrics.cyclomatic_complexity == 5
        assert metrics.lines_of_code == 20
        assert metrics.nesting_depth == 2
        assert metrics.parameters == 3
        assert metrics.return_statements == 1
        assert metrics.comments_ratio == 0.1
        assert metrics.maintainability_index == 75.5
        assert metrics.ai_risk_score == 25.0


class TestFileMetrics:
    """Test cases for FileMetrics dataclass."""
    
    def test_file_metrics_creation(self):
        """Test FileMetrics object creation."""
        function_metrics = [
            FunctionMetrics(
                name="func1",
                cyclomatic_complexity=3,
                lines_of_code=15,
                nesting_depth=1,
                parameters=2,
                return_statements=1,
                comments_ratio=0.1,
                maintainability_index=80.0,
                ai_risk_score=20.0
            )
        ]
        
        file_metrics = FileMetrics(
            filename="test.py",
            functions=function_metrics,
            total_lines=50,
            total_functions=1,
            average_complexity=3.0,
            maintainability_index=80.0,
            ai_anomaly_score=20.0
        )
        
        assert file_metrics.filename == "test.py"
        assert len(file_metrics.functions) == 1
        assert file_metrics.total_lines == 50
        assert file_metrics.total_functions == 1
        assert file_metrics.average_complexity == 3.0
        assert file_metrics.maintainability_index == 80.0
        assert file_metrics.ai_anomaly_score == 20.0
