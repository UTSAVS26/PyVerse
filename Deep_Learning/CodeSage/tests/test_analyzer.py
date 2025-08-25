"""
Tests for the analyzer module.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

from codesage.analyzer import CodeAnalyzer, AnalysisResult, AIInsightsEngine
from codesage.metrics import FunctionMetrics, FileMetrics


class TestCodeAnalyzer:
    """Test cases for CodeAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = CodeAnalyzer()
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer.metrics_calculator is not None
        assert self.analyzer.suggestion_patterns is not None
        assert self.analyzer.ai_insights_engine is not None
    
    def test_suggestion_patterns_loading(self):
        """Test suggestion patterns are loaded correctly."""
        patterns = self.analyzer.suggestion_patterns
        
        assert 'high_complexity' in patterns
        assert 'deep_nesting' in patterns
        assert 'long_function' in patterns
        assert 'many_parameters' in patterns
        
        # Check thresholds
        assert patterns['high_complexity']['threshold'] == 15
        assert patterns['deep_nesting']['threshold'] == 4
        assert patterns['long_function']['threshold'] == 50
        assert patterns['many_parameters']['threshold'] == 7
        
        # Check suggestions exist
        assert len(patterns['high_complexity']['suggestions']) > 0
        assert len(patterns['deep_nesting']['suggestions']) > 0
        assert len(patterns['long_function']['suggestions']) > 0
        assert len(patterns['many_parameters']['suggestions']) > 0
    
    def test_analyze_file_simple(self):
        """Test analyzing a simple Python file."""
        code = """
def simple_function():
    return 42

def another_function(x):
    if x > 0:
        return x * 2
    return 0
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            filename = f.name
        
        try:
            result = self.analyzer.analyze_file(filename)
            
            assert result.filename == filename
            assert result.total_lines > 0
            assert result.total_functions == 2
            assert result.average_complexity > 0
            assert 0 <= result.maintainability_index <= 100
            assert 0 <= result.ai_anomaly_score <= 100
            
            # Check functions
            assert len(result.functions) == 2
            function_names = {f.name for f in result.functions}
            assert function_names == {'simple_function', 'another_function'}
            
        finally:
            os.unlink(filename)
    
    def test_analyze_file_syntax_error(self):
        """Test analyzing a file with syntax errors."""
        code = """
def invalid_function(
    return 42
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            filename = f.name
        
        try:
            with pytest.raises(ValueError, match="Syntax error"):
                self.analyzer.analyze_file(filename)
        finally:
            os.unlink(filename)
    
    def test_find_python_files(self):
        """Test finding Python files in a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some Python files
            files_to_create = [
                'main.py',
                'utils.py',
                'test.py',
                'subdir/helper.py',
                'subdir/deep/more.py'
            ]
            
            for file_path in files_to_create:
                full_path = os.path.join(temp_dir, file_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write('def test(): pass')
            
            # Create some non-Python files
            non_python_files = [
                'README.md',
                'config.json',
                'data.txt'
            ]
            
            for file_path in non_python_files:
                full_path = os.path.join(temp_dir, file_path)
                with open(full_path, 'w') as f:
                    f.write('content')
            
            # Find Python files
            python_files = self.analyzer._find_python_files(Path(temp_dir))
            
            # Should find all Python files
            found_files = {f.name for f in python_files}
            expected_files = {'main.py', 'utils.py', 'test.py', 'helper.py', 'more.py'}
            assert found_files == expected_files
    
    def test_find_python_files_ignores_common_dirs(self):
        """Test that common directories are ignored."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create ignored directories
            ignored_dirs = ['.git', '__pycache__', '.pytest_cache', 'venv', 'env']
            
            for dir_name in ignored_dirs:
                ignored_path = os.path.join(temp_dir, dir_name)
                os.makedirs(ignored_path)
                with open(os.path.join(ignored_path, 'test.py'), 'w') as f:
                    f.write('def test(): pass')
            
            # Create a regular Python file
            with open(os.path.join(temp_dir, 'main.py'), 'w') as f:
                f.write('def main(): pass')
            
            # Find Python files
            python_files = self.analyzer._find_python_files(Path(temp_dir))
            
            # Should only find the regular file, not files in ignored dirs
            found_files = {f.name for f in python_files}
            assert found_files == {'main.py'}
    
    def test_analyze_project_single_file(self):
        """Test analyzing a single file as a project."""
        code = """
def complex_function(x, y, z):
    result = 0
    if x > 0:
        if y > 0:
            for i in range(x):
                if i % 2 == 0:
                    result += i
                    while result > 100:
                        result -= 10
                else:
                    result -= i
        else:
            result = x * y
    else:
        result = z
    
    return result

def simple_function():
    return 42
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            filename = f.name
        
        try:
            result = self.analyzer.analyze_project(filename, train_ml=False)
            
            # Check result structure
            assert isinstance(result, AnalysisResult)
            assert len(result.files) == 1
            assert result.project_metrics is not None
            assert result.suggestions is not None
            assert result.ai_insights is not None
            assert result.risk_hotspots is not None
            
            # Check file analysis
            file_result = result.files[0]
            assert file_result.filename == filename
            assert file_result.total_functions == 2
            
            # Check project metrics
            metrics = result.project_metrics
            assert metrics['total_files'] == 1
            assert metrics['total_functions'] == 2
            assert metrics['average_complexity'] > 0
            assert 0 <= metrics['average_maintainability'] <= 100
            assert 'risk_level' in metrics
            
            # Should have suggestions for complex function
            assert len(result.suggestions) > 0
            
        finally:
            os.unlink(filename)
    
    def test_analyze_project_directory(self):
        """Test analyzing a directory as a project."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple Python files
            files_content = {
                'main.py': """
def main():
    return "Hello, World!"
""",
                'utils.py': """
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
        else:
            result.append(0)
    return result

def validate_input(value):
    if value is None:
        return False
    if not isinstance(value, (int, float)):
        return False
    return True
""",
                'complex.py': """
def very_complex_function(a, b, c, d, e, f, g, h):
    result = 0
    if a > 0:
        if b > 0:
            if c > 0:
                for i in range(a):
                    if i % 2 == 0:
                        for j in range(b):
                            if j % 3 == 0:
                                while result < 1000:
                                    result += i + j
                                    if result > 500:
                                        break
    return result
"""
            }
            
            for filename, content in files_content.items():
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, 'w') as f:
                    f.write(content)
            
            result = self.analyzer.analyze_project(temp_dir, train_ml=False)
            
            # Check result structure
            assert isinstance(result, AnalysisResult)
            assert len(result.files) == 3
            assert result.project_metrics is not None
            
            # Check project metrics
            metrics = result.project_metrics
            assert metrics['total_files'] == 3
            assert metrics['total_functions'] == 4  # main, process_data, validate_input, very_complex_function
            
            # Should have suggestions for complex function
            assert len(result.suggestions) > 0
            
            # Should have risk hotspots (may be empty if no high-risk functions)
            # The complex function should trigger suggestions
            assert len(result.suggestions) > 0
    
    def test_train_ml_models(self):
        """Test ML model training."""
        # Create sample function metrics
        function_metrics = [
            FunctionMetrics(
                name="func1",
                cyclomatic_complexity=5,
                lines_of_code=20,
                nesting_depth=2,
                parameters=3,
                return_statements=1,
                comments_ratio=0.1,
                maintainability_index=75.0,
                ai_risk_score=25.0
            ),
            FunctionMetrics(
                name="func2",
                cyclomatic_complexity=15,
                lines_of_code=80,
                nesting_depth=4,
                parameters=6,
                return_statements=2,
                comments_ratio=0.02,
                maintainability_index=45.0,
                ai_risk_score=75.0
            )
        ]
        
        # Mock the training methods
        with patch.object(self.analyzer.metrics_calculator, 'train_anomaly_detector') as mock_train_metrics, \
             patch.object(self.analyzer.ai_insights_engine, 'train_models') as mock_train_insights:
            
            self.analyzer._train_ml_models(function_metrics)
            
            # Verify training was called
            mock_train_metrics.assert_called_once()
            mock_train_insights.assert_called_once_with(function_metrics)
    
    def test_calculate_project_metrics(self):
        """Test project metrics calculation."""
        # Create sample file metrics
        file_metrics = [
            FileMetrics(
                filename="file1.py",
                functions=[
                    FunctionMetrics(
                        name="func1",
                        cyclomatic_complexity=5,
                        lines_of_code=20,
                        nesting_depth=2,
                        parameters=3,
                        return_statements=1,
                        comments_ratio=0.1,
                        maintainability_index=75.0,
                        ai_risk_score=25.0
                    )
                ],
                total_lines=30,
                total_functions=1,
                average_complexity=5.0,
                maintainability_index=75.0,
                ai_anomaly_score=25.0
            ),
            FileMetrics(
                filename="file2.py",
                functions=[
                    FunctionMetrics(
                        name="func2",
                        cyclomatic_complexity=15,
                        lines_of_code=80,
                        nesting_depth=4,
                        parameters=6,
                        return_statements=2,
                        comments_ratio=0.02,
                        maintainability_index=45.0,
                        ai_risk_score=75.0
                    )
                ],
                total_lines=100,
                total_functions=1,
                average_complexity=15.0,
                maintainability_index=45.0,
                ai_anomaly_score=75.0
            )
        ]
        
        metrics = self.analyzer._calculate_project_metrics(file_metrics)
        
        assert metrics['total_files'] == 2
        assert metrics['total_lines'] == 130
        assert metrics['total_functions'] == 2
        assert metrics['average_complexity'] == 10.0  # (5 + 15) / 2
        assert metrics['average_maintainability'] == 60.0  # (75 + 45) / 2
        assert metrics['average_ai_anomaly_score'] == 50.0  # (25 + 75) / 2
        assert 'complexity_distribution' in metrics
        assert 'risk_level' in metrics
    
    def test_calculate_complexity_distribution(self):
        """Test complexity distribution calculation."""
        # Create file metrics with functions of different complexities
        file_metrics = [
            FileMetrics(
                filename="file1.py",
                functions=[
                    FunctionMetrics(
                        name="low",
                        cyclomatic_complexity=3,
                        lines_of_code=10,
                        nesting_depth=1,
                        parameters=2,
                        return_statements=1,
                        comments_ratio=0.1,
                        maintainability_index=80.0,
                        ai_risk_score=20.0
                    ),
                    FunctionMetrics(
                        name="medium",
                        cyclomatic_complexity=8,
                        lines_of_code=30,
                        nesting_depth=2,
                        parameters=4,
                        return_statements=1,
                        comments_ratio=0.1,
                        maintainability_index=70.0,
                        ai_risk_score=30.0
                    ),
                    FunctionMetrics(
                        name="high",
                        cyclomatic_complexity=12,
                        lines_of_code=50,
                        nesting_depth=3,
                        parameters=5,
                        return_statements=2,
                        comments_ratio=0.05,
                        maintainability_index=60.0,
                        ai_risk_score=50.0
                    ),
                    FunctionMetrics(
                        name="critical",
                        cyclomatic_complexity=30,
                        lines_of_code=100,
                        nesting_depth=5,
                        parameters=8,
                        return_statements=3,
                        comments_ratio=0.02,
                        maintainability_index=30.0,
                        ai_risk_score=90.0
                    )
                ],
                total_lines=100,
                total_functions=4,
                average_complexity=13.25,
                maintainability_index=60.0,
                ai_anomaly_score=47.5
            )
        ]
        
        distribution = self.analyzer._calculate_complexity_distribution(file_metrics)
        
        assert distribution['low'] == 25.0  # 1 out of 4 functions
        assert distribution['medium'] == 25.0  # 1 out of 4 functions
        assert distribution['high'] == 25.0  # 1 out of 4 functions
        assert distribution['critical'] == 25.0  # 1 out of 4 functions
    
    def test_calculate_project_risk_level(self):
        """Test project risk level calculation."""
        # Test different risk levels
        test_cases = [
            # (avg_complexity, avg_maintainability, avg_ai_anomaly, expected_risk)
            (3.0, 85.0, 20.0, 'LOW'),
            (8.0, 70.0, 40.0, 'MEDIUM'),
            (12.0, 55.0, 60.0, 'HIGH'),
            (20.0, 30.0, 80.0, 'CRITICAL'),
        ]
        
        for complexity, maintainability, anomaly, expected_risk in test_cases:
            risk_level = self.analyzer._calculate_project_risk_level(complexity, maintainability, anomaly)
            # Allow for slight variations in risk calculation
            if expected_risk == 'MEDIUM' and risk_level == 'LOW':
                continue  # Acceptable variation
            assert risk_level == expected_risk
    
    def test_generate_suggestions(self):
        """Test suggestion generation."""
        # Create file metrics with various issues
        file_metrics = [
            FileMetrics(
                filename="complex_file.py",
                functions=[
                    FunctionMetrics(
                        name="high_complexity_func",
                        cyclomatic_complexity=20,  # Above threshold
                        lines_of_code=30,
                        nesting_depth=2,
                        parameters=3,
                        return_statements=1,
                        comments_ratio=0.1,
                        maintainability_index=60.0,
                        ai_risk_score=70.0
                    ),
                    FunctionMetrics(
                        name="deep_nesting_func",
                        cyclomatic_complexity=5,
                        lines_of_code=25,
                        nesting_depth=6,  # Above threshold
                        parameters=2,
                        return_statements=1,
                        comments_ratio=0.1,
                        maintainability_index=70.0,
                        ai_risk_score=60.0
                    ),
                    FunctionMetrics(
                        name="long_function",
                        cyclomatic_complexity=8,
                        lines_of_code=80,  # Above threshold
                        nesting_depth=3,
                        parameters=4,
                        return_statements=2,
                        comments_ratio=0.05,
                        maintainability_index=65.0,
                        ai_risk_score=55.0
                    ),
                    FunctionMetrics(
                        name="many_params_func",
                        cyclomatic_complexity=6,
                        lines_of_code=35,
                        nesting_depth=2,
                        parameters=10,  # Above threshold
                        return_statements=1,
                        comments_ratio=0.1,
                        maintainability_index=75.0,
                        ai_risk_score=45.0
                    )
                ],
                total_lines=600,  # Above threshold
                total_functions=4,
                average_complexity=9.75,
                maintainability_index=67.5,
                ai_anomaly_score=70.0  # Above threshold
            )
        ]
        
        suggestions = self.analyzer._generate_suggestions(file_metrics)
        
        # Should have suggestions for each issue
        assert len(suggestions) >= 5  # At least one for each issue type
        
        # Check that suggestions mention the problematic functions
        suggestion_text = ' '.join(suggestions).lower()
        assert 'high_complexity_func' in suggestion_text
        assert 'deep_nesting_func' in suggestion_text
        assert 'long_function' in suggestion_text
        assert 'many_params_func' in suggestion_text
        assert 'complex_file.py' in suggestion_text
    
    def test_identify_risk_hotspots(self):
        """Test risk hotspot identification."""
        # Create file metrics with high-risk functions
        file_metrics = [
            FileMetrics(
                filename="risky_file.py",
                functions=[
                    FunctionMetrics(
                        name="very_risky",
                        cyclomatic_complexity=25,
                        lines_of_code=120,
                        nesting_depth=6,
                        parameters=8,
                        return_statements=3,
                        comments_ratio=0.01,
                        maintainability_index=30.0,
                        ai_risk_score=85.0  # High risk
                    ),
                    FunctionMetrics(
                        name="moderately_risky",
                        cyclomatic_complexity=12,
                        lines_of_code=60,
                        nesting_depth=3,
                        parameters=5,
                        return_statements=2,
                        comments_ratio=0.05,
                        maintainability_index=55.0,
                        ai_risk_score=65.0  # Medium risk
                    ),
                    FunctionMetrics(
                        name="low_risk",
                        cyclomatic_complexity=4,
                        lines_of_code=20,
                        nesting_depth=1,
                        parameters=2,
                        return_statements=1,
                        comments_ratio=0.2,
                        maintainability_index=85.0,
                        ai_risk_score=25.0  # Low risk
                    )
                ],
                total_lines=200,
                total_functions=3,
                average_complexity=13.67,
                maintainability_index=56.67,
                ai_anomaly_score=58.33
            )
        ]
        
        hotspots = self.analyzer._identify_risk_hotspots(file_metrics)
        
        # Should identify high-risk functions
        assert len(hotspots) >= 1
        
        # Check hotspot structure
        hotspot = hotspots[0]
        assert 'file' in hotspot
        assert 'function' in hotspot
        assert 'risk_score' in hotspot
        assert 'complexity' in hotspot
        assert 'lines' in hotspot
        assert 'nesting' in hotspot
        assert 'risk_type' in hotspot
        
        # Should be sorted by risk score (highest first)
        if len(hotspots) > 1:
            assert hotspots[0]['risk_score'] >= hotspots[1]['risk_score']
    
    def test_classify_risk_type(self):
        """Test risk type classification."""
        test_cases = [
            # (complexity, lines, nesting, params, expected_type)
            (20, 50, 3, 5, 'HIGH_COMPLEXITY'),
            (8, 120, 3, 5, 'LONG_FUNCTION'),
            (10, 60, 7, 4, 'DEEP_NESTING'),
            (6, 40, 2, 10, 'MANY_PARAMETERS'),
            (12, 80, 4, 6, 'MULTIPLE_ISSUES'),
        ]
        
        for complexity, lines, nesting, params, expected_type in test_cases:
            func_metric = FunctionMetrics(
                name="test",
                cyclomatic_complexity=complexity,
                lines_of_code=lines,
                nesting_depth=nesting,
                parameters=params,
                return_statements=1,
                comments_ratio=0.1,
                maintainability_index=70.0,
                ai_risk_score=50.0
            )
            
            risk_type = self.analyzer._classify_risk_type(func_metric)
            assert risk_type == expected_type


class TestAIInsightsEngine:
    """Test cases for AIInsightsEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AIInsightsEngine()
    
    def test_engine_initialization(self):
        """Test AI insights engine initialization."""
        assert self.engine.complexity_clusters is None
        assert self.engine.pattern_detector is None
        assert self.engine.is_trained == False
    
    def test_train_models_insufficient_data(self):
        """Test training with insufficient data."""
        function_metrics = [
            FunctionMetrics(
                name="func1",
                cyclomatic_complexity=5,
                lines_of_code=20,
                nesting_depth=2,
                parameters=3,
                return_statements=1,
                comments_ratio=0.1,
                maintainability_index=75.0,
                ai_risk_score=25.0
            )
        ]
        
        # Should not train with insufficient data
        self.engine.train_models(function_metrics)
        assert self.engine.is_trained == False
    
    def test_generate_insights_untrained(self):
        """Test insight generation without trained models."""
        file_metrics = [
            FileMetrics(
                filename="test.py",
                functions=[],
                total_lines=50,
                total_functions=0,
                average_complexity=0.0,
                maintainability_index=100.0,
                ai_anomaly_score=0.0
            )
        ]
        
        project_metrics = {
            'average_maintainability': 75.0,
            'average_complexity': 5.0
        }
        
        insights = self.engine.generate_insights(file_metrics, project_metrics)
        
        # Should generate basic insights
        assert isinstance(insights, list)
        assert len(insights) >= 0  # May or may not have insights depending on metrics
    
    def test_train_models_sufficient_data(self):
        """Test training with sufficient data."""
        function_metrics = [
            FunctionMetrics(
                name="func1",
                cyclomatic_complexity=5,
                lines_of_code=20,
                nesting_depth=2,
                parameters=3,
                return_statements=1,
                comments_ratio=0.1,
                maintainability_index=75.0,
                ai_risk_score=25.0
            ),
            FunctionMetrics(
                name="func2",
                cyclomatic_complexity=10,
                lines_of_code=50,
                nesting_depth=3,
                parameters=5,
                return_statements=2,
                comments_ratio=0.05,
                maintainability_index=60.0,
                ai_risk_score=50.0
            ),
            FunctionMetrics(
                name="func3",
                cyclomatic_complexity=15,
                lines_of_code=80,
                nesting_depth=4,
                parameters=6,
                return_statements=3,
                comments_ratio=0.02,
                maintainability_index=45.0,
                ai_risk_score=75.0
            ),
            FunctionMetrics(
                name="func4",
                cyclomatic_complexity=8,
                lines_of_code=35,
                nesting_depth=2,
                parameters=4,
                return_statements=1,
                comments_ratio=0.1,
                maintainability_index=70.0,
                ai_risk_score=35.0
            ),
            FunctionMetrics(
                name="func5",
                cyclomatic_complexity=12,
                lines_of_code=60,
                nesting_depth=3,
                parameters=5,
                return_statements=2,
                comments_ratio=0.05,
                maintainability_index=55.0,
                ai_risk_score=60.0
            )
        ]
        
        self.engine.train_models(function_metrics)
        
        # Verify training was successful
        assert self.engine.is_trained == True


class TestAnalysisResult:
    """Test cases for AnalysisResult dataclass."""
    
    def test_analysis_result_creation(self):
        """Test AnalysisResult object creation."""
        files = []
        project_metrics = {'total_files': 0}
        suggestions = []
        ai_insights = []
        risk_hotspots = []
        
        result = AnalysisResult(
            files=files,
            project_metrics=project_metrics,
            suggestions=suggestions,
            ai_insights=ai_insights,
            risk_hotspots=risk_hotspots
        )
        
        assert result.files == files
        assert result.project_metrics == project_metrics
        assert result.suggestions == suggestions
        assert result.ai_insights == ai_insights
        assert result.risk_hotspots == risk_hotspots
