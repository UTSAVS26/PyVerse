"""
Tests for the reporter module.
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
import numpy as np

from codesage.reporter import ReportGenerator
from codesage.analyzer import AnalysisResult
from codesage.metrics import FunctionMetrics, FileMetrics


class TestReportGenerator:
    """Test cases for ReportGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.reporter = ReportGenerator()
    
    def test_reporter_initialization(self):
        """Test reporter initialization."""
        assert self.reporter.console is not None
    
    def test_generate_cli_report_basic(self):
        """Test basic CLI report generation."""
        # Create sample analysis result
        result = self._create_sample_analysis_result()
        
        # Test that no exception is raised
        try:
            self.reporter.generate_cli_report(result, detailed=False)
        except Exception as e:
            pytest.fail(f"CLI report generation failed: {e}")
    
    def test_generate_cli_report_detailed(self):
        """Test detailed CLI report generation."""
        result = self._create_sample_analysis_result()
        
        try:
            self.reporter.generate_cli_report(result, detailed=True)
        except Exception as e:
            pytest.fail(f"Detailed CLI report generation failed: {e}")
    
    def test_generate_html_report(self):
        """Test HTML report generation."""
        result = self._create_sample_analysis_result()
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            output_path = f.name
        
        try:
            html_path = self.reporter.generate_html_report(result, output_path)
            
            assert html_path == output_path
            assert os.path.exists(html_path)
            
            # Check that HTML file contains expected content
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
                
            assert 'CodeSage Analysis Report' in html_content
            assert 'AI-Enhanced Code Complexity Analysis' in html_content
            # Check for any section headers
            assert 'section' in html_content
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_generate_html_report_default_path(self):
        """Test HTML report generation with default path."""
        result = self._create_sample_analysis_result()
        
        try:
            html_path = self.reporter.generate_html_report(result)
            
            assert html_path == "codesage_report.html"
            assert os.path.exists(html_path)
            
        finally:
            if os.path.exists("codesage_report.html"):
                os.unlink("codesage_report.html")
    
    def test_get_complexity_status(self):
        """Test complexity status calculation."""
        # Test different complexity levels
        assert self.reporter._get_complexity_status(3) == "✅"  # Low
        assert self.reporter._get_complexity_status(8) == "⚠️"  # Medium
        assert self.reporter._get_complexity_status(12) == "🔶"  # High
        assert self.reporter._get_complexity_status(25) == "🔴"  # Critical
    
    def test_get_maintainability_status(self):
        """Test maintainability status calculation."""
        # Test different maintainability levels
        assert self.reporter._get_maintainability_status(85) == "✅"  # High
        assert self.reporter._get_maintainability_status(70) == "⚠️"  # Medium
        assert self.reporter._get_maintainability_status(45) == "🔶"  # Low
        assert self.reporter._get_maintainability_status(25) == "🔴"  # Very low
    
    def test_get_risk_status(self):
        """Test risk status calculation."""
        # Test different risk levels
        assert self.reporter._get_risk_status('LOW') == "✅"
        assert self.reporter._get_risk_status('MEDIUM') == "⚠️"
        assert self.reporter._get_risk_status('HIGH') == "🔶"
        assert self.reporter._get_risk_status('CRITICAL') == "🔴"
        assert self.reporter._get_risk_status('UNKNOWN') == "❓"
    
    def test_print_complexity_distribution(self):
        """Test complexity distribution printing."""
        distribution = {
            'low': 60.0,
            'medium': 25.0,
            'high': 10.0,
            'critical': 5.0
        }
        
        # Test that no exception is raised
        try:
            self.reporter._print_complexity_distribution(distribution)
        except Exception as e:
            pytest.fail(f"Complexity distribution printing failed: {e}")
    
    def test_create_complexity_chart(self):
        """Test complexity chart creation."""
        result = self._create_sample_analysis_result()
        
        chart_html = self.reporter._create_complexity_chart(result)
        
        assert isinstance(chart_html, str)
        assert len(chart_html) > 0
        # Should contain plotly chart elements
        assert 'plotly' in chart_html.lower() or 'chart' in chart_html.lower()
    
    def test_create_complexity_chart_no_data(self):
        """Test complexity chart creation with no data."""
        # Create result with no functions
        result = AnalysisResult(
            files=[
                FileMetrics(
                    filename="empty.py",
                    functions=[],
                    total_lines=10,
                    total_functions=0,
                    average_complexity=0.0,
                    maintainability_index=100.0,
                    ai_anomaly_score=0.0
                )
            ],
            project_metrics={'total_files': 1, 'total_functions': 0},
            suggestions=[],
            ai_insights=[],
            risk_hotspots=[]
        )
        
        chart_html = self.reporter._create_complexity_chart(result)
        
        assert isinstance(chart_html, str)
        assert "No function complexity data available" in chart_html
    
    def test_create_maintainability_chart(self):
        """Test maintainability chart creation."""
        result = self._create_sample_analysis_result()
        
        chart_html = self.reporter._create_maintainability_chart(result)
        
        assert isinstance(chart_html, str)
        assert len(chart_html) > 0
        # Should contain plotly chart elements
        assert 'plotly' in chart_html.lower() or 'chart' in chart_html.lower()
    
    def test_create_maintainability_chart_no_data(self):
        """Test maintainability chart creation with no data."""
        result = AnalysisResult(
            files=[],
            project_metrics={},
            suggestions=[],
            ai_insights=[],
            risk_hotspots=[]
        )
        
        chart_html = self.reporter._create_maintainability_chart(result)
        
        assert isinstance(chart_html, str)
        assert "No data available for maintainability chart" in chart_html
    
    def test_create_risk_hotspots_chart(self):
        """Test risk hotspots chart creation."""
        result = self._create_sample_analysis_result()
        
        chart_html = self.reporter._create_risk_hotspots_chart(result)
        
        assert isinstance(chart_html, str)
        assert len(chart_html) > 0
        # Should contain plotly chart elements
        assert 'plotly' in chart_html.lower() or 'chart' in chart_html.lower()
    
    def test_create_risk_hotspots_chart_no_hotspots(self):
        """Test risk hotspots chart creation with no hotspots."""
        result = AnalysisResult(
            files=[],
            project_metrics={},
            suggestions=[],
            ai_insights=[],
            risk_hotspots=[]
        )
        
        chart_html = self.reporter._create_risk_hotspots_chart(result)
        
        assert isinstance(chart_html, str)
        assert "No risk hotspots identified" in chart_html
    
    def test_generate_html_content(self):
        """Test HTML content generation."""
        result = self._create_sample_analysis_result()
        
        html_content = self.reporter._generate_html_content(result)
        
        assert isinstance(html_content, str)
        assert len(html_content) > 0
        assert '<!DOCTYPE html>' in html_content
        assert '<html' in html_content
        assert '</html>' in html_content
        assert 'CodeSage Analysis Report' in html_content
        assert 'AI-Enhanced Code Complexity Analysis' in html_content
    
    def test_print_project_overview(self):
        """Test project overview printing."""
        project_metrics = {
            'total_files': 5,
            'total_lines': 1000,
            'total_functions': 25,
            'average_complexity': 8.5,
            'average_maintainability': 72.3,
            'risk_level': 'MEDIUM',
            'complexity_distribution': {
                'low': 60.0,
                'medium': 25.0,
                'high': 10.0,
                'critical': 5.0
            }
        }
        
        # Test that no exception is raised
        try:
            self.reporter._print_project_overview(project_metrics)
        except Exception as e:
            pytest.fail(f"Project overview printing failed: {e}")
    
    def test_print_ai_insights(self):
        """Test AI insights printing."""
        ai_insights = [
            "AI detected a cluster of 5 functions with high complexity",
            "Files contain many functions on average",
            "More than 30% of files have low maintainability scores"
        ]
        
        # Test that no exception is raised
        try:
            self.reporter._print_ai_insights(ai_insights)
        except Exception as e:
            pytest.fail(f"AI insights printing failed: {e}")
    
    def test_print_risk_hotspots(self):
        """Test risk hotspots printing."""
        risk_hotspots = [
            {
                'file': 'complex_file.py',
                'function': 'very_complex_function',
                'risk_score': 85.5,
                'complexity': 25,
                'lines': 120,
                'nesting': 6,
                'risk_type': 'HIGH_COMPLEXITY'
            },
            {
                'file': 'another_file.py',
                'function': 'deep_nested_function',
                'risk_score': 72.3,
                'complexity': 15,
                'lines': 80,
                'nesting': 5,
                'risk_type': 'DEEP_NESTING'
            }
        ]
        
        # Test that no exception is raised
        try:
            self.reporter._print_risk_hotspots(risk_hotspots)
        except Exception as e:
            pytest.fail(f"Risk hotspots printing failed: {e}")
    
    def test_print_suggestions(self):
        """Test suggestions printing."""
        suggestions = [
            "Function 'complex_function' in main.py: Consider breaking this function into smaller, more focused functions",
            "Function 'deep_nested_function' in utils.py: Use early returns to reduce nesting depth",
            "File large_file.py is quite large (800 lines). Consider breaking it into smaller, more focused modules."
        ]
        
        # Test that no exception is raised
        try:
            self.reporter._print_suggestions(suggestions)
        except Exception as e:
            pytest.fail(f"Suggestions printing failed: {e}")
    
    def test_print_summary(self):
        """Test summary printing."""
        result = self._create_sample_analysis_result()
        
        # Test that no exception is raised
        try:
            self.reporter._print_summary(result)
        except Exception as e:
            pytest.fail(f"Summary printing failed: {e}")
    
    def test_print_detailed_file_analysis(self):
        """Test detailed file analysis printing."""
        result = self._create_sample_analysis_result()
        
        # Test that no exception is raised
        try:
            self.reporter._print_detailed_file_analysis(result.files)
        except Exception as e:
            pytest.fail(f"Detailed file analysis printing failed: {e}")
    
    def test_print_summary_file_analysis(self):
        """Test summary file analysis printing."""
        result = self._create_sample_analysis_result()
        
        # Test that no exception is raised
        try:
            self.reporter._print_summary_file_analysis(result.files)
        except Exception as e:
            pytest.fail(f"Summary file analysis printing failed: {e}")
    
    def _create_sample_analysis_result(self) -> AnalysisResult:
        """Create a sample analysis result for testing."""
        # Create sample functions
        functions = [
            FunctionMetrics(
                name="simple_function",
                cyclomatic_complexity=3,
                lines_of_code=15,
                nesting_depth=1,
                parameters=2,
                return_statements=1,
                comments_ratio=0.2,
                maintainability_index=85.0,
                ai_risk_score=20.0
            ),
            FunctionMetrics(
                name="complex_function",
                cyclomatic_complexity=18,
                lines_of_code=80,
                nesting_depth=4,
                parameters=6,
                return_statements=3,
                comments_ratio=0.05,
                maintainability_index=45.0,
                ai_risk_score=75.0
            ),
            FunctionMetrics(
                name="deep_nested_function",
                cyclomatic_complexity=12,
                lines_of_code=60,
                nesting_depth=6,
                parameters=4,
                return_statements=2,
                comments_ratio=0.1,
                maintainability_index=55.0,
                ai_risk_score=65.0
            )
        ]
        
        # Create sample files
        files = [
            FileMetrics(
                filename="main.py",
                functions=functions[:2],
                total_lines=150,
                total_functions=2,
                average_complexity=10.5,
                maintainability_index=65.0,
                ai_anomaly_score=47.5
            ),
            FileMetrics(
                filename="utils.py",
                functions=functions[2:],
                total_lines=100,
                total_functions=1,
                average_complexity=12.0,
                maintainability_index=55.0,
                ai_anomaly_score=65.0
            )
        ]
        
        # Create project metrics
        project_metrics = {
            'total_files': 2,
            'total_lines': 250,
            'total_functions': 3,
            'average_complexity': 11.0,
            'average_maintainability': 61.67,
            'average_ai_anomaly_score': 53.33,
            'complexity_distribution': {
                'low': 33.33,
                'medium': 0.0,
                'high': 33.33,
                'critical': 33.33
            },
            'risk_level': 'MEDIUM'
        }
        
        # Create suggestions
        suggestions = [
            "Function 'complex_function' in main.py: Consider breaking this function into smaller, more focused functions",
            "Function 'deep_nested_function' in utils.py: Use early returns to reduce nesting depth"
        ]
        
        # Create AI insights
        ai_insights = [
            "AI detected a cluster of 2 functions with high complexity (avg: 15.0). These may benefit from refactoring.",
            "Files contain few functions on average. This may indicate good separation of concerns."
        ]
        
        # Create risk hotspots
        risk_hotspots = [
            {
                'file': 'main.py',
                'function': 'complex_function',
                'risk_score': 75.0,
                'complexity': 18,
                'lines': 80,
                'nesting': 4,
                'risk_type': 'HIGH_COMPLEXITY'
            },
            {
                'file': 'utils.py',
                'function': 'deep_nested_function',
                'risk_score': 65.0,
                'complexity': 12,
                'lines': 60,
                'nesting': 6,
                'risk_type': 'DEEP_NESTING'
            }
        ]
        
        return AnalysisResult(
            files=files,
            project_metrics=project_metrics,
            suggestions=suggestions,
            ai_insights=ai_insights,
            risk_hotspots=risk_hotspots
        )
    
    @patch('plotly.graph_objects.Figure')
    def test_plotly_integration(self, mock_figure):
        """Test Plotly integration for chart generation."""
        # Mock Plotly components
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_trace.return_value = mock_fig
        mock_fig.update_layout.return_value = mock_fig
        mock_fig.to_html.return_value = "<div>Mock Chart</div>"
        
        result = self._create_sample_analysis_result()
        
        # Test chart generation
        chart_html = self.reporter._create_complexity_chart(result)
        
        assert chart_html == "<div>Mock Chart</div>"
        mock_fig.add_trace.assert_called()
        mock_fig.update_layout.assert_called()
        mock_fig.to_html.assert_called_once()
    
    def test_empty_analysis_result(self):
        """Test handling of empty analysis result."""
        empty_result = AnalysisResult(
            files=[],
            project_metrics={},
            suggestions=[],
            ai_insights=[],
            risk_hotspots=[]
        )
        
        # Test CLI report
        try:
            self.reporter.generate_cli_report(empty_result, detailed=False)
        except Exception as e:
            pytest.fail(f"Empty CLI report generation failed: {e}")
        
        # Test HTML report
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            output_path = f.name
        
        try:
            html_path = self.reporter.generate_html_report(empty_result, output_path)
            assert os.path.exists(html_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_large_analysis_result(self):
        """Test handling of large analysis result."""
        # Create many files and functions
        files = []
        for i in range(10):
            functions = []
            for j in range(5):
                functions.append(FunctionMetrics(
                    name=f"function_{i}_{j}",
                    cyclomatic_complexity=5 + (i + j) % 10,
                    lines_of_code=20 + (i + j) * 5,
                    nesting_depth=1 + (i + j) % 3,
                    parameters=2 + (i + j) % 5,
                    return_statements=1,
                    comments_ratio=0.1,
                    maintainability_index=70.0,
                    ai_risk_score=30.0
                ))
            
            files.append(FileMetrics(
                filename=f"file_{i}.py",
                functions=functions,
                total_lines=100 + i * 10,
                total_functions=5,
                average_complexity=8.0,
                maintainability_index=70.0,
                ai_anomaly_score=30.0
            ))
        
        project_metrics = {
            'total_files': 10,
            'total_lines': 1000,
            'total_functions': 50,
            'average_complexity': 8.0,
            'average_maintainability': 70.0,
            'average_ai_anomaly_score': 30.0,
            'complexity_distribution': {'low': 40, 'medium': 30, 'high': 20, 'critical': 10},
            'risk_level': 'LOW'
        }
        
        large_result = AnalysisResult(
            files=files,
            project_metrics=project_metrics,
            suggestions=["Suggestion 1", "Suggestion 2"],
            ai_insights=["Insight 1", "Insight 2"],
            risk_hotspots=[{
                'file': 'file_0.py',
                'function': 'function_0_0',
                'risk_score': 50.0,
                'complexity': 10,
                'lines': 50,
                'nesting': 3,
                'risk_type': 'HIGH_COMPLEXITY'
            }]
        )
        
        # Test that large result can be processed
        try:
            self.reporter.generate_cli_report(large_result, detailed=False)
        except Exception as e:
            pytest.fail(f"Large CLI report generation failed: {e}")
        
        # Test HTML generation
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            output_path = f.name
        
        try:
            html_path = self.reporter.generate_html_report(large_result, output_path)
            assert os.path.exists(html_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
