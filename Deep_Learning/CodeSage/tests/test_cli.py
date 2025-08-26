"""
Tests for the CLI module.
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import argparse

from codesage.cli import main, create_argument_parser, display_welcome_message, run_analysis


class TestCLI:
    """Test cases for CLI functionality."""
    
    def test_create_argument_parser(self):
        """Test argument parser creation and configuration."""
        parser = create_argument_parser()
        
        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.prog == "codesage"
        assert "AI-Enhanced Code Complexity Estimator" in parser.description
    
    def test_argument_parser_help(self):
        """Test that help argument works correctly."""
        parser = create_argument_parser()
        
        # Test that help can be parsed without error
        try:
            args = parser.parse_args(['--help'])
            # Should not reach here as help exits
            pytest.fail("Help should exit")
        except SystemExit:
            # Expected behavior
            pass
    
    def test_argument_parser_version(self):
        """Test that version argument works correctly."""
        parser = create_argument_parser()
        
        # Test that version can be parsed without error
        try:
            args = parser.parse_args(['--version'])
            # Should not reach here as version exits
            pytest.fail("Version should exit")
        except SystemExit:
            # Expected behavior
            pass
    
    def test_argument_parser_basic_args(self):
        """Test basic argument parsing."""
        parser = create_argument_parser()
        
        # Test minimal required arguments
        args = parser.parse_args(['test_path.py'])
        assert args.path == 'test_path.py'
        assert args.html == False
        assert args.html_output == 'codesage_report.html'
        assert args.detailed == False
        assert args.no_ml == False
        assert args.quiet == False
        assert args.verbose == False
        assert args.strict == False
    
    def test_argument_parser_all_options(self):
        """Test all optional arguments."""
        parser = create_argument_parser()
        
        args = parser.parse_args([
            'test_path.py',
            '--html',
            '--html-output', 'custom_report.html',
            '--detailed',
            '--no-ml',
            '--quiet',
            '--verbose',
            '--strict'
        ])
        
        assert args.path == 'test_path.py'
        assert args.html == True
        assert args.html_output == 'custom_report.html'
        assert args.detailed == True
        assert args.no_ml == True
        assert args.quiet == True
        assert args.verbose == True
        assert args.strict == True
    
    def test_display_welcome_message(self):
        """Test welcome message display."""
        with patch('codesage.cli.Console') as mock_console:
            mock_console_instance = MagicMock()
            mock_console.return_value = mock_console_instance
            
            display_welcome_message(mock_console_instance)
            
            # Verify that print was called
            mock_console_instance.print.assert_called()
    
    def test_run_analysis_success(self):
        """Test successful analysis execution."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def simple_function():
    return 42

def complex_function(x, y, z):
    result = 0
    if x > 0:
        if y > 0:
            for i in range(x):
                if i % 2 == 0:
                    result += i
    return result
""")
            test_file = f.name
        
        try:
            result = run_analysis(test_file, quiet=True)
            
            assert isinstance(result, dict)
            assert 'files' in result
            assert 'project_metrics' in result
            assert 'suggestions' in result
            assert 'ai_insights' in result
            assert 'risk_hotspots' in result
            
            # Check that analysis was performed
            assert len(result['files']) == 1
            assert result['project_metrics']['total_files'] == 1
            assert result['project_metrics']['total_functions'] == 2
            
        finally:
            os.unlink(test_file)
    
    def test_run_analysis_nonexistent_path(self):
        """Test analysis with nonexistent path."""
        with pytest.raises(ValueError, match="Path.*does not exist"):
            run_analysis("nonexistent_file.py", quiet=True)
    
    def test_run_analysis_with_html_output(self):
        """Test analysis with HTML output."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def test_function():
    return "Hello, World!"
""")
            test_file = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as html_f:
            html_output = html_f.name
        
        try:
            result = run_analysis(test_file, html_output=html_output, quiet=True)
            
            assert isinstance(result, dict)
            assert os.path.exists(html_output)
            
            # Check HTML file content
            with open(html_output, 'r', encoding='utf-8') as f:
                html_content = f.read()
                assert 'CodeSage Analysis Report' in html_content
            
        finally:
            os.unlink(test_file)
            if os.path.exists(html_output):
                os.unlink(html_output)
    
    def test_run_analysis_detailed(self):
        """Test detailed analysis."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def function1():
    return 1

def function2(x):
    if x > 0:
        return x * 2
    return 0
""")
            test_file = f.name
        
        try:
            result = run_analysis(test_file, detailed=True, quiet=True)
            
            assert isinstance(result, dict)
            assert len(result['files']) == 1
            assert result['project_metrics']['total_functions'] == 2
            
        finally:
            os.unlink(test_file)
    
    def test_run_analysis_no_ml(self):
        """Test analysis without ML enhancements."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def test_function():
    return 42
""")
            test_file = f.name
        
        try:
            result = run_analysis(test_file, train_ml=False, quiet=True)
            
            assert isinstance(result, dict)
            # Should still work without ML
            assert 'files' in result
            assert 'project_metrics' in result
            
        finally:
            os.unlink(test_file)
    
    @patch('codesage.cli.CodeAnalyzer')
    @patch('codesage.cli.ReportGenerator')
    @patch('codesage.cli.Console')
    @patch('codesage.cli.Progress')
    def test_main_success(self, mock_progress, mock_console, mock_reporter, mock_analyzer):
        """Test successful main execution."""
        # Mock components
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        
        mock_analyzer_instance = MagicMock()
        mock_analyzer.return_value = mock_analyzer_instance
        
        mock_reporter_instance = MagicMock()
        mock_reporter.return_value = mock_reporter_instance
        
        mock_progress_instance = MagicMock()
        mock_progress.return_value.__enter__.return_value = mock_progress_instance
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def test_function():
    return 42
""")
            test_file = f.name
        
        try:
            # Mock analysis result
            mock_result = MagicMock()
            mock_result.project_metrics = {'risk_level': 'LOW'}
            mock_analyzer_instance.analyze_project.return_value = mock_result
            
            # Test main function
            with patch.object(sys, 'argv', ['codesage', test_file]):
                main()
            
            # Verify components were called
            mock_analyzer_instance.analyze_project.assert_called_once()
            mock_reporter_instance.generate_cli_report.assert_called_once()
            
        finally:
            os.unlink(test_file)
    
    @patch('codesage.cli.Console')
    def test_main_nonexistent_path(self, mock_console):
        """Test main with nonexistent path."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        
        with patch.object(sys, 'argv', ['codesage', 'nonexistent_file.py']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            assert exc_info.value.code == 1
    
    @patch('codesage.cli.CodeAnalyzer')
    @patch('codesage.cli.ReportGenerator')
    @patch('codesage.cli.Console')
    @patch('codesage.cli.Progress')
    def test_main_with_html_output(self, mock_progress, mock_console, mock_reporter, mock_analyzer):
        """Test main with HTML output."""
        # Mock components
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        
        mock_analyzer_instance = MagicMock()
        mock_analyzer.return_value = mock_analyzer_instance
        
        mock_reporter_instance = MagicMock()
        mock_reporter.return_value = mock_reporter_instance
        
        mock_progress_instance = MagicMock()
        mock_progress.return_value.__enter__.return_value = mock_progress_instance
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def test_function():
    return 42
""")
            test_file = f.name
        
        try:
            # Mock analysis result
            mock_result = MagicMock()
            mock_result.project_metrics = {'risk_level': 'LOW'}
            mock_analyzer_instance.analyze_project.return_value = mock_result
            
            # Test main function with HTML output
            with patch.object(sys, 'argv', ['codesage', test_file, '--html']):
                main()
            
            # Verify HTML report was generated
            mock_reporter_instance.generate_html_report.assert_called_once()
            
        finally:
            os.unlink(test_file)
    
    @patch('codesage.cli.CodeAnalyzer')
    @patch('codesage.cli.ReportGenerator')
    @patch('codesage.cli.Console')
    @patch('codesage.cli.Progress')
    def test_main_strict_mode_high_risk(self, mock_progress, mock_console, mock_reporter, mock_analyzer):
        """Test main in strict mode with high risk."""
        # Mock components
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        
        mock_analyzer_instance = MagicMock()
        mock_analyzer.return_value = mock_analyzer_instance
        
        mock_reporter_instance = MagicMock()
        mock_reporter.return_value = mock_reporter_instance
        
        mock_progress_instance = MagicMock()
        mock_progress.return_value.__enter__.return_value = mock_progress_instance
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def test_function():
    return 42
""")
            test_file = f.name
        
        try:
            # Mock analysis result with high risk
            mock_result = MagicMock()
            mock_result.project_metrics = {'risk_level': 'HIGH'}
            mock_analyzer_instance.analyze_project.return_value = mock_result
            
            # Test main function in strict mode
            with patch.object(sys, 'argv', ['codesage', test_file, '--strict']):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                
                assert exc_info.value.code == 1
            
        finally:
            os.unlink(test_file)
    
    @patch('codesage.cli.CodeAnalyzer')
    @patch('codesage.cli.ReportGenerator')
    @patch('codesage.cli.Console')
    @patch('codesage.cli.Progress')
    def test_main_analysis_error(self, mock_progress, mock_console, mock_reporter, mock_analyzer):
        """Test main with analysis error."""
        # Mock components
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        
        mock_analyzer_instance = MagicMock()
        mock_analyzer.return_value = mock_analyzer_instance
        
        mock_progress_instance = MagicMock()
        mock_progress.return_value.__enter__.return_value = mock_progress_instance
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def test_function():
    return 42
""")
            test_file = f.name
        
        try:
            # Mock analysis error
            mock_analyzer_instance.analyze_project.side_effect = Exception("Analysis failed")
            
            # Test main function
            with patch.object(sys, 'argv', ['codesage', test_file]):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                
                assert exc_info.value.code == 1
            
        finally:
            os.unlink(test_file)
    
    @patch('codesage.cli.Console')
    def test_main_keyboard_interrupt(self, mock_console):
        """Test main with keyboard interrupt."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        
        with patch.object(sys, 'argv', ['codesage', 'test.py']):
            with patch('codesage.cli.Path') as mock_path:
                mock_path.return_value.exists.return_value = True
                
                with patch('codesage.cli.CodeAnalyzer') as mock_analyzer:
                    mock_analyzer_instance = MagicMock()
                    mock_analyzer.return_value = mock_analyzer_instance
                    mock_analyzer_instance.analyze_project.side_effect = KeyboardInterrupt()
                    
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    
                    assert exc_info.value.code == 130
    
    def test_run_analysis_with_output(self):
        """Test run_analysis with output display."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def test_function():
    return 42
""")
            test_file = f.name
        
        try:
            # Test with output (not quiet)
            with patch('codesage.cli.Console') as mock_console:
                mock_console_instance = MagicMock()
                mock_console.return_value = mock_console_instance
                
                with patch('codesage.cli.Progress') as mock_progress:
                    mock_progress_instance = MagicMock()
                    mock_progress.return_value.__enter__.return_value = mock_progress_instance
                    
                    result = run_analysis(test_file, quiet=False)
                    
                    assert isinstance(result, dict)
                    # Verify progress was shown
                    mock_progress_instance.add_task.assert_called()
                    
        finally:
            os.unlink(test_file)
    
    def test_run_analysis_error_handling(self):
        """Test run_analysis error handling."""
        with pytest.raises(ValueError, match="Path.*does not exist"):
            run_analysis("nonexistent_file.py", quiet=True)
        
        # Test with quiet=False to ensure error is printed
        with patch('codesage.cli.Console') as mock_console:
            mock_console_instance = MagicMock()
            mock_console.return_value = mock_console_instance
            
            with pytest.raises(ValueError):
                run_analysis("nonexistent_file.py", quiet=False)
            
            # Verify error was printed
            mock_console_instance.print.assert_called()
    
    def test_argument_parser_edge_cases(self):
        """Test argument parser edge cases."""
        parser = create_argument_parser()
        
        # Test with just the required argument
        args = parser.parse_args(['.'])
        assert args.path == '.'
        
        # Test with relative path
        args = parser.parse_args(['./test.py'])
        assert args.path == './test.py'
        
        # Test with absolute path
        args = parser.parse_args(['/absolute/path/test.py'])
        assert args.path == '/absolute/path/test.py'
    
    def test_argument_parser_invalid_args(self):
        """Test argument parser with invalid arguments."""
        parser = create_argument_parser()
        
        # Test missing required argument
        with pytest.raises(SystemExit):
            parser.parse_args([])
        
        # Test invalid option
        with pytest.raises(SystemExit):
            parser.parse_args(['test.py', '--invalid-option'])
    
    def test_display_welcome_message_content(self):
        """Test welcome message content."""
        with patch('codesage.cli.Console') as mock_console:
            mock_console_instance = MagicMock()
            mock_console.return_value = mock_console_instance
            
            display_welcome_message(mock_console_instance)
            
            # Verify the welcome message was printed
            mock_console_instance.print.assert_called()
            
            # Get the call arguments to check content
            if mock_console_instance.print.call_args and len(mock_console_instance.print.call_args[0]) > 0:
                call_args = mock_console_instance.print.call_args[0][0]
                assert "CodeSage" in str(call_args)
                assert "AI-Enhanced" in str(call_args)
