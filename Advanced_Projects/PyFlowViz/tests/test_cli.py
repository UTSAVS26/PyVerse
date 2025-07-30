"""
Tests for CLI functionality.
"""

import pytest
import tempfile
import os
import subprocess
import sys
from pathlib import Path


class TestCLI:
    """Test CLI functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.example_file = os.path.join(self.test_dir, "test_example.py")
        
        # Create a test Python file
        with open(self.example_file, 'w') as f:
            f.write("""
def test_function():
    if x > 0:
        return "positive"
    else:
        return "negative"
""")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_cli_help(self):
        """Test CLI help output."""
        result = subprocess.run(
            [sys.executable, "pyflowviz.py", "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "PyFlowViz" in result.stdout
        assert "Code-to-Flowchart" in result.stdout
    
    def test_cli_svg_generation(self):
        """Test CLI SVG generation."""
        output_file = os.path.join(self.test_dir, "output.svg")
        
        result = subprocess.run(
            [sys.executable, "pyflowviz.py", self.example_file, "--output", output_file],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert os.path.exists(output_file)
        assert "Generated SVG" in result.stdout
    
    def test_cli_png_generation(self):
        """Test CLI PNG generation."""
        output_file = os.path.join(self.test_dir, "output.png")
        
        result = subprocess.run(
            [sys.executable, "pyflowviz.py", self.example_file, "--output", output_file],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert os.path.exists(output_file)
        assert "Generated PNG" in result.stdout
    
    def test_cli_html_generation(self):
        """Test CLI HTML generation."""
        output_file = os.path.join(self.test_dir, "output.html")
        
        result = subprocess.run(
            [sys.executable, "pyflowviz.py", self.example_file, "--html", "--output", output_file],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert os.path.exists(output_file)
        assert "Generated Interactive HTML" in result.stdout
    
    def test_cli_mermaid_generation(self):
        """Test CLI Mermaid HTML generation."""
        output_file = os.path.join(self.test_dir, "output.html")
        
        result = subprocess.run(
            [sys.executable, "pyflowviz.py", self.example_file, "--mermaid", "--output", output_file],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert os.path.exists(output_file)
        assert "Generated Mermaid HTML" in result.stdout
    
    def test_cli_bytecode_parser(self):
        """Test CLI with bytecode parser."""
        output_file = os.path.join(self.test_dir, "output.svg")
        
        result = subprocess.run(
            [sys.executable, "pyflowviz.py", self.example_file, "--parser", "bytecode", "--output", output_file],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert os.path.exists(output_file)
    
    def test_cli_verbose_output(self):
        """Test CLI verbose output."""
        output_file = os.path.join(self.test_dir, "output.svg")
        
        result = subprocess.run(
            [sys.executable, "pyflowviz.py", self.example_file, "--verbose", "--output", output_file],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Parsing file" in result.stdout
        assert "Parsed successfully" in result.stdout
    
    def test_cli_nonexistent_file(self):
        """Test CLI with nonexistent file."""
        result = subprocess.run(
            [sys.executable, "pyflowviz.py", "nonexistent.py"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0  # Our CLI returns 0 but prints error to stdout
        assert "not found" in result.stdout
    
    def test_cli_invalid_syntax(self):
        """Test CLI with invalid Python syntax."""
        invalid_file = os.path.join(self.test_dir, "invalid.py")
        
        with open(invalid_file, 'w') as f:
            f.write("""
def invalid_function(
    print("Missing closing parenthesis"
""")
        
        result = subprocess.run(
            [sys.executable, "pyflowviz.py", invalid_file],
            capture_output=True,
            text=True
        )
        
        assert result.returncode != 0
        assert "Error" in result.stdout  # Error is printed to stdout, not stderr
    
    def test_cli_batch_processing(self):
        """Test CLI batch processing."""
        # Create multiple test files
        for i in range(3):
            test_file = os.path.join(self.test_dir, f"test_{i}.py")
            with open(test_file, 'w') as f:
                f.write(f"""
def test_function_{i}():
    return {i}
""")
        
        output_dir = os.path.join(self.test_dir, "output")
        
        result = subprocess.run(
            [sys.executable, "pyflowviz.py", "--batch", self.test_dir, "--output", output_dir],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Found 4 Python files" in result.stdout  # 3 created + 1 original
        assert "Successfully processed" in result.stdout
        
        # Check that output files were created
        assert os.path.exists(output_dir)
        output_files = list(Path(output_dir).glob("*.svg"))
        assert len(output_files) > 0
    
    def test_cli_no_arguments(self):
        """Test CLI with no arguments (should show help)."""
        result = subprocess.run(
            [sys.executable, "pyflowviz.py"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()


class TestCLIIntegration:
    """Test CLI integration with different file types."""
    
    def test_cli_with_complex_code(self):
        """Test CLI with complex Python code."""
        with tempfile.TemporaryDirectory() as temp_dir:
            complex_file = os.path.join(temp_dir, "complex.py")
            
            with open(complex_file, 'w') as f:
                f.write("""
def complex_function():
    try:
        for i in range(10):
            if i % 2 == 0:
                print("even")
            else:
                print("odd")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Done")
""")
            
            output_file = os.path.join(temp_dir, "output.svg")
            
            result = subprocess.run(
                [sys.executable, "pyflowviz.py", complex_file, "--output", output_file],
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0
            assert os.path.exists(output_file)
    
    def test_cli_with_function_calls(self):
        """Test CLI with function calls."""
        with tempfile.TemporaryDirectory() as temp_dir:
            func_file = os.path.join(temp_dir, "functions.py")
            
            with open(func_file, 'w') as f:
                f.write("""
def helper_function(x):
    return x * 2

def main_function():
    result = helper_function(5)
    display_result(result)
    return result
""")
            
            output_file = os.path.join(temp_dir, "output.svg")
            
            result = subprocess.run(
                [sys.executable, "pyflowviz.py", func_file, "--output", output_file],
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0
            assert os.path.exists(output_file)
    
    def test_cli_output_formats(self):
        """Test CLI with different output formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.py")
            
            with open(test_file, 'w') as f:
                f.write("""
def test_function():
    return "Hello"
""")
            
            # Test SVG
            svg_output = os.path.join(temp_dir, "output.svg")
            result = subprocess.run(
                [sys.executable, "pyflowviz.py", test_file, "--output", svg_output],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0
            assert os.path.exists(svg_output)
            
            # Test PNG
            png_output = os.path.join(temp_dir, "output.png")
            result = subprocess.run(
                [sys.executable, "pyflowviz.py", test_file, "--output", png_output],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0
            assert os.path.exists(png_output)
            
            # Test HTML
            html_output = os.path.join(temp_dir, "output.html")
            result = subprocess.run(
                [sys.executable, "pyflowviz.py", test_file, "--html", "--output", html_output],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0
            assert os.path.exists(html_output) 