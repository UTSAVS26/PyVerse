"""
Tests for Code Cleaner module.
"""

import pytest
import tempfile
import os
from pypolish.code_cleaner import CodeCleaner


class TestCodeCleaner:
    """Test cases for CodeCleaner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = CodeCleaner()
    
    def test_clean_code_basic(self):
        """Test basic code cleaning."""
        code = """import math,sys
def calc(x): 
  if x%2==0: print("Even") 
  else: print("Odd")
"""
        
        cleaned_code, analysis_results = self.cleaner.clean_code(code, show_analysis=False, show_diff=False)
        
        assert 'error' not in analysis_results
        assert cleaned_code != code  # Should be different after cleaning
        assert "def calc(" in cleaned_code
        assert "import math" in cleaned_code
    
    def test_clean_code_already_clean(self):
        """Test cleaning already clean code."""
        code = """import math

def calc(x: int) -> None:
    \"\"\"Calculate if number is even or odd.\"\"\"
    print("Even" if x % 2 == 0 else "Odd")
"""
        
        cleaned_code, analysis_results = self.cleaner.clean_code(code, show_analysis=False, show_diff=False)
        
        assert 'error' not in analysis_results
        # Should be very similar (might have minor formatting differences)
        assert "def calc" in cleaned_code
        assert "import math" in cleaned_code
    
    def test_clean_code_with_syntax_error(self):
        """Test cleaning code with syntax errors."""
        code = """def test(
    print("Missing closing parenthesis"
"""
        
        cleaned_code, analysis_results = self.cleaner.clean_code(code, show_analysis=False, show_diff=False)
        
        assert 'error' in analysis_results
        assert cleaned_code == code  # Should return original code
    
    def test_clean_file(self):
        """Test cleaning a file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""import math,sys
def calc(x): 
  if x%2==0: print("Even") 
  else: print("Odd")
""")
            temp_file = f.name
        
        try:
            cleaned_code, analysis_results = self.cleaner.clean_file(temp_file, show_analysis=False, show_diff=False)
            
            assert 'error' not in analysis_results
            assert cleaned_code != ""
            assert "def calc(" in cleaned_code
        finally:
            os.unlink(temp_file)
    
    def test_clean_file_with_output(self):
        """Test cleaning a file and saving to output."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""import math,sys
def calc(x): 
  if x%2==0: print("Even") 
  else: print("Odd")
""")
            temp_file = f.name
        
        output_file = temp_file + "_cleaned.py"
        
        try:
            cleaned_code, analysis_results = self.cleaner.clean_file(
                temp_file, 
                output_path=output_file,
                show_analysis=False, 
                show_diff=False
            )
            
            assert 'error' not in analysis_results
            assert os.path.exists(output_file)
            
            # Check that output file contains cleaned code
            with open(output_file, 'r') as f:
                saved_code = f.read()
            assert saved_code == cleaned_code
        finally:
            os.unlink(temp_file)
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_clean_file_not_found(self):
        """Test cleaning a non-existent file."""
        cleaned_code, analysis_results = self.cleaner.clean_file("nonexistent.py", show_analysis=False, show_diff=False)
        
        assert 'error' in analysis_results
        assert "not found" in analysis_results['error']
        assert cleaned_code == ""
    
    def test_get_cleaning_report(self):
        """Test generating cleaning report."""
        original_code = """import math,sys
def calc(x): 
  if x%2==0: print("Even") 
  else: print("Odd")
"""
        
        cleaned_code, analysis_results = self.cleaner.clean_code(original_code, show_analysis=False, show_diff=False)
        
        report = self.cleaner.get_cleaning_report(original_code, cleaned_code, analysis_results)
        
        assert 'original_lines' in report
        assert 'cleaned_lines' in report
        assert 'original_chars' in report
        assert 'cleaned_chars' in report
        assert 'line_difference' in report
        assert 'char_difference' in report
        assert 'issues_found' in report
        assert 'suggestions_made' in report
        assert 'improvement_percentage' in report
        assert 'has_changes' in report
        assert report['has_changes'] == True
    
    def test_validate_code_valid(self):
        """Test validating valid code."""
        code = """def test():
    return True
"""
        
        assert self.cleaner.validate_code(code) == True
    
    def test_validate_code_invalid(self):
        """Test validating invalid code."""
        code = """def test(
    return True
"""
        
        assert self.cleaner.validate_code(code) == False
    
    def test_get_code_metrics(self):
        """Test getting code metrics."""
        code = """import math
import sys

def test():
    return 42

class Calculator:
    def add(self, a, b):
        return a + b
"""
        
        metrics = self.cleaner.get_code_metrics(code)
        
        assert 'error' not in metrics
        assert metrics['function_count'] == 2  # test and add
        assert metrics['class_count'] == 1
        assert metrics['total_lines'] >= 10
        assert metrics['total_chars'] > 0
        assert metrics['unused_imports'] >= 1  # sys is unused
    
    def test_get_code_metrics_with_error(self):
        """Test getting code metrics with syntax error."""
        code = """def test(
    return True
"""
        
        metrics = self.cleaner.get_code_metrics(code)
        
        assert 'error' in metrics
    
    def test_apply_improvements(self):
        """Test applying improvements to code."""
        code = """import math,sys
def calc(x): 
  if x%2==0: print("Even") 
  else: print("Odd")
"""
        
        analysis_results = self.cleaner.analyzer.analyze(code)
        improved_code = self.cleaner._apply_improvements(code, analysis_results)
        
        # Should have some improvements
        assert improved_code != code
        assert "def calc(" in improved_code
    
    def test_apply_specific_improvements(self):
        """Test applying specific improvements."""
        code = """def test():
    return True
"""
        
        analysis_results = {'issues': [], 'suggestions': []}
        improved_code = self.cleaner._apply_specific_improvements(code, analysis_results)
        
        # Should return the code as-is (simplified implementation)
        assert improved_code == code
    
    def test_clean_code_with_analysis_display(self):
        """Test cleaning code with analysis display."""
        code = """import math,sys
def calc(x): 
  if x%2==0: print("Even") 
  else: print("Odd")
"""
        
        # This should not raise any exceptions
        cleaned_code, analysis_results = self.cleaner.clean_code(code, show_analysis=True, show_diff=False)
        
        assert 'error' not in analysis_results
        assert cleaned_code != code
    
    def test_clean_code_with_diff_display(self):
        """Test cleaning code with diff display."""
        code = """import math,sys
def calc(x): 
  if x%2==0: print("Even") 
  else: print("Odd")
"""
        
        # This should not raise any exceptions
        cleaned_code, analysis_results = self.cleaner.clean_code(code, show_analysis=False, show_diff=True)
        
        assert 'error' not in analysis_results
        assert cleaned_code != code
    
    def test_clean_code_empty(self):
        """Test cleaning empty code."""
        code = ""
        
        cleaned_code, analysis_results = self.cleaner.clean_code(code, show_analysis=False, show_diff=False)
        
        assert 'error' not in analysis_results
        assert cleaned_code == code
    
    def test_clean_code_with_comments(self):
        """Test cleaning code with comments."""
        code = """# This is a comment
import math,sys
def calc(x): 
  if x%2==0: print("Even")  # Even number
  else: print("Odd")  # Odd number
"""
        
        cleaned_code, analysis_results = self.cleaner.clean_code(code, show_analysis=False, show_diff=False)
        
        assert 'error' not in analysis_results
        assert "# This is a comment" in cleaned_code
        assert "# Even number" in cleaned_code
        assert "# Odd number" in cleaned_code
    
    def test_clean_code_complex(self):
        """Test cleaning complex code."""
        code = """import sys,os
from math import pi,sqrt
import json

class Calculator:
    def __init__(self):
        self.value=0
    
    def add(self,x,y):
        return x+y
    
    def multiply(self,x,y):
        return x*y

def main():
    calc=Calculator()
    result=calc.add(5,3)
    print(result)
    return result
"""
        
        cleaned_code, analysis_results = self.cleaner.clean_code(code, show_analysis=False, show_diff=False)
        
        assert 'error' not in analysis_results
        assert "class Calculator:" in cleaned_code
        assert "def __init__(self):" in cleaned_code
        assert "def add(self, x, y):" in cleaned_code
        assert "def multiply(self, x, y):" in cleaned_code
        assert "def main():" in cleaned_code
    
    def test_custom_line_length(self):
        """Test cleaning with custom line length."""
        cleaner = CodeCleaner(line_length=50)
        
        code = """def very_long_function_name_with_many_characters(parameter1, parameter2, parameter3):
    return parameter1 + parameter2 + parameter3"""
        
        cleaned_code, analysis_results = cleaner.clean_code(code, show_analysis=False, show_diff=False)
        
        assert 'error' not in analysis_results
        # Should respect the line length limit
        lines = cleaned_code.split('\n')
        for line in lines:
            assert len(line) <= 50
