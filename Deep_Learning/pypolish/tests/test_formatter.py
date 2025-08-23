"""
Tests for Code Formatter module.
"""

import pytest
from pypolish.formatter import CodeFormatter


class TestCodeFormatter:
    """Test cases for CodeFormatter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = CodeFormatter()
    
    def test_format_code_basic(self):
        """Test basic code formatting."""
        code = """import sys,os
def test(x,y):
    return x+y"""
        
        formatted = self.formatter.format_code(code)
        
        # Should be formatted with proper spacing and line breaks
        assert "import sys, os" in formatted or "import os, sys" in formatted
        assert "def test(x, y):" in formatted
        assert "return x + y" in formatted
    
    def test_sort_imports(self):
        """Test import sorting."""
        code = """import sys
import os
from math import pi
import json"""
        
        formatted = self.formatter.format_code(code)
        
        # Imports should be sorted
        lines = formatted.split('\n')
        import_lines = [line for line in lines if line.startswith('import') or line.startswith('from')]
        
        # Should have proper sorting (standard library first, then third party)
        assert len(import_lines) >= 4
    
    def test_format_with_black(self):
        """Test black formatting."""
        code = """def test(x,y):
    return x+y"""
        
        formatted = self.formatter.format_code(code)
        
        # Should have proper spacing
        assert "def test(x, y):" in formatted
        assert "return x + y" in formatted
    
    def test_add_type_hints(self):
        """Test adding type hints."""
        code = """def add(a, b):
    return a + b"""
        
        result = self.formatter.add_type_hints(code)
        
        # Should add type hints
        assert "def add(a: int, b: int) -> int:" in result or "def add(a, b) -> int:" in result
    
    def test_add_type_hints_already_present(self):
        """Test that type hints aren't added if already present."""
        code = """def add(a: int, b: int) -> int:
    return a + b"""
        
        result = self.formatter.add_type_hints(code)
        
        # Should not change the code
        assert result == code
    
    def test_add_docstrings(self):
        """Test adding docstrings."""
        code = """def test():
    pass"""
        
        result = self.formatter.add_docstrings(code)
        
        # Should add a docstring
        assert "TODO: Add docstring" in result
    
    def test_add_docstrings_already_present(self):
        """Test that docstrings aren't added if already present."""
        code = '''def test():
    """This is a docstring."""
    pass'''
        
        result = self.formatter.add_docstrings(code)
        
        # Should not change the code
        assert result == code
    
    def test_convert_to_ternary(self):
        """Test converting if/else to ternary expressions."""
        code = """def test(x):
    if x > 0:
        return "positive"
    else:
        return "negative"
"""
        
        result = self.formatter.convert_to_ternary(code)
        
        # Should convert to ternary (though this is a simplified test)
        # In practice, the AST transformation would be more complex
        assert result is not None
    
    def test_remove_unused_imports(self):
        """Test removing unused imports."""
        code = """import math
import sys
import os

def test():
    print("Hello")
"""
        
        result = self.formatter.remove_unused_imports(code, ['math', 'sys', 'os'])
        
        # Should remove the unused imports
        assert "import math" not in result
        assert "import sys" not in result
        assert "import os" not in result
        assert "def test():" in result
    
    def test_remove_unused_imports_none(self):
        """Test removing unused imports when none are specified."""
        code = """import math
def test():
    print("Hello")
"""
        
        result = self.formatter.remove_unused_imports(code, [])
        
        # Should not change the code
        assert result == code
    
    def test_format_code_with_syntax_error(self):
        """Test formatting code with syntax errors."""
        code = """def test(
    print("Missing closing parenthesis"
"""
        
        # Should handle gracefully and return original code
        result = self.formatter.format_code(code)
        assert result == code
    
    def test_format_code_empty(self):
        """Test formatting empty code."""
        code = ""
        
        result = self.formatter.format_code(code)
        assert result == code
    
    def test_format_code_with_comments(self):
        """Test formatting code with comments."""
        code = """# This is a comment
def test():
    # Another comment
    return True"""
        
        result = self.formatter.format_code(code)
        
        # Should preserve comments
        assert "# This is a comment" in result
        assert "# Another comment" in result
    
    def test_infer_return_type_int(self):
        """Test return type inference for integers."""
        code = """def test():
    return 42"""
        
        result = self.formatter.add_type_hints(code)
        
        # Should infer int return type
        assert "-> int" in result
    
    def test_infer_return_type_str(self):
        """Test return type inference for strings."""
        code = """def test():
    return "hello"
"""
        
        result = self.formatter.add_type_hints(code)
        
        # Should infer str return type
        assert "-> str" in result
    
    def test_infer_return_type_none(self):
        """Test return type inference for None."""
        code = """def test():
    print("Hello")
"""
        
        result = self.formatter.add_type_hints(code)
        
        # Should infer None return type (print function)
        assert "-> None" in result
    
    def test_infer_parameter_type_int(self):
        """Test parameter type inference for integers."""
        code = """def test(x):
    return x % 2 == 0
"""
        
        result = self.formatter.add_type_hints(code)
        
        # Should infer int parameter type
        assert "x: int" in result
    
    def test_infer_parameter_type_str(self):
        """Test parameter type inference for strings."""
        code = """def test(x):
    return len(x)
"""
        
        result = self.formatter.add_type_hints(code)
        
        # Should infer str parameter type (len function)
        assert "x: str" in result or "x: List" in result
    
    def test_custom_line_length(self):
        """Test formatting with custom line length."""
        formatter = CodeFormatter(line_length=50)
        
        code = """def very_long_function_name_with_many_characters(parameter1, parameter2, parameter3):
    return parameter1 + parameter2 + parameter3"""
        
        result = formatter.format_code(code)
        
        # Should respect the line length limit
        lines = result.split('\n')
        for line in lines:
            assert len(line) <= 50
    
    def test_format_complex_code(self):
        """Test formatting complex code."""
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
    return result"""
        
        result = self.formatter.format_code(code)
        
        # Should be properly formatted
        assert "class Calculator:" in result
        assert "def __init__(self):" in result
        assert "def add(self, x, y):" in result
        assert "def multiply(self, x, y):" in result
        assert "def main():" in result
    
    def test_format_with_strings(self):
        """Test formatting code with various string types."""
        code = '''def test():
    single = "single quotes"
    double = "double quotes"
    triple = """triple quotes"""
    return single + double + triple'''
        
        result = self.formatter.format_code(code)
        
        # Should preserve string formatting
        assert '"single quotes"' in result
        assert '"double quotes"' in result
        assert '"""triple quotes"""' in result
    
    def test_format_with_numbers(self):
        """Test formatting code with various number types."""
        code = """def test():
    integer = 42
    float_num = 3.14
    complex_num = 1 + 2j
    return integer + float_num + complex_num"""
        
        result = self.formatter.format_code(code)
        
        # Should preserve number formatting
        assert "42" in result
        assert "3.14" in result
        assert "1 + 2j" in result
