"""
Tests for AST Analyzer module.
"""

import pytest
from pypolish.ast_analyzer import ASTAnalyzer


class TestASTAnalyzer:
    """Test cases for ASTAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ASTAnalyzer()
    
    def test_analyze_simple_function(self):
        """Test analyzing a simple function."""
        code = """
def hello():
    print("Hello, World!")
"""
        results = self.analyzer.analyze(code)
        
        assert 'error' not in results
        assert results['function_count'] == 1
        assert results['class_count'] == 0
        assert len(results['suggestions']) > 0  # Should suggest docstring
    
    def test_analyze_function_with_type_hints(self):
        """Test analyzing a function with type hints."""
        code = """
def add(a: int, b: int) -> int:
    return a + b
"""
        results = self.analyzer.analyze(code)
        
        assert 'error' not in results
        assert results['function_count'] == 1
        # Should not suggest type hints since they're already present
        type_hint_suggestions = [s for s in results['suggestions'] if s['type'] == 'missing_type_hint']
        assert len(type_hint_suggestions) == 0
    
    def test_analyze_function_with_docstring(self):
        """Test analyzing a function with docstring."""
        code = '''
def multiply(a, b):
    """Multiply two numbers."""
    return a * b
'''
        results = self.analyzer.analyze(code)
        
        assert 'error' not in results
        assert results['function_count'] == 1
        # Should not suggest docstring since it's already present
        docstring_suggestions = [s for s in results['suggestions'] if s['type'] == 'missing_docstring']
        assert len(docstring_suggestions) == 0
    
    def test_analyze_class(self):
        """Test analyzing a class."""
        code = """
class Calculator:
    def add(self, a, b):
        return a + b
"""
        results = self.analyzer.analyze(code)
        
        assert 'error' not in results
        assert results['class_count'] == 1
        assert results['function_count'] == 1
    
    def test_analyze_imports(self):
        """Test analyzing imports."""
        code = """
import math
import sys
from os import path

def test():
    print(math.pi)
"""
        results = self.analyzer.analyze(code)
        
        assert 'error' not in results
        assert 'math' in results['used_names']
        assert 'sys' not in results['used_names']  # sys is imported but not used
        assert 'path' in results['used_names']
    
    def test_analyze_unused_imports(self):
        """Test detecting unused imports."""
        code = """
import math
import sys
import os

def test():
    print("Hello")
"""
        results = self.analyzer.analyze(code)
        
        unused_imports = self.analyzer.get_unused_imports()
        assert 'math' in unused_imports
        assert 'sys' in unused_imports
        assert 'os' in unused_imports
    
    def test_analyze_undefined_names(self):
        """Test detecting undefined names."""
        code = """
def test():
    print(undefined_variable)
"""
        results = self.analyzer.analyze(code)
        
        undefined_names = self.analyzer.get_undefined_names()
        assert 'undefined_variable' in undefined_names
    
    def test_analyze_long_function(self):
        """Test detecting long functions."""
        code = """
def long_function():
    """ + "\n".join([f"    print({i})" for i in range(25)]) + """
"""
        results = self.analyzer.analyze(code)
        
        long_function_issues = [i for i in results['issues'] if i['type'] == 'long_function']
        assert len(long_function_issues) == 1
    
    def test_analyze_infinite_loop(self):
        """Test detecting infinite loops."""
        code = """
def test():
    while True:
        print("Infinite loop")
"""
        results = self.analyzer.analyze(code)
        
        infinite_loop_issues = [i for i in results['issues'] if i['type'] == 'infinite_loop']
        assert len(infinite_loop_issues) == 1
    
    def test_analyze_print_statements(self):
        """Test detecting print statements."""
        code = """
def test():
    print("Hello")
    print("World")
"""
        results = self.analyzer.analyze(code)
        
        print_suggestions = [s for s in results['suggestions'] if s['type'] == 'use_logging']
        assert len(print_suggestions) == 2
    
    def test_analyze_syntax_error(self):
        """Test handling syntax errors."""
        code = """
def test(
    print("Missing closing parenthesis"
"""
        results = self.analyzer.analyze(code)
        
        assert 'error' in results
        assert 'Syntax error' in results['error']
    
    def test_analyze_empty_code(self):
        """Test analyzing empty code."""
        code = ""
        results = self.analyzer.analyze(code)
        
        assert 'error' not in results
        assert results['function_count'] == 0
        assert results['class_count'] == 0
    
    def test_analyze_comments(self):
        """Test analyzing code with comments."""
        code = """
# This is a comment
def test():
    # Another comment
    print("Hello")
"""
        results = self.analyzer.analyze(code)
        
        assert 'error' not in results
        assert results['function_count'] == 1
    
    def test_analyze_multiple_functions(self):
        """Test analyzing multiple functions."""
        code = """
def func1():
    pass

def func2():
    pass

def func3():
    pass
"""
        results = self.analyzer.analyze(code)
        
        assert 'error' not in results
        assert results['function_count'] == 3
    
    def test_analyze_nested_functions(self):
        """Test analyzing nested functions."""
        code = """
def outer():
    def inner():
        pass
    return inner
"""
        results = self.analyzer.analyze(code)
        
        assert 'error' not in results
        assert results['function_count'] == 2  # Both outer and inner functions
    
    def test_analyze_ternary_opportunity(self):
        """Test detecting ternary expression opportunities."""
        code = """
def test(x):
    if x > 0:
        return "positive"
    else:
        return "negative"
"""
        results = self.analyzer.analyze(code)
        
        ternary_suggestions = [s for s in results['suggestions'] if s['type'] == 'ternary_expression']
        assert len(ternary_suggestions) > 0
    
    def test_analyze_list_comprehension_opportunity(self):
        """Test detecting list comprehension opportunities."""
        code = """
def test():
    result = []
    for i in range(10):
        result.append(i * 2)
    return result
"""
        results = self.analyzer.analyze(code)
        
        list_comp_suggestions = [s for s in results['suggestions'] if s['type'] == 'list_comprehension']
        assert len(list_comp_suggestions) > 0
    
    def test_reset_state(self):
        """Test that state is properly reset between analyses."""
        code1 = "def func1(): pass"
        code2 = "def func2(): pass"
        
        results1 = self.analyzer.analyze(code1)
        results2 = self.analyzer.analyze(code2)
        
        assert results1['function_count'] == 1
        assert results2['function_count'] == 1
        assert results1 != results2  # Should be different results
