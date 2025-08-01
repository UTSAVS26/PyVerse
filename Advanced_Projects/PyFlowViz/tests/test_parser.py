"""
Tests for parser modules.
"""

import pytest
import tempfile
import os
from parser import ASTParser, BytecodeParser


class TestASTParser:
    """Test AST parser functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = ASTParser()
    
    def test_parse_simple_function(self):
        """Test parsing a simple function."""
        code = """
def hello():
    print("Hello, World!")
"""
        result = self.parser.parse_string(code)
        
        assert 'nodes' in result
        assert 'edges' in result
        assert 'source_code' in result
        assert len(result['nodes']) > 0
        assert len(result['edges']) > 0
    
    def test_parse_if_statement(self):
        """Test parsing if-elif-else statements."""
        code = """
def check_number(x):
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"
"""
        result = self.parser.parse_string(code)
        
        # Should have condition nodes
        condition_nodes = [n for n in result['nodes'] if n.node_type == 'condition']
        assert len(condition_nodes) > 0
    
    def test_parse_for_loop(self):
        """Test parsing for loops."""
        code = """
def sum_list(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""
        result = self.parser.parse_string(code)
        
        # Should have loop nodes
        loop_nodes = [n for n in result['nodes'] if n.node_type == 'loop']
        assert len(loop_nodes) > 0
    
    def test_parse_while_loop(self):
        """Test parsing while loops."""
        code = """
def countdown(n):
    while n > 0:
        print(n)
        n -= 1
    print("Done!")
"""
        result = self.parser.parse_string(code)
        
        # Should have loop nodes
        loop_nodes = [n for n in result['nodes'] if n.node_type == 'loop']
        assert len(loop_nodes) > 0
    
    def test_parse_try_except(self):
        """Test parsing try-except blocks."""
        code = """
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None
    finally:
        print("Operation completed")
"""
        result = self.parser.parse_string(code)
        
        # Should have try, except, and finally nodes
        try_nodes = [n for n in result['nodes'] if n.node_type == 'try']
        except_nodes = [n for n in result['nodes'] if n.node_type == 'except']
        finally_nodes = [n for n in result['nodes'] if n.node_type == 'finally']
        
        assert len(try_nodes) > 0
        assert len(except_nodes) > 0
        assert len(finally_nodes) > 0
    
    def test_parse_function_calls(self):
        """Test parsing function calls."""
        code = """
def process_data():
    result = calculate_value(10)
    display_result(result)
"""
        result = self.parser.parse_string(code)
        
        # Should have function call nodes
        call_nodes = [n for n in result['nodes'] if n.node_type == 'function_call']
        assert len(call_nodes) > 0
    
    def test_parse_file(self):
        """Test parsing from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def test_function():
    return "Hello"
""")
            temp_file = f.name
        
        try:
            result = self.parser.parse_file(temp_file)
            assert 'nodes' in result
            assert 'edges' in result
            assert 'source_code' in result
        finally:
            os.unlink(temp_file)
    
    def test_invalid_syntax(self):
        """Test handling of invalid syntax."""
        invalid_code = """
def invalid_function(
    print("Missing closing parenthesis"
"""
        
        with pytest.raises(ValueError):
            self.parser.parse_string(invalid_code)
    
    def test_empty_code(self):
        """Test parsing empty code."""
        result = self.parser.parse_string("")
        assert 'nodes' in result
        assert 'edges' in result
        assert 'source_code' in result


class TestBytecodeParser:
    """Test Bytecode parser functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = BytecodeParser()
    
    def test_parse_simple_function(self):
        """Test parsing a simple function with bytecode."""
        code = """
def simple_function():
    return 42
"""
        result = self.parser.parse_string(code)
        
        assert 'nodes' in result
        assert 'edges' in result
        assert 'source_code' in result
        assert len(result['nodes']) > 0
        assert len(result['edges']) > 0
    
    def test_parse_arithmetic(self):
        """Test parsing arithmetic operations."""
        code = """
def add_numbers(a, b):
    result = a + b
    return result
"""
        result = self.parser.parse_string(code)
        
        # Should have bytecode block nodes
        block_nodes = [n for n in result['nodes'] if n.node_type == 'bytecode_block']
        assert len(block_nodes) > 0
    
    def test_parse_conditionals(self):
        """Test parsing conditional statements."""
        code = """
def check_value(x):
    if x > 0:
        return "positive"
    else:
        return "negative"
"""
        result = self.parser.parse_string(code)
        
        # Should have nodes
        assert len(result['nodes']) > 0
        assert len(result['edges']) > 0
    
    def test_parse_file(self):
        """Test parsing from file with bytecode."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def bytecode_test():
    x = 1 + 2
    return x
""")
            temp_file = f.name
        
        try:
            result = self.parser.parse_file(temp_file)
            assert 'nodes' in result
            assert 'edges' in result
            assert 'source_code' in result
        finally:
            os.unlink(temp_file)
    
    def test_invalid_syntax(self):
        """Test handling of invalid syntax in bytecode parser."""
        invalid_code = """
def invalid_function(
    print("Missing closing parenthesis"
"""
        
        with pytest.raises(ValueError):
            self.parser.parse_string(invalid_code)
    
    def test_empty_code(self):
        """Test parsing empty code with bytecode parser."""
        result = self.parser.parse_string("")
        assert 'nodes' in result
        assert 'edges' in result
        assert 'source_code' in result


class TestParserComparison:
    """Test comparison between AST and Bytecode parsers."""
    
    def test_same_code_different_parsers(self):
        """Test that both parsers can handle the same code."""
        code = """
def compare_parsers():
    x = 10
    if x > 5:
        return "greater"
    else:
        return "less"
"""
        
        ast_parser = ASTParser()
        bytecode_parser = BytecodeParser()
        
        ast_result = ast_parser.parse_string(code)
        bytecode_result = bytecode_parser.parse_string(code)
        
        # Both should succeed
        assert 'nodes' in ast_result
        assert 'edges' in ast_result
        assert 'nodes' in bytecode_result
        assert 'edges' in bytecode_result
        
        # Both should have some nodes
        assert len(ast_result['nodes']) > 0
        assert len(bytecode_result['nodes']) > 0 