"""
Tests for visualizer modules.
"""

import pytest
import tempfile
import os
from visualizer import GraphvizGenerator, HTMLRenderer
from parser import ASTParser


class TestGraphvizGenerator:
    """Test Graphviz generator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = GraphvizGenerator()
        self.parser = ASTParser()
    
    def test_generate_svg(self):
        """Test SVG generation."""
        code = """
def test_function():
    if x > 0:
        return "positive"
    else:
        return "negative"
"""
        flowchart_data = self.parser.parse_string(code)
        
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as f:
            temp_file = f.name
        
        try:
            result_path = self.generator.generate_svg(flowchart_data, temp_file)
            assert os.path.exists(result_path)
            assert result_path.endswith('.svg')
            
            # Check file content
            with open(result_path, 'r') as f:
                content = f.read()
                assert '<?xml' in content  # SVG should start with XML declaration
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_generate_png(self):
        """Test PNG generation."""
        code = """
def simple_function():
    return 42
"""
        flowchart_data = self.parser.parse_string(code)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_file = f.name
        
        try:
            result_path = self.generator.generate_png(flowchart_data, temp_file)
            assert os.path.exists(result_path)
            assert result_path.endswith('.png')
            
            # Check file size (should be non-zero)
            assert os.path.getsize(result_path) > 0
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_generate_dot(self):
        """Test DOT file generation."""
        code = """
def test_function():
    x = 1 + 2
    return x
"""
        flowchart_data = self.parser.parse_string(code)
        
        with tempfile.NamedTemporaryFile(suffix='.dot', delete=False) as f:
            temp_file = f.name
        
        try:
            result_path = self.generator.generate_dot(flowchart_data, temp_file)
            assert os.path.exists(result_path)
            assert result_path.endswith('.dot')
            
            # Check file content
            with open(result_path, 'r') as f:
                content = f.read()
                assert 'digraph' in content  # DOT files should contain digraph
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_node_colors(self):
        """Test that node colors are properly assigned."""
        code = """
def test_function():
    if x > 0:
        return "positive"
    else:
        return "negative"
"""
        flowchart_data = self.parser.parse_string(code)
        
        # Check that colors are defined for all node types
        node_types = set(node.node_type for node in flowchart_data['nodes'])
        for node_type in node_types:
            assert node_type in self.generator.node_colors
    
    def test_edge_colors(self):
        """Test that edge colors are properly assigned."""
        code = """
def test_function():
    if x > 0:
        return "positive"
    else:
        return "negative"
"""
        flowchart_data = self.parser.parse_string(code)
        
        # Check that colors are defined for edge labels
        edge_labels = set(edge.label for edge in flowchart_data['edges'])
        for label in edge_labels:
            if label:  # Only check non-empty labels
                assert label in self.generator.edge_colors
    
    def test_node_shapes(self):
        """Test that node shapes are properly assigned."""
        code = """
def test_function():
    if x > 0:
        return "positive"
    else:
        return "negative"
"""
        flowchart_data = self.parser.parse_string(code)
        
        # Check that shapes are defined for all node types
        node_types = set(node.node_type for node in flowchart_data['nodes'])
        for node_type in node_types:
            shape = self.generator._get_node_shape(node_type)
            assert shape in ['ellipse', 'box', 'diamond']
    
    def test_empty_flowchart(self):
        """Test handling of empty flowchart data."""
        empty_data = {
            'nodes': [],
            'edges': [],
            'source_code': ''
        }
        
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as f:
            temp_file = f.name
        
        try:
            result_path = self.generator.generate_svg(empty_data, temp_file)
            assert os.path.exists(result_path)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestHTMLRenderer:
    """Test HTML renderer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.renderer = HTMLRenderer()
        self.parser = ASTParser()
    
    def test_generate_html(self):
        """Test interactive HTML generation."""
        code = """
def test_function():
    if x > 0:
        return "positive"
    else:
        return "negative"
"""
        flowchart_data = self.parser.parse_string(code)
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_file = f.name
        
        try:
            result_path = self.renderer.generate_html(flowchart_data, temp_file)
            assert os.path.exists(result_path)
            assert result_path.endswith('.html')
            
            # Check file content
            with open(result_path, 'r') as f:
                content = f.read()
                assert '<!DOCTYPE html>' in content
                assert 'pyvis' in content.lower() or 'network' in content.lower()
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_generate_mermaid_html(self):
        """Test Mermaid HTML generation."""
        code = """
def test_function():
    if x > 0:
        return "positive"
    else:
        return "negative"
"""
        flowchart_data = self.parser.parse_string(code)
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_file = f.name
        
        try:
            result_path = self.renderer.generate_mermaid_html(flowchart_data, temp_file)
            assert os.path.exists(result_path)
            assert result_path.endswith('.html')
            
            # Check file content
            with open(result_path, 'r') as f:
                content = f.read()
                assert '<!DOCTYPE html>' in content
                assert 'mermaid' in content.lower()
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_mermaid_code_generation(self):
        """Test Mermaid code generation."""
        code = """
def test_function():
    if x > 0:
        return "positive"
    else:
        return "negative"
"""
        flowchart_data = self.parser.parse_string(code)
        
        mermaid_code = self.renderer._generate_mermaid_code(flowchart_data)
        
        assert 'flowchart TD' in mermaid_code
        assert len(mermaid_code.split('\n')) > 1
    
    def test_network_options(self):
        """Test network options generation."""
        options = self.renderer._get_network_options()
        
        # Should be valid JSON
        import json
        parsed_options = json.loads(options)
        
        # Should have required keys
        assert 'nodes' in parsed_options
        assert 'edges' in parsed_options
        assert 'physics' in parsed_options
        assert 'interaction' in parsed_options
    
    def test_node_colors(self):
        """Test that node colors are properly assigned."""
        code = """
def test_function():
    if x > 0:
        return "positive"
    else:
        return "negative"
"""
        flowchart_data = self.parser.parse_string(code)
        
        # Check that colors are defined for all node types
        node_types = set(node.node_type for node in flowchart_data['nodes'])
        for node_type in node_types:
            assert node_type in self.renderer.node_colors
    
    def test_edge_colors(self):
        """Test that edge colors are properly assigned."""
        code = """
def test_function():
    if x > 0:
        return "positive"
    else:
        return "negative"
"""
        flowchart_data = self.parser.parse_string(code)
        
        # Check that colors are defined for edge labels
        edge_labels = set(edge.label for edge in flowchart_data['edges'])
        for label in edge_labels:
            if label:  # Only check non-empty labels
                assert label in self.renderer.edge_colors
    
    def test_node_shapes(self):
        """Test that node shapes are properly assigned."""
        code = """
def test_function():
    if x > 0:
        return "positive"
    else:
        return "negative"
"""
        flowchart_data = self.parser.parse_string(code)
        
        # Check that shapes are defined for all node types
        node_types = set(node.node_type for node in flowchart_data['nodes'])
        for node_type in node_types:
            shape = self.renderer._get_node_shape(node_type)
            assert shape in ['ellipse', 'box', 'diamond']
    
    def test_empty_flowchart(self):
        """Test handling of empty flowchart data."""
        empty_data = {
            'nodes': [],
            'edges': [],
            'source_code': ''
        }
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_file = f.name
        
        try:
            result_path = self.renderer.generate_html(empty_data, temp_file)
            assert os.path.exists(result_path)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestVisualizerIntegration:
    """Test integration between parsers and visualizers."""
    
    def test_ast_to_graphviz_integration(self):
        """Test integration between AST parser and Graphviz generator."""
        code = """
def complex_function():
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                print("even")
            else:
                print("odd")
    else:
        print("negative")
"""
        
        parser = ASTParser()
        generator = GraphvizGenerator()
        
        flowchart_data = parser.parse_string(code)
        
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as f:
            temp_file = f.name
        
        try:
            result_path = generator.generate_svg(flowchart_data, temp_file)
            assert os.path.exists(result_path)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_ast_to_html_integration(self):
        """Test integration between AST parser and HTML renderer."""
        code = """
def test_function():
    try:
        result = process_data()
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None
"""
        
        parser = ASTParser()
        renderer = HTMLRenderer()
        
        flowchart_data = parser.parse_string(code)
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_file = f.name
        
        try:
            result_path = renderer.generate_html(flowchart_data, temp_file)
            assert os.path.exists(result_path)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_bytecode_to_visualizer_integration(self):
        """Test integration between Bytecode parser and visualizers."""
        code = """
def bytecode_test():
    x = 1 + 2
    if x > 2:
        return "greater"
    else:
        return "less"
"""
        
        from parser import BytecodeParser
        
        parser = BytecodeParser()
        generator = GraphvizGenerator()
        renderer = HTMLRenderer()
        
        flowchart_data = parser.parse_string(code)
        
        # Test with Graphviz
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as f:
            temp_file = f.name
        
        try:
            result_path = generator.generate_svg(flowchart_data, temp_file)
            assert os.path.exists(result_path)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        # Test with HTML renderer
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_file = f.name
        
        try:
            result_path = renderer.generate_html(flowchart_data, temp_file)
            assert os.path.exists(result_path)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file) 