"""
Graphviz-based flowchart generator.
"""

import os
from typing import Dict, List, Any
from graphviz import Digraph


class GraphvizGenerator:
    """Generate flowcharts using Graphviz."""
    
    def __init__(self):
        self.node_colors = {
            'start': '#4CAF50',
            'end': '#F44336',
            'function': '#2196F3',
            'condition': '#FF9800',
            'loop': '#9C27B0',
            'try': '#795548',
            'except': '#FF5722',
            'finally': '#607D8B',
            'function_call': '#00BCD4',
            'expression': '#E0E0E0',
            'statement': '#BDBDBD',
            'bytecode_block': '#FFC107'
        }
        
        self.edge_colors = {
            'True': '#4CAF50',
            'False': '#F44336',
            'next': '#9C27B0',
            'done': '#FF9800',
            '': '#000000'
        }
    
    def generate_svg(self, flowchart_data: Dict[str, Any], output_path: str = None) -> str:
        """Generate SVG flowchart from parsed data."""
        dot = self._create_digraph(flowchart_data)
        
        if output_path is None:
            output_path = "flowchart.svg"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Render SVG
        svg_content = dot.pipe(format='svg').decode('utf-8')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        return output_path
    
    def generate_png(self, flowchart_data: Dict[str, Any], output_path: str = None) -> str:
        """Generate PNG flowchart from parsed data."""
        dot = self._create_digraph(flowchart_data)
        
        if output_path is None:
            output_path = "flowchart.png"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Render PNG
        png_content = dot.pipe(format='png')
        
        with open(output_path, 'wb') as f:
            f.write(png_content)
        
        return output_path
    
    def _create_digraph(self, flowchart_data: Dict[str, Any]) -> Digraph:
        """Create a Graphviz Digraph from flowchart data."""
        dot = Digraph(comment='Python Code Flowchart')
        dot.attr(rankdir='TB')  # Top to bottom layout
        
        # Add nodes
        for node in flowchart_data['nodes']:
            color = self.node_colors.get(node.node_type, '#E0E0E0')
            shape = self._get_node_shape(node.node_type)
            
            dot.node(
                node.id,
                node.label,
                style='filled',
                fillcolor=color,
                shape=shape,
                fontname='Arial',
                fontsize='10'
            )
        
        # Add edges
        for edge in flowchart_data['edges']:
            color = self.edge_colors.get(edge.label, '#000000')
            style = 'bold' if edge.label else 'solid'
            
            dot.edge(
                edge.source,
                edge.target,
                label=edge.label,
                color=color,
                style=style,
                fontname='Arial',
                fontsize='8'
            )
        
        return dot
    
    def _get_node_shape(self, node_type: str) -> str:
        """Get the appropriate shape for a node type."""
        shapes = {
            'start': 'ellipse',
            'end': 'ellipse',
            'function': 'box',
            'condition': 'diamond',
            'loop': 'box',
            'try': 'box',
            'except': 'box',
            'finally': 'box',
            'function_call': 'box',
            'expression': 'box',
            'statement': 'box',
            'bytecode_block': 'box'
        }
        return shapes.get(node_type, 'box')
    
    def generate_dot(self, flowchart_data: Dict[str, Any], output_path: str = None) -> str:
        """Generate DOT file from parsed data."""
        dot = self._create_digraph(flowchart_data)
        
        if output_path is None:
            output_path = "flowchart.dot"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Save DOT file
        dot.save(output_path)
        
        return output_path 