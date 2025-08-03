"""
HTML-based interactive flowchart renderer.
"""

import os
import json
from typing import Dict, List, Any
from pyvis.network import Network


class HTMLRenderer:
    """Generate interactive HTML flowcharts using pyvis."""
    
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
    
    def generate_html(self, flowchart_data: Dict[str, Any], output_path: str = None) -> str:
        """Generate interactive HTML flowchart from parsed data."""
        if output_path is None:
            output_path = "flowchart.html"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Create network
        net = Network(height='800px', width='100%', bgcolor='#ffffff', font_color='#000000')
        net.set_options(self._get_network_options())
        
        # Add nodes
        for node in flowchart_data['nodes']:
            color = self.node_colors.get(node.node_type, '#E0E0E0')
            shape = self._get_node_shape(node.node_type)
            
            net.add_node(
                node.id,
                label=node.label,
                color=color,
                shape=shape,
                title=f"Type: {node.node_type}<br>Line: {node.line_number}"
            )
        
        # Add edges
        for edge in flowchart_data['edges']:
            color = self.edge_colors.get(edge.label, '#000000')
            title = f"Condition: {edge.label}" if edge.label else "Flow"
            
            net.add_edge(
                edge.source,
                edge.target,
                title=title,
                color=color,
                arrows='to'
            )
        
        # Generate HTML with DOCTYPE
        html_content = self._generate_html_with_doctype(net)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def generate_mermaid_html(self, flowchart_data: Dict[str, Any], output_path: str = None) -> str:
        """Generate HTML with Mermaid.js flowchart."""
        if output_path is None:
            output_path = "flowchart_mermaid.html"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Generate Mermaid syntax
        mermaid_code = self._generate_mermaid_code(flowchart_data)
        
        # Create HTML template
        html_content = self._get_mermaid_html_template(mermaid_code)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_html_with_doctype(self, net: Network) -> str:
        """Generate HTML with proper DOCTYPE declaration."""
        # Get the HTML from pyvis
        html_content = net.generate_html()
        
        # Add DOCTYPE if not present
        if not html_content.startswith('<!DOCTYPE'):
            html_content = '<!DOCTYPE html>\n' + html_content
        
        return html_content
    
    def _get_network_options(self) -> str:
        """Get network visualization options."""
        options = {
            "nodes": {
                "font": {
                    "size": 12,
                    "face": "Arial"
                },
                "borderWidth": 2,
                "shadow": True
            },
            "edges": {
                "font": {
                    "size": 10,
                    "face": "Arial"
                },
                "shadow": True,
                "smooth": {
                    "type": "continuous"
                }
            },
            "physics": {
                "enabled": True,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                }
            },
            "interaction": {
                "hover": True,
                "tooltipDelay": 200
            }
        }
        return json.dumps(options)
    
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
    
    def _generate_mermaid_code(self, flowchart_data: Dict[str, Any]) -> str:
        """Generate Mermaid.js flowchart code."""
        lines = ["flowchart TD"]
        
        # Add nodes
        for node in flowchart_data['nodes']:
            node_id = node.id.replace('-', '_')
            label = node.label.replace('"', '\\"')
            
            if node.node_type == 'start':
                lines.append(f"    {node_id}[{label}]")
            elif node.node_type == 'end':
                lines.append(f"    {node_id}([{label}])")
            elif node.node_type == 'condition':
                lines.append(f"    {node_id}{{{label}}}")
            else:
                lines.append(f"    {node_id}[{label}]")
        
        # Add edges
        for edge in flowchart_data['edges']:
            source_id = edge.source.replace('-', '_')
            target_id = edge.target.replace('-', '_')
            
            if edge.label:
                lines.append(f"    {source_id} -->|{edge.label}| {target_id}")
            else:
                lines.append(f"    {source_id} --> {target_id}")
        
        return '\n'.join(lines)
    
    def _get_mermaid_html_template(self, mermaid_code: str) -> str:
        """Get HTML template with Mermaid.js."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>PyFlowViz - Flowchart</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }}
        .mermaid {{
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Python Code Flowchart</h1>
        <div class="mermaid">
{mermaid_code}
        </div>
    </div>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            flowchart: {{
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            }}
        }});
    </script>
</body>
</html>""" 