"""
DAGViewer - Generates DAG visualizations using graphviz.
"""

import networkx as nx
from typing import Optional, Dict, Any
from ..engine.logger import Logger


class DAGViewer:
    """Generates visual representations of DAGs."""
    
    def __init__(self):
        self.logger = Logger()
        
    def create_dot_graph(self, dag: nx.DiGraph, workflow_name: str = "Workflow") -> str:
        """
        Create a Graphviz DOT representation of the DAG.
        
        Args:
            dag: NetworkX directed graph
            workflow_name: Name of the workflow
            
        Returns:
            DOT source code
        """
        try:
            import graphviz
            
            # Create Graphviz digraph
            dot = graphviz.Digraph(comment=f'{workflow_name} DAG')
            dot.attr(rankdir='TB')  # Top to bottom layout
            dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
            
            # Add nodes
            for node in dag.nodes():
                node_data = dag.nodes[node]
                label = f"{node}\n{node_data.get('run', '')[:30]}..."
                dot.node(node, label)
                
            # Add edges
            for edge in dag.edges():
                dot.edge(edge[0], edge[1])
                
            return dot.source
            
        except ImportError:
            self.logger.warning("Graphviz not available, using text representation")
            return self._create_text_graph(dag, workflow_name)
    
    def _create_text_graph(self, dag: nx.DiGraph, workflow_name: str) -> str:
        """Create a text representation of the DAG."""
        lines = [f"# {workflow_name} DAG", ""]
        
        # Add nodes
        lines.append("## Nodes:")
        for node in dag.nodes():
            node_data = dag.nodes[node]
            lines.append(f"  - {node}: {node_data.get('run', '')}")
            
        # Add edges
        lines.append("\n## Dependencies:")
        for edge in dag.edges():
            lines.append(f"  - {edge[0]} -> {edge[1]}")
            
        return "\n".join(lines)
    
    def render_dag(self, dag: nx.DiGraph, output_file: str, 
                   format: str = 'svg', workflow_name: str = "Workflow") -> bool:
        """
        Render DAG to file.
        
        Args:
            dag: NetworkX directed graph
            output_file: Output file path
            format: Output format (svg, png, pdf)
            workflow_name: Name of the workflow
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import graphviz
            
            # Create Graphviz digraph
            dot = graphviz.Digraph(comment=f'{workflow_name} DAG')
            dot.attr(rankdir='TB')
            dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
            
            # Add nodes with different colors based on type
            for node in dag.nodes():
                node_data = dag.nodes[node]
                task_type = node_data.get('type', 'shell')
                
                # Color nodes based on task type
                if task_type == 'shell':
                    fillcolor = 'lightgreen'
                elif task_type == 'python':
                    fillcolor = 'lightyellow'
                else:
                    fillcolor = 'lightblue'
                    
                label = f"{node}\n{node_data.get('run', '')[:30]}..."
                dot.node(node, label, fillcolor=fillcolor)
                
            # Add edges
            for edge in dag.edges():
                dot.edge(edge[0], edge[1])
                
            # Render to file
            dot.render(output_file, format=format, cleanup=True)
            self.logger.info(f"DAG rendered to: {output_file}.{format}")
            return True
            
        except ImportError:
            self.logger.error("Graphviz not available. Install with: pip install graphviz")
            return False
        except Exception as e:
            self.logger.error(f"Failed to render DAG: {e}")
            return False
    
    def create_interactive_html(self, dag: nx.DiGraph, output_file: str, 
                               workflow_name: str = "Workflow") -> bool:
        """
        Create an interactive HTML visualization of the DAG.
        
        Args:
            dag: NetworkX directed graph
            output_file: Output file path
            workflow_name: Name of the workflow
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import json
            
            # Convert DAG to JSON for JavaScript visualization
            nodes = []
            edges = []
            
            for node in dag.nodes():
                node_data = dag.nodes[node]
                nodes.append({
                    'id': node,
                    'label': node,
                    'type': node_data.get('type', 'shell'),
                    'command': node_data.get('run', '')
                })
                
            for edge in dag.edges():
                edges.append({
                    'from': edge[0],
                    'to': edge[1]
                })
                
            # Create HTML template
            html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{workflow_name} DAG</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        #dag-container {{
            width: 100%;
            height: 600px;
            border: 1px solid #ccc;
        }}
    </style>
</head>
<body>
    <h1>{workflow_name} DAG</h1>
    <div id="dag-container"></div>
    
    <script>
        // Create nodes
        const nodes = new vis.DataSet({json.dumps(nodes)});
        
        // Create edges
        const edges = new vis.DataSet({json.dumps(edges)});
        
        // Create network
        const container = document.getElementById('dag-container');
        const data = {{ nodes: nodes, edges: edges }};
        const options = {{
            layout: {{
                hierarchical: {{
                    direction: 'UD',
                    sortMethod: 'directed'
                }}
            }},
            nodes: {{
                shape: 'box',
                font: {{ size: 12 }},
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                arrows: 'to',
                smooth: {{ type: 'cubicBezier' }}
            }}
        }};
        
        const network = new vis.Network(container, data, options);
    </script>
</body>
</html>
"""
            
            with open(output_file, 'w') as f:
                f.write(html_template)
                
            self.logger.info(f"Interactive DAG created: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create interactive HTML: {e}")
            return False
    
    def get_dag_statistics(self, dag: nx.DiGraph) -> Dict[str, Any]:
        """
        Get statistics about the DAG.
        
        Args:
            dag: NetworkX directed graph
            
        Returns:
            Dictionary with DAG statistics
        """
        stats = {
            'total_nodes': dag.number_of_nodes(),
            'total_edges': dag.number_of_edges(),
            'entry_nodes': len([n for n in dag.nodes() if dag.in_degree(n) == 0]),
            'exit_nodes': len([n for n in dag.nodes() if dag.out_degree(n) == 0]),
            'max_depth': 0,
            'task_types': {}
        }
        
        # Calculate max depth
        if dag.number_of_nodes() > 0:
            try:
                # Use NetworkX to find longest path
                longest_path = nx.dag_longest_path(dag)
                stats['max_depth'] = len(longest_path)
            except:
                # Fallback calculation
                levels = {}
                for node in nx.topological_sort(dag):
                    level = max([levels.get(pred, 0) for pred in dag.predecessors(node)], default=0) + 1
                    levels[node] = level
                stats['max_depth'] = max(levels.values()) if levels else 0
        
        # Count task types
        for node in dag.nodes():
            node_data = dag.nodes[node]
            task_type = node_data.get('type', 'shell')
            stats['task_types'][task_type] = stats['task_types'].get(task_type, 0) + 1
            
        return stats 