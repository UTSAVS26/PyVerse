"""
DAGBuilder - Builds and validates directed acyclic graphs from workflow configurations.
"""

import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from .logger import Logger


class DAGBuilder:
    """Builds and validates DAGs from workflow configurations."""
    
    def __init__(self):
        self.logger = Logger()
        
    def build_dag(self, workflow: Dict[str, Any]) -> nx.DiGraph:
        """
        Build a NetworkX DAG from workflow configuration.
        
        Args:
            workflow: Workflow configuration dictionary
            
        Returns:
            NetworkX directed graph representing the workflow
        """
        tasks = workflow.get('tasks', {})
        dag = nx.DiGraph()
        
        # Add nodes (tasks)
        for task_name in tasks:
            dag.add_node(task_name, **tasks[task_name])
            
        # Add edges (dependencies)
        for task_name, task_config in tasks.items():
            dependencies = task_config.get('depends_on', [])
            for dep in dependencies:
                if dep not in tasks:
                    raise ValueError(f"Task '{task_name}' depends on '{dep}' which doesn't exist")
                dag.add_edge(dep, task_name)
                
        return dag
    
    def validate_dag(self, dag: nx.DiGraph) -> Tuple[bool, List[str]]:
        """
        Validate that the DAG is acyclic and well-formed.
        
        Args:
            dag: NetworkX directed graph
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(dag))
            if cycles:
                errors.append(f"Circular dependencies detected: {cycles}")
        except nx.NetworkXNoCycle:
            pass  # No cycles, which is good
            
        # Check for disconnected components
        if not nx.is_weakly_connected(dag):
            # This might be okay for some workflows, but warn about it
            self.logger.warning("Workflow has disconnected components")
            
        # Check for nodes with no incoming edges (entry points)
        entry_points = [node for node in dag.nodes() if dag.in_degree(node) == 0]
        if not entry_points:
            errors.append("No entry points found in workflow")
            
        # Check for nodes with no outgoing edges (exit points)
        exit_points = [node for node in dag.nodes() if dag.out_degree(node) == 0]
        if not exit_points:
            errors.append("No exit points found in workflow")
            
        return len(errors) == 0, errors
    
    def get_execution_order(self, dag: nx.DiGraph) -> List[List[str]]:
        """
        Get topological sort of the DAG for execution order.
        
        Args:
            dag: NetworkX directed graph
            
        Returns:
            List of task batches that can be executed in parallel
        """
        try:
            # Get topological sort
            sorted_nodes = list(nx.topological_sort(dag))
            
            # Group nodes by their level in the DAG
            levels = {}
            for node in sorted_nodes:
                level = self._get_node_level(dag, node)
                if level not in levels:
                    levels[level] = []
                levels[level].append(node)
                
            # Return as list of batches
            return [levels[level] for level in sorted(levels.keys())]
            
        except nx.NetworkXError as e:
            raise ValueError(f"Invalid DAG: {e}")
    
    def _get_node_level(self, dag: nx.DiGraph, node: str) -> int:
        """
        Get the level of a node in the DAG (distance from entry points).
        
        Args:
            dag: NetworkX directed graph
            node: Node name
            
        Returns:
            Level of the node
        """
        if dag.in_degree(node) == 0:
            return 0
            
        # Find the maximum level of all predecessors
        pred_levels = [self._get_node_level(dag, pred) for pred in dag.predecessors(node)]
        return max(pred_levels) + 1
    
    def get_task_dependencies(self, dag: nx.DiGraph, task_name: str) -> List[str]:
        """
        Get all dependencies for a specific task.
        
        Args:
            dag: NetworkX directed graph
            task_name: Name of the task
            
        Returns:
            List of task names that this task depends on
        """
        if task_name not in dag:
            raise ValueError(f"Task '{task_name}' not found in DAG")
            
        return list(dag.predecessors(task_name))
    
    def get_task_dependents(self, dag: nx.DiGraph, task_name: str) -> List[str]:
        """
        Get all tasks that depend on a specific task.
        
        Args:
            dag: NetworkX directed graph
            task_name: Name of the task
            
        Returns:
            List of task names that depend on this task
        """
        if task_name not in dag:
            raise ValueError(f"Task '{task_name}' not found in DAG")
            
        return list(dag.successors(task_name))
    
    def get_critical_path(self, dag: nx.DiGraph) -> List[str]:
        """
        Find the critical path in the DAG (longest path from entry to exit).
        
        Args:
            dag: NetworkX directed graph
            
        Returns:
            List of task names in the critical path
        """
        # Find entry and exit points
        entry_points = [node for node in dag.nodes() if dag.in_degree(node) == 0]
        exit_points = [node for node in dag.nodes() if dag.out_degree(node) == 0]
        
        if not entry_points or not exit_points:
            raise ValueError("Cannot find critical path: missing entry or exit points")
            
        # Find longest path from each entry to each exit
        longest_path = []
        max_length = 0
        
        for entry in entry_points:
            for exit_point in exit_points:
                try:
                    path = nx.dag_longest_path(dag, weight='duration')
                    if len(path) > max_length:
                        max_length = len(path)
                        longest_path = path
                except nx.NetworkXError:
                    continue
                    
        return longest_path
    
    def estimate_workflow_duration(self, dag: nx.DiGraph) -> float:
        """
        Estimate total workflow duration based on critical path.
        
        Args:
            dag: NetworkX directed graph
            
        Returns:
            Estimated duration in seconds
        """
        try:
            critical_path = self.get_critical_path(dag)
            total_duration = 0.0
            
            for task_name in critical_path:
                # Get task duration from node attributes
                task_data = dag.nodes[task_name]
                duration = task_data.get('duration', 1.0)  # Default 1 second
                total_duration += duration
                
            return total_duration
            
        except Exception as e:
            self.logger.warning(f"Could not estimate workflow duration: {e}")
            return 0.0
    
    def get_parallel_tasks(self, dag: nx.DiGraph) -> List[List[str]]:
        """
        Get groups of tasks that can be executed in parallel.
        
        Args:
            dag: NetworkX directed graph
            
        Returns:
            List of task groups that can run in parallel
        """
        return self.get_execution_order(dag)
    
    def visualize_dag(self, dag: nx.DiGraph, output_file: Optional[str] = None) -> str:
        """
        Generate a visual representation of the DAG.
        
        Args:
            dag: NetworkX directed graph
            output_file: Optional file path to save the visualization
            
        Returns:
            Graphviz DOT representation of the DAG
        """
        try:
            import graphviz
            
            # Create Graphviz digraph
            dot = graphviz.Digraph(comment='LightFlow Workflow DAG')
            dot.attr(rankdir='TB')  # Top to bottom layout
            
            # Add nodes
            for node in dag.nodes():
                node_data = dag.nodes[node]
                label = f"{node}\n{node_data.get('run', '')[:30]}..."
                dot.node(node, label)
                
            # Add edges
            for edge in dag.edges():
                dot.edge(edge[0], edge[1])
                
            # Save to file if specified
            if output_file:
                dot.render(output_file, format='svg', cleanup=True)
                
            return dot.source
            
        except ImportError:
            self.logger.warning("Graphviz not available, cannot generate visualization")
            return str(dag.edges()) 