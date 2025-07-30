"""
Tests for DAGBuilder class.
"""

import pytest
import networkx as nx
from lightflow.engine.dag_builder import DAGBuilder


class TestDAGBuilder:
    """Test cases for DAGBuilder."""
    
    def setup_method(self):
        """Setup test method."""
        self.dag_builder = DAGBuilder()
    
    def test_build_dag(self):
        """Test building DAG from workflow."""
        workflow = {
            'tasks': {
                'task1': {
                    'run': 'echo "Hello"',
                    'type': 'shell',
                    'depends_on': []
                },
                'task2': {
                    'run': 'echo "World"',
                    'type': 'shell',
                    'depends_on': ['task1']
                },
                'task3': {
                    'run': 'echo "Test"',
                    'type': 'shell',
                    'depends_on': ['task2']
                }
            }
        }
        
        dag = self.dag_builder.build_dag(workflow)
        
        assert isinstance(dag, nx.DiGraph)
        assert dag.number_of_nodes() == 3
        assert dag.number_of_edges() == 2
        assert 'task1' in dag.nodes()
        assert 'task2' in dag.nodes()
        assert 'task3' in dag.nodes()
    
    def test_build_dag_with_invalid_dependency(self):
        """Test building DAG with invalid dependency."""
        workflow = {
            'tasks': {
                'task1': {
                    'run': 'echo "Hello"',
                    'type': 'shell',
                    'depends_on': ['nonexistent']
                }
            }
        }
        
        with pytest.raises(ValueError):
            self.dag_builder.build_dag(workflow)
    
    def test_validate_dag_valid(self):
        """Test validating a valid DAG."""
        dag = nx.DiGraph()
        dag.add_node('task1')
        dag.add_node('task2')
        dag.add_node('task3')
        dag.add_edge('task1', 'task2')
        dag.add_edge('task2', 'task3')
        
        is_valid, errors = self.dag_builder.validate_dag(dag)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_dag_with_cycle(self):
        """Test validating DAG with cycle."""
        dag = nx.DiGraph()
        dag.add_node('task1')
        dag.add_node('task2')
        dag.add_edge('task1', 'task2')
        dag.add_edge('task2', 'task1')  # Creates cycle
        
        is_valid, errors = self.dag_builder.validate_dag(dag)
        assert not is_valid
        assert len(errors) > 0
        assert any('Circular dependencies' in error for error in errors)
    
    def test_get_execution_order(self):
        """Test getting execution order."""
        dag = nx.DiGraph()
        dag.add_node('task1')
        dag.add_node('task2')
        dag.add_node('task3')
        dag.add_edge('task1', 'task2')
        dag.add_edge('task2', 'task3')
        
        execution_order = self.dag_builder.get_execution_order(dag)
        
        assert len(execution_order) == 3
        assert execution_order[0] == ['task1']
        assert execution_order[1] == ['task2']
        assert execution_order[2] == ['task3']
    
    def test_get_execution_order_parallel(self):
        """Test getting execution order with parallel tasks."""
        dag = nx.DiGraph()
        dag.add_node('task1')
        dag.add_node('task2')
        dag.add_node('task3')
        dag.add_edge('task1', 'task2')
        dag.add_edge('task1', 'task3')
        
        execution_order = self.dag_builder.get_execution_order(dag)
        
        assert len(execution_order) == 2
        assert execution_order[0] == ['task1']
        assert set(execution_order[1]) == {'task2', 'task3'}
    
    def test_get_task_dependencies(self):
        """Test getting task dependencies."""
        dag = nx.DiGraph()
        dag.add_node('task1')
        dag.add_node('task2')
        dag.add_edge('task1', 'task2')
        
        deps = self.dag_builder.get_task_dependencies(dag, 'task2')
        assert deps == ['task1']
    
    def test_get_task_dependents(self):
        """Test getting task dependents."""
        dag = nx.DiGraph()
        dag.add_node('task1')
        dag.add_node('task2')
        dag.add_edge('task1', 'task2')
        
        dependents = self.dag_builder.get_task_dependents(dag, 'task1')
        assert dependents == ['task2']
    
    def test_get_critical_path(self):
        """Test getting critical path."""
        dag = nx.DiGraph()
        dag.add_node('task1')
        dag.add_node('task2')
        dag.add_node('task3')
        dag.add_edge('task1', 'task2')
        dag.add_edge('task2', 'task3')
        
        critical_path = self.dag_builder.get_critical_path(dag)
        assert critical_path == ['task1', 'task2', 'task3']
    
    def test_estimate_workflow_duration(self):
        """Test estimating workflow duration."""
        dag = nx.DiGraph()
        dag.add_node('task1', duration=1.0)
        dag.add_node('task2', duration=2.0)
        dag.add_node('task3', duration=1.5)
        dag.add_edge('task1', 'task2')
        dag.add_edge('task2', 'task3')
        
        duration = self.dag_builder.estimate_workflow_duration(dag)
        assert duration == 4.5  # 1.0 + 2.0 + 1.5
    
    def test_get_parallel_tasks(self):
        """Test getting parallel tasks."""
        dag = nx.DiGraph()
        dag.add_node('task1')
        dag.add_node('task2')
        dag.add_node('task3')
        dag.add_edge('task1', 'task2')
        dag.add_edge('task1', 'task3')
        
        parallel_tasks = self.dag_builder.get_parallel_tasks(dag)
        
        assert len(parallel_tasks) == 2
        assert parallel_tasks[0] == ['task1']
        assert set(parallel_tasks[1]) == {'task2', 'task3'}
    
    def test_visualize_dag(self):
        """Test DAG visualization."""
        dag = nx.DiGraph()
        dag.add_node('task1', run='echo "Hello"')
        dag.add_node('task2', run='echo "World"')
        dag.add_edge('task1', 'task2')
        
        dot_source = self.dag_builder.visualize_dag(dag)
        assert isinstance(dot_source, str)
        assert 'digraph' in dot_source
        assert 'task1' in dot_source
        assert 'task2' in dot_source
    
    def test_get_node_level(self):
        """Test getting node level."""
        dag = nx.DiGraph()
        dag.add_node('task1')
        dag.add_node('task2')
        dag.add_node('task3')
        dag.add_edge('task1', 'task2')
        dag.add_edge('task2', 'task3')
        
        level1 = self.dag_builder._get_node_level(dag, 'task1')
        level2 = self.dag_builder._get_node_level(dag, 'task2')
        level3 = self.dag_builder._get_node_level(dag, 'task3')
        
        assert level1 == 0
        assert level2 == 1
        assert level3 == 2 