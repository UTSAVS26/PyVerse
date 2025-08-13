"""
Tests for WorkflowExecutor class.
"""

import pytest
import tempfile
import os
from lightflow.engine.executor import WorkflowExecutor, ExecutionMode, TaskResult


class TestWorkflowExecutor:
    """Test cases for WorkflowExecutor."""
    
    def setup_method(self):
        """Setup test method."""
        self.executor = WorkflowExecutor(max_workers=2, mode=ExecutionMode.THREAD)
    
    def test_execute_simple_workflow(self):
        """Test executing a simple workflow."""
        workflow = {
            'workflow_name': 'test_workflow',
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
                }
            },
            'settings': {
                'max_parallel_tasks': 2
            }
        }
        
        results = self.executor.execute_workflow(workflow)
        
        assert len(results) == 2
        assert 'task1' in results
        assert 'task2' in results
        assert results['task1'].success
        assert results['task2'].success
    
    def test_execute_python_workflow(self):
        """Test executing a Python workflow."""
        workflow = {
            'workflow_name': 'test_python_workflow',
            'tasks': {
                'task1': {
                    'run': 'print("Hello from Python")',
                    'type': 'python',
                    'depends_on': []
                }
            }
        }
        
        results = self.executor.execute_workflow(workflow)
        
        assert len(results) == 1
        assert 'task1' in results
        assert results['task1'].success
        assert 'Hello from Python' in results['task1'].output
    
    def test_execute_workflow_with_failure(self):
        """Test executing a workflow with a failing task."""
        workflow = {
            'workflow_name': 'test_failing_workflow',
            'tasks': {
                'task1': {
                    'run': 'exit 1',  # This will fail
                    'type': 'shell',
                    'depends_on': []
                }
            }
        }
        
        results = self.executor.execute_workflow(workflow)
        
        assert len(results) == 1
        assert 'task1' in results
        assert not results['task1'].success
        assert results['task1'].exit_code == 1
    
    def test_build_execution_order(self):
        """Test building execution order."""
        tasks = {
            'task1': {'depends_on': []},
            'task2': {'depends_on': ['task1']},
            'task3': {'depends_on': ['task1']},
            'task4': {'depends_on': ['task2', 'task3']}
        }
        
        execution_order = self.executor._build_execution_order(tasks)
        
        assert len(execution_order) == 3
        assert execution_order[0] == ['task1']
        assert set(execution_order[1]) == {'task2', 'task3'}
        assert execution_order[2] == ['task4']
    
    def test_build_execution_order_with_circular_dependency(self):
        """Test building execution order with circular dependency."""
        tasks = {
            'task1': {'depends_on': ['task2']},
            'task2': {'depends_on': ['task1']}
        }
        
        with pytest.raises(ValueError, match="Circular dependency"):
            self.executor._build_execution_order(tasks)
    
    def test_execute_shell_task(self):
        """Test executing a shell task."""
        task_config = {
            'run': 'echo "Test output"',
            'type': 'shell'
        }
        
        output, exit_code = self.executor._execute_shell_task(task_config)
        
        assert exit_code == 0
        assert 'Test output' in output
    
    def test_execute_python_task(self):
        """Test executing a Python task."""
        task_config = {
            'run': 'print("Python test")',
            'type': 'python'
        }
        
        output, exit_code = self.executor._execute_python_task(task_config)
        
        assert exit_code == 0
        assert 'Python test' in output
    
    def test_execute_single_task(self):
        """Test executing a single task."""
        task_config = {
            'run': 'echo "Single task test"',
            'type': 'shell'
        }
        
        result = self.executor._execute_single_task('test_task', task_config)
        
        assert isinstance(result, TaskResult)
        assert result.task_name == 'test_task'
        assert result.success
        assert result.duration > 0
        assert result.exit_code == 0
    
    def test_execute_single_task_with_error(self):
        """Test executing a single task that fails."""
        task_config = {
            'run': 'nonexistent_command',
            'type': 'shell'
        }
        
        result = self.executor._execute_single_task('failing_task', task_config)
        
        assert isinstance(result, TaskResult)
        assert result.task_name == 'failing_task'
        assert not result.success
        assert result.exit_code != 0
    
    def test_execution_modes(self):
        """Test different execution modes."""
        # Test thread mode
        thread_executor = WorkflowExecutor(mode=ExecutionMode.THREAD)
        assert thread_executor.mode == ExecutionMode.THREAD
        
        # Test process mode
        process_executor = WorkflowExecutor(mode=ExecutionMode.PROCESS)
        assert process_executor.mode == ExecutionMode.PROCESS
        
        # Test async mode
        async_executor = WorkflowExecutor(mode=ExecutionMode.ASYNC)
        assert async_executor.mode == ExecutionMode.ASYNC
    
    def test_task_result_creation(self):
        """Test TaskResult creation."""
        result = TaskResult(
            task_name='test_task',
            success=True,
            output='Test output',
            error=None,
            duration=1.5,
            exit_code=0
        )
        
        assert result.task_name == 'test_task'
        assert result.success
        assert result.output == 'Test output'
        assert result.error is None
        assert result.duration == 1.5
        assert result.exit_code == 0 