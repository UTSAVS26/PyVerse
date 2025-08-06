"""
Tests for WorkflowLoader class.
"""

import pytest
import tempfile
import os
from lightflow.parser.workflow_loader import WorkflowLoader


class TestWorkflowLoader:
    """Test cases for WorkflowLoader."""
    
    def setup_method(self):
        """Setup test method."""
        self.loader = WorkflowLoader()
        
    def test_load_yaml_workflow(self):
        """Test loading YAML workflow."""
        yaml_content = """
workflow_name: test_workflow
tasks:
  task1:
    run: echo "Hello"
    type: shell
    depends_on: []
  task2:
    run: echo "World"
    type: shell
    depends_on: [task1]
settings:
  max_parallel_tasks: 2
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_file = f.name
            
        try:
            workflow = self.loader.load_workflow(temp_file)
            
            assert workflow['workflow_name'] == 'test_workflow'
            assert len(workflow['tasks']) == 2
            assert 'task1' in workflow['tasks']
            assert 'task2' in workflow['tasks']
            assert workflow['settings']['max_parallel_tasks'] == 2
            
        finally:
            os.unlink(temp_file)
    
    def test_load_json_workflow(self):
        """Test loading JSON workflow."""
        json_content = """
{
  "workflow_name": "test_workflow",
  "tasks": {
    "task1": {
      "run": "echo \\"Hello\\"",
      "type": "shell",
      "depends_on": []
    },
    "task2": {
      "run": "echo \\"World\\"",
      "type": "shell",
      "depends_on": ["task1"]
    }
  },
  "settings": {
    "max_parallel_tasks": 2
  }
}
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(json_content)
            temp_file = f.name
            
        try:
            workflow = self.loader.load_workflow(temp_file)
            
            assert workflow['workflow_name'] == 'test_workflow'
            assert len(workflow['tasks']) == 2
            
        finally:
            os.unlink(temp_file)
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            self.loader.load_workflow('nonexistent.yaml')
    
    def test_load_invalid_format(self):
        """Test loading file with invalid format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is not a workflow file")
            temp_file = f.name
            
        try:
            with pytest.raises(ValueError):
                self.loader.load_workflow(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_validate_valid_workflow(self):
        """Test validating a valid workflow."""
        workflow = {
            'workflow_name': 'test_workflow',
            'tasks': {
                'task1': {
                    'run': 'echo "Hello"',
                    'type': 'shell',
                    'depends_on': []
                }
            },
            'settings': {
                'max_parallel_tasks': 2
            }
        }
        
        is_valid, errors = self.loader.validate_workflow(workflow)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_invalid_workflow(self):
        """Test validating an invalid workflow."""
        workflow = {
            'tasks': {
                'task1': {
                    'type': 'shell',
                    'depends_on': []
                }
            }
        }
        
        is_valid, errors = self.loader.validate_workflow(workflow)
        assert not is_valid
        assert len(errors) > 0
        assert any('workflow_name' in error for error in errors)
    
    def test_validate_task_without_run(self):
        """Test validating task without run field."""
        workflow = {
            'workflow_name': 'test_workflow',
            'tasks': {
                'task1': {
                    'type': 'shell',
                    'depends_on': []
                }
            }
        }
        
        is_valid, errors = self.loader.validate_workflow(workflow)
        assert not is_valid
        assert any('run' in error for error in errors)
    
    def test_validate_invalid_dependency(self):
        """Test validating workflow with invalid dependency."""
        workflow = {
            'workflow_name': 'test_workflow',
            'tasks': {
                'task1': {
                    'run': 'echo "Hello"',
                    'type': 'shell',
                    'depends_on': ['nonexistent_task']
                }
            }
        }
        
        is_valid, errors = self.loader.validate_workflow(workflow)
        assert not is_valid
        assert any('nonexistent_task' in error for error in errors)
    
    def test_create_workflow_template(self):
        """Test creating workflow template."""
        template = self.loader.create_workflow_template('test_workflow')
        
        assert template['workflow_name'] == 'test_workflow'
        assert 'tasks' in template
        assert 'settings' in template
        assert 'example_task' in template['tasks']
    
    def test_save_workflow_yaml(self):
        """Test saving workflow to YAML."""
        workflow = {
            'workflow_name': 'test_workflow',
            'tasks': {
                'task1': {
                    'run': 'echo "Hello"',
                    'type': 'shell'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_file = f.name
            
        try:
            self.loader.save_workflow(workflow, temp_file, 'yaml')
            
            # Verify file was created and can be loaded
            loaded_workflow = self.loader.load_workflow(temp_file)
            assert loaded_workflow['workflow_name'] == 'test_workflow'
            
        finally:
            os.unlink(temp_file)
    
    def test_save_workflow_json(self):
        """Test saving workflow to JSON."""
        workflow = {
            'workflow_name': 'test_workflow',
            'tasks': {
                'task1': {
                    'run': 'echo "Hello"',
                    'type': 'shell'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
            
        try:
            self.loader.save_workflow(workflow, temp_file, 'json')
            
            # Verify file was created and can be loaded
            loaded_workflow = self.loader.load_workflow(temp_file)
            assert loaded_workflow['workflow_name'] == 'test_workflow'
            
        finally:
            os.unlink(temp_file)
    
    def test_extract_task_dependencies(self):
        """Test extracting task dependencies."""
        workflow = {
            'tasks': {
                'task1': {
                    'run': 'echo "Hello"',
                    'depends_on': []
                },
                'task2': {
                    'run': 'echo "World"',
                    'depends_on': ['task1']
                }
            }
        }
        
        dependencies = self.loader.extract_task_dependencies(workflow)
        
        assert dependencies['task1'] == []
        assert dependencies['task2'] == ['task1']
    
    def test_get_workflow_info(self):
        """Test getting workflow information."""
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
                    'type': 'python',
                    'depends_on': ['task1']
                }
            },
            'settings': {
                'max_parallel_tasks': 2
            }
        }
        
        info = self.loader.get_workflow_info(workflow)
        
        assert info['workflow_name'] == 'test_workflow'
        assert info['total_tasks'] == 2
        assert info['task_types']['shell'] == 1
        assert info['task_types']['python'] == 1
        assert info['entry_tasks'] == ['task1']
        assert info['exit_tasks'] == ['task2']
        assert info['has_settings'] is True 