"""
WorkflowLoader - Loads and parses YAML/JSON workflow files.
"""

import yaml
import json
import os
from typing import Dict, Any, Optional, Union
from pathlib import Path
from ..engine.logger import Logger


class WorkflowLoader:
    """Loads and parses workflow configuration files."""
    
    def __init__(self):
        self.logger = Logger()
        
    def load_workflow(self, file_path: str) -> Dict[str, Any]:
        """
        Load workflow from file (YAML or JSON).
        
        Args:
            file_path: Path to the workflow file
            
        Returns:
            Workflow configuration dictionary
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Workflow file not found: {file_path}")
            
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in ['.yaml', '.yml']:
            return self._load_yaml(file_path)
        elif file_ext == '.json':
            return self._load_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _load_yaml(self, file_path: str) -> Dict[str, Any]:
        """Load workflow from YAML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow = yaml.safe_load(f)
                
            if not isinstance(workflow, dict):
                raise ValueError("YAML file must contain a dictionary")
                
            return workflow
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load YAML file: {e}")
    
    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """Load workflow from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
                
            if not isinstance(workflow, dict):
                raise ValueError("JSON file must contain a dictionary")
                
            return workflow
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load JSON file: {e}")
    
    def validate_workflow(self, workflow: Dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate workflow configuration.
        
        Args:
            workflow: Workflow configuration dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        if 'workflow_name' not in workflow:
            errors.append("Missing required field: workflow_name")
            
        if 'tasks' not in workflow:
            errors.append("Missing required field: tasks")
        elif not isinstance(workflow['tasks'], dict):
            errors.append("Field 'tasks' must be a dictionary")
        elif len(workflow['tasks']) == 0:
            errors.append("No tasks defined in workflow")
            
        # Validate tasks
        if 'tasks' in workflow and isinstance(workflow['tasks'], dict):
            task_errors = self._validate_tasks(workflow['tasks'])
            errors.extend(task_errors)
            
        # Validate settings
        if 'settings' in workflow:
            settings_errors = self._validate_settings(workflow['settings'])
            errors.extend(settings_errors)
            
        return len(errors) == 0, errors
    
    def _validate_tasks(self, tasks: Dict[str, Any]) -> list[str]:
        """Validate task configurations."""
        errors = []
        
        for task_name, task_config in tasks.items():
            if not isinstance(task_config, dict):
                errors.append(f"Task '{task_name}' configuration must be a dictionary")
                continue
                
            # Check required task fields
            if 'run' not in task_config:
                errors.append(f"Task '{task_name}' missing required field: run")
                
            # Validate dependencies
            if 'depends_on' in task_config:
                deps = task_config['depends_on']
                if not isinstance(deps, list):
                    errors.append(f"Task '{task_name}' depends_on must be a list")
                else:
                    for dep in deps:
                        if dep not in tasks:
                            errors.append(f"Task '{task_name}' depends on '{dep}' which doesn't exist")
                            
            # Validate task type
            task_type = task_config.get('type', 'shell')
            if task_type not in ['shell', 'python']:
                errors.append(f"Task '{task_name}' has invalid type: {task_type}")
                
        return errors
    
    def _validate_settings(self, settings: Dict[str, Any]) -> list[str]:
        """Validate workflow settings."""
        errors = []
        
        # Validate max_parallel_tasks
        if 'max_parallel_tasks' in settings:
            max_tasks = settings['max_parallel_tasks']
            if not isinstance(max_tasks, int) or max_tasks <= 0:
                errors.append("max_parallel_tasks must be a positive integer")
                
        # Validate retries
        if 'retries' in settings:
            retries = settings['retries']
            if not isinstance(retries, int) or retries < 0:
                errors.append("retries must be a non-negative integer")
                
        # Validate log_dir
        if 'log_dir' in settings:
            log_dir = settings['log_dir']
            if not isinstance(log_dir, str):
                errors.append("log_dir must be a string")
                
        return errors
    
    def create_workflow_template(self, workflow_name: str) -> Dict[str, Any]:
        """
        Create a basic workflow template.
        
        Args:
            workflow_name: Name for the workflow
            
        Returns:
            Template workflow configuration
        """
        return {
            'workflow_name': workflow_name,
            'tasks': {
                'example_task': {
                    'run': 'echo "Hello, LightFlow!"',
                    'type': 'shell',
                    'depends_on': []
                }
            },
            'settings': {
                'max_parallel_tasks': 4,
                'retries': 2,
                'log_dir': 'logs'
            }
        }
    
    def save_workflow(self, workflow: Dict[str, Any], file_path: str, format: str = 'yaml'):
        """
        Save workflow configuration to file.
        
        Args:
            workflow: Workflow configuration dictionary
            file_path: Path to save the workflow file
            format: Output format ('yaml' or 'json')
        """
        if format.lower() == 'yaml':
            self._save_yaml(workflow, file_path)
        elif format.lower() == 'json':
            self._save_json(workflow, file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_yaml(self, workflow: Dict[str, Any], file_path: str):
        """Save workflow to YAML file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(workflow, f, default_flow_style=False, indent=2)
    
    def _save_json(self, workflow: Dict[str, Any], file_path: str):
        """Save workflow to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(workflow, f, indent=2)
    
    def merge_workflows(self, workflows: list[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple workflows into a single workflow.
        
        Args:
            workflows: List of workflow configurations
            
        Returns:
            Merged workflow configuration
        """
        if not workflows:
            raise ValueError("No workflows to merge")
            
        merged = {
            'workflow_name': f"merged_{workflows[0]['workflow_name']}",
            'tasks': {},
            'settings': {}
        }
        
        # Merge tasks
        for workflow in workflows:
            workflow_name = workflow['workflow_name']
            tasks = workflow.get('tasks', {})
            
            for task_name, task_config in tasks.items():
                # Prefix task name with workflow name to avoid conflicts
                prefixed_name = f"{workflow_name}_{task_name}"
                merged['tasks'][prefixed_name] = task_config.copy()
                
        # Merge settings (use first workflow's settings as base)
        if workflows:
            merged['settings'] = workflows[0].get('settings', {}).copy()
            
        return merged
    
    def extract_task_dependencies(self, workflow: Dict[str, Any]) -> Dict[str, list[str]]:
        """
        Extract task dependencies from workflow.
        
        Args:
            workflow: Workflow configuration
            
        Returns:
            Dictionary mapping task names to their dependencies
        """
        dependencies = {}
        tasks = workflow.get('tasks', {})
        
        for task_name, task_config in tasks.items():
            deps = task_config.get('depends_on', [])
            dependencies[task_name] = deps
            
        return dependencies
    
    def get_workflow_info(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get information about a workflow.
        
        Args:
            workflow: Workflow configuration
            
        Returns:
            Dictionary with workflow information
        """
        tasks = workflow.get('tasks', {})
        
        # Count tasks by type
        task_types = {}
        for task_config in tasks.values():
            task_type = task_config.get('type', 'shell')
            task_types[task_type] = task_types.get(task_type, 0) + 1
            
        # Find entry and exit tasks
        entry_tasks = []
        exit_tasks = []
        
        for task_name, task_config in tasks.items():
            deps = task_config.get('depends_on', [])
            dependents = []
            
            # Find tasks that depend on this task
            for other_name, other_config in tasks.items():
                if task_name in other_config.get('depends_on', []):
                    dependents.append(other_name)
                    
            if not deps:
                entry_tasks.append(task_name)
            if not dependents:
                exit_tasks.append(task_name)
                
        return {
            'workflow_name': workflow.get('workflow_name', 'Unknown'),
            'total_tasks': len(tasks),
            'task_types': task_types,
            'entry_tasks': entry_tasks,
            'exit_tasks': exit_tasks,
            'has_settings': 'settings' in workflow
        } 