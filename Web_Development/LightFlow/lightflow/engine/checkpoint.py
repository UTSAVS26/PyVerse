"""
CheckpointManager - Handles saving and loading execution state for workflow resumption.
"""

import json
import os
import pickle
from datetime import datetime
from typing import Dict, Any, Optional
from .logger import Logger


class CheckpointManager:
    """Manages workflow execution checkpoints for resumption."""
    
    def __init__(self, checkpoint_dir: str = ".lightflow-checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.logger = Logger()
        
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, workflow_name: str, completed_tasks: Dict[str, Any]) -> str:
        """
        Save workflow execution state to checkpoint file.
        
        Args:
            workflow_name: Name of the workflow
            completed_tasks: Dictionary of completed task results
            
        Returns:
            Path to the saved checkpoint file
        """
        checkpoint_data = {
            'workflow_name': workflow_name,
            'timestamp': datetime.now().isoformat(),
            'completed_tasks': completed_tasks,
            'total_tasks': len(completed_tasks),
            'version': '1.0'
        }
        
        # Create filename with timestamp and microseconds for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{workflow_name}_{timestamp}.json"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
            
        # Also save latest checkpoint for this workflow
        latest_filepath = os.path.join(self.checkpoint_dir, f"{workflow_name}_latest.json")
        with open(latest_filepath, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
            
        self.logger.info(f"Checkpoint saved: {filepath}")
        return filepath
    
    def load_checkpoint(self, workflow_name: str) -> Dict[str, Any]:
        """
        Load the latest checkpoint for a workflow.
        
        Args:
            workflow_name: Name of the workflow
            
        Returns:
            Checkpoint data dictionary
        """
        latest_filepath = os.path.join(self.checkpoint_dir, f"{workflow_name}_latest.json")
        
        if not os.path.exists(latest_filepath):
            self.logger.info(f"No checkpoint found for workflow: {workflow_name}")
            return {'completed': {}, 'timestamp': None}
            
        try:
            with open(latest_filepath, 'r') as f:
                checkpoint_data = json.load(f)
                
            self.logger.info(f"Loaded checkpoint for workflow: {workflow_name}")
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return {'completed': {}, 'timestamp': None}
    
    def load_specific_checkpoint(self, checkpoint_file: str) -> Dict[str, Any]:
        """
        Load a specific checkpoint file.
        
        Args:
            checkpoint_file: Path to the checkpoint file
            
        Returns:
            Checkpoint data dictionary
        """
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
            
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                
            self.logger.info(f"Loaded specific checkpoint: {checkpoint_file}")
            return checkpoint_data
            
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint file: {e}")
    
    def list_checkpoints(self, workflow_name: Optional[str] = None) -> list:
        """
        List available checkpoints.
        
        Args:
            workflow_name: Optional workflow name to filter by
            
        Returns:
            List of checkpoint file paths
        """
        checkpoints = []
        
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.json'):
                if workflow_name is None or filename.startswith(workflow_name):
                    filepath = os.path.join(self.checkpoint_dir, filename)
                    checkpoints.append(filepath)
                    
        return sorted(checkpoints, reverse=True)  # Most recent first
    
    def delete_checkpoint(self, checkpoint_file: str) -> bool:
        """
        Delete a specific checkpoint file.
        
        Args:
            checkpoint_file: Path to the checkpoint file
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                self.logger.info(f"Deleted checkpoint: {checkpoint_file}")
                return True
            else:
                self.logger.warning(f"Checkpoint file not found: {checkpoint_file}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete checkpoint: {e}")
            return False
    
    def clear_checkpoints(self, workflow_name: Optional[str] = None) -> int:
        """
        Clear all checkpoints for a workflow or all workflows.
        
        Args:
            workflow_name: Optional workflow name to clear checkpoints for
            
        Returns:
            Number of checkpoints deleted
        """
        deleted_count = 0
        
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.json'):
                if workflow_name is None or filename.startswith(workflow_name):
                    filepath = os.path.join(self.checkpoint_dir, filename)
                    if self.delete_checkpoint(filepath):
                        deleted_count += 1
                        
        self.logger.info(f"Cleared {deleted_count} checkpoints")
        return deleted_count
    
    def get_checkpoint_info(self, checkpoint_file: str) -> Dict[str, Any]:
        """
        Get information about a checkpoint file.
        
        Args:
            checkpoint_file: Path to the checkpoint file
            
        Returns:
            Dictionary with checkpoint information
        """
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                
            return {
                'filepath': checkpoint_file,
                'workflow_name': checkpoint_data.get('workflow_name'),
                'timestamp': checkpoint_data.get('timestamp'),
                'total_tasks': checkpoint_data.get('total_tasks', 0),
                'completed_tasks': list(checkpoint_data.get('completed_tasks', {}).keys()),
                'file_size': os.path.getsize(checkpoint_file)
            }
            
        except Exception as e:
            return {
                'filepath': checkpoint_file,
                'error': str(e)
            }
    
    def is_task_completed(self, workflow_name: str, task_name: str) -> bool:
        """
        Check if a specific task is completed in the latest checkpoint.
        
        Args:
            workflow_name: Name of the workflow
            task_name: Name of the task
            
        Returns:
            True if task is completed, False otherwise
        """
        checkpoint = self.load_checkpoint(workflow_name)
        completed_tasks = checkpoint.get('completed_tasks', {})
        return task_name in completed_tasks
    
    def get_completed_tasks(self, workflow_name: str) -> list:
        """
        Get list of completed tasks for a workflow.
        
        Args:
            workflow_name: Name of the workflow
            
        Returns:
            List of completed task names
        """
        checkpoint = self.load_checkpoint(workflow_name)
        completed_tasks = checkpoint.get('completed_tasks', {})
        return list(completed_tasks.keys())
    
    def get_task_result(self, workflow_name: str, task_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the result of a specific task from checkpoint.
        
        Args:
            workflow_name: Name of the workflow
            task_name: Name of the task
            
        Returns:
            Task result dictionary or None if not found
        """
        checkpoint = self.load_checkpoint(workflow_name)
        completed_tasks = checkpoint.get('completed_tasks', {})
        return completed_tasks.get(task_name)
    
    def merge_checkpoints(self, workflow_name: str, checkpoint_files: list) -> str:
        """
        Merge multiple checkpoint files into a single checkpoint.
        
        Args:
            workflow_name: Name of the workflow
            checkpoint_files: List of checkpoint file paths to merge
            
        Returns:
            Path to the merged checkpoint file
        """
        merged_data = {
            'workflow_name': workflow_name,
            'timestamp': datetime.now().isoformat(),
            'completed_tasks': {},
            'total_tasks': 0,
            'version': '1.0',
            'merged_from': checkpoint_files
        }
        
        for checkpoint_file in checkpoint_files:
            try:
                checkpoint_data = self.load_specific_checkpoint(checkpoint_file)
                completed_tasks = checkpoint_data.get('completed_tasks', {})
                
                # Merge completed tasks (later checkpoints override earlier ones)
                merged_data['completed_tasks'].update(completed_tasks)
                
            except Exception as e:
                self.logger.warning(f"Failed to merge checkpoint {checkpoint_file}: {e}")
                
        merged_data['total_tasks'] = len(merged_data['completed_tasks'])
        
        # Save merged checkpoint
        return self.save_checkpoint(workflow_name, merged_data['completed_tasks']) 