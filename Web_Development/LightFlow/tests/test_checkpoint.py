"""
Tests for CheckpointManager class.
"""

import pytest
import tempfile
import os
import json
from lightflow.engine.checkpoint import CheckpointManager


class TestCheckpointManager:
    """Test cases for CheckpointManager."""
    
    def setup_method(self):
        """Setup test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(self.temp_dir)
    
    def teardown_method(self):
        """Teardown test method."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoint."""
        workflow_name = "test_workflow"
        completed_tasks = {
            "task1": {
                "success": True,
                "output": "Task 1 completed",
                "duration": 1.5,
                "exit_code": 0
            },
            "task2": {
                "success": False,
                "output": "Task 2 failed",
                "duration": 2.0,
                "exit_code": 1
            }
        }
        
        # Save checkpoint
        filepath = self.checkpoint_manager.save_checkpoint(workflow_name, completed_tasks)
        assert os.path.exists(filepath)
        
        # Load checkpoint
        checkpoint_data = self.checkpoint_manager.load_checkpoint(workflow_name)
        assert checkpoint_data['workflow_name'] == workflow_name
        assert len(checkpoint_data['completed_tasks']) == 2
        assert 'task1' in checkpoint_data['completed_tasks']
        assert 'task2' in checkpoint_data['completed_tasks']
    
    def test_load_nonexistent_checkpoint(self):
        """Test loading nonexistent checkpoint."""
        checkpoint_data = self.checkpoint_manager.load_checkpoint("nonexistent")
        assert checkpoint_data['completed'] == {}
        assert checkpoint_data['timestamp'] is None
    
    def test_load_specific_checkpoint(self):
        """Test loading specific checkpoint file."""
        workflow_name = "test_workflow"
        completed_tasks = {"task1": {"success": True}}
        
        # Save checkpoint
        filepath = self.checkpoint_manager.save_checkpoint(workflow_name, completed_tasks)
        
        # Load specific checkpoint
        checkpoint_data = self.checkpoint_manager.load_specific_checkpoint(filepath)
        assert checkpoint_data['workflow_name'] == workflow_name
        assert 'task1' in checkpoint_data['completed_tasks']
    
    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        workflow_name = "test_workflow"
        completed_tasks = {"task1": {"success": True}}
        
        # Save multiple checkpoints
        self.checkpoint_manager.save_checkpoint(workflow_name, completed_tasks)
        self.checkpoint_manager.save_checkpoint(workflow_name, completed_tasks)
        
        # List checkpoints
        checkpoints = self.checkpoint_manager.list_checkpoints(workflow_name)
        assert len(checkpoints) >= 2  # At least 2 timestamped + 1 latest
        
        # List all checkpoints
        all_checkpoints = self.checkpoint_manager.list_checkpoints()
        assert len(all_checkpoints) >= 2
    
    def test_delete_checkpoint(self):
        """Test deleting checkpoint."""
        workflow_name = "test_workflow"
        completed_tasks = {"task1": {"success": True}}
        
        # Save checkpoint
        filepath = self.checkpoint_manager.save_checkpoint(workflow_name, completed_tasks)
        
        # Delete checkpoint
        success = self.checkpoint_manager.delete_checkpoint(filepath)
        assert success
        assert not os.path.exists(filepath)
    
    def test_clear_checkpoints(self):
        """Test clearing checkpoints."""
        workflow_name = "test_workflow"
        completed_tasks = {"task1": {"success": True}}
        
        # Save checkpoints
        self.checkpoint_manager.save_checkpoint(workflow_name, completed_tasks)
        self.checkpoint_manager.save_checkpoint("other_workflow", completed_tasks)
        
        # Clear specific workflow checkpoints
        deleted_count = self.checkpoint_manager.clear_checkpoints(workflow_name)
        assert deleted_count > 0
        
        # Clear all checkpoints
        deleted_count = self.checkpoint_manager.clear_checkpoints()
        assert deleted_count >= 0
    
    def test_get_checkpoint_info(self):
        """Test getting checkpoint information."""
        workflow_name = "test_workflow"
        completed_tasks = {"task1": {"success": True}}
        
        # Save checkpoint
        filepath = self.checkpoint_manager.save_checkpoint(workflow_name, completed_tasks)
        
        # Get checkpoint info
        info = self.checkpoint_manager.get_checkpoint_info(filepath)
        assert info['workflow_name'] == workflow_name
        assert info['total_tasks'] == 1
        assert 'task1' in info['completed_tasks']
        assert 'file_size' in info
    
    def test_is_task_completed(self):
        """Test checking if task is completed."""
        workflow_name = "test_workflow"
        completed_tasks = {"task1": {"success": True}}
        
        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(workflow_name, completed_tasks)
        
        # Check task completion
        assert self.checkpoint_manager.is_task_completed(workflow_name, "task1")
        assert not self.checkpoint_manager.is_task_completed(workflow_name, "task2")
    
    def test_get_completed_tasks(self):
        """Test getting completed tasks."""
        workflow_name = "test_workflow"
        completed_tasks = {"task1": {"success": True}, "task2": {"success": False}}
        
        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(workflow_name, completed_tasks)
        
        # Get completed tasks
        completed = self.checkpoint_manager.get_completed_tasks(workflow_name)
        assert set(completed) == {"task1", "task2"}
    
    def test_get_task_result(self):
        """Test getting task result."""
        workflow_name = "test_workflow"
        task_result = {"success": True, "output": "Task completed", "duration": 1.5}
        completed_tasks = {"task1": task_result}
        
        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(workflow_name, completed_tasks)
        
        # Get task result
        result = self.checkpoint_manager.get_task_result(workflow_name, "task1")
        assert result == task_result
        
        # Get nonexistent task result
        result = self.checkpoint_manager.get_task_result(workflow_name, "nonexistent")
        assert result is None
    
    def test_merge_checkpoints(self):
        """Test merging checkpoints."""
        workflow_name = "test_workflow"
        
        # Create multiple checkpoints
        checkpoint1 = {"task1": {"success": True}}
        checkpoint2 = {"task2": {"success": True}}
        
        filepath1 = self.checkpoint_manager.save_checkpoint(workflow_name, checkpoint1)
        filepath2 = self.checkpoint_manager.save_checkpoint(workflow_name, checkpoint2)
        
        # Merge checkpoints
        merged_filepath = self.checkpoint_manager.merge_checkpoints(
            workflow_name, [filepath1, filepath2]
        )
        
        # Verify merged checkpoint
        merged_data = self.checkpoint_manager.load_specific_checkpoint(merged_filepath)
        assert 'task1' in merged_data['completed_tasks']
        assert 'task2' in merged_data['completed_tasks']
        assert merged_data['total_tasks'] == 2 