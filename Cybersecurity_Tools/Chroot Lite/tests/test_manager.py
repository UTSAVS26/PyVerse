"""
Unit tests for SandboxManager.
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from sandbox.manager import SandboxManager


class TestSandboxManager:
    """Test cases for SandboxManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def manager(self, temp_dir):
        """Create a SandboxManager instance for testing."""
        return SandboxManager(base_dir=temp_dir)
    
    def test_init(self, temp_dir):
        """Test SandboxManager initialization."""
        manager = SandboxManager(base_dir=temp_dir)
        assert manager.base_dir == temp_dir
        assert manager.sandboxes == {}
        assert manager.config_file == os.path.join(temp_dir, "sandboxes.json")
    
    def test_create_sandbox_success(self, manager):
        """Test successful sandbox creation."""
        result = manager.create_sandbox("testbox", memory_limit_mb=256, cpu_limit_seconds=60)
        
        assert result is True
        assert "testbox" in manager.sandboxes
        assert manager.sandboxes["testbox"]["name"] == "testbox"
        assert manager.sandboxes["testbox"]["memory_limit_mb"] == 256
        assert manager.sandboxes["testbox"]["cpu_limit_seconds"] == 60
        assert manager.sandboxes["testbox"]["block_network"] is True
        
        # Check if sandbox directory was created
        sandbox_path = manager.sandboxes["testbox"]["path"]
        assert os.path.exists(sandbox_path)
        assert os.path.isdir(sandbox_path)
    
    def test_create_sandbox_already_exists(self, manager):
        """Test creating a sandbox that already exists."""
        # Create first sandbox
        result1 = manager.create_sandbox("testbox")
        assert result1 is True
        
        # Try to create the same sandbox again
        result2 = manager.create_sandbox("testbox")
        assert result2 is False
    
    def test_create_sandbox_with_network_allowed(self, manager):
        """Test creating a sandbox with network access allowed."""
        result = manager.create_sandbox("testbox", block_network=False)
        
        assert result is True
        assert manager.sandboxes["testbox"]["block_network"] is False
    
    def test_delete_sandbox_success(self, manager):
        """Test successful sandbox deletion."""
        # Create a sandbox first
        manager.create_sandbox("testbox")
        assert "testbox" in manager.sandboxes
        
        # Delete the sandbox
        result = manager.delete_sandbox("testbox")
        assert result is True
        assert "testbox" not in manager.sandboxes
    
    def test_delete_sandbox_not_exists(self, manager):
        """Test deleting a sandbox that doesn't exist."""
        result = manager.delete_sandbox("nonexistent")
        assert result is False
    
    def test_list_sandboxes_empty(self, manager):
        """Test listing sandboxes when none exist."""
        sandboxes = manager.list_sandboxes()
        assert sandboxes == []
    
    def test_list_sandboxes_with_sandboxes(self, manager):
        """Test listing sandboxes when some exist."""
        # Create some sandboxes
        manager.create_sandbox("box1")
        manager.create_sandbox("box2")
        
        sandboxes = manager.list_sandboxes()
        assert len(sandboxes) == 2
        assert any(s["name"] == "box1" for s in sandboxes)
        assert any(s["name"] == "box2" for s in sandboxes)
    
    def test_get_sandbox_info_exists(self, manager):
        """Test getting info for an existing sandbox."""
        manager.create_sandbox("testbox", memory_limit_mb=512)
        info = manager.get_sandbox_info("testbox")
        
        assert info is not None
        assert info["name"] == "testbox"
        assert info["memory_limit_mb"] == 512
    
    def test_get_sandbox_info_not_exists(self, manager):
        """Test getting info for a non-existing sandbox."""
        info = manager.get_sandbox_info("nonexistent")
        assert info is None
    
    def test_get_sandbox_status_created(self, manager):
        """Test getting status of a created sandbox."""
        manager.create_sandbox("testbox")
        status = manager.get_sandbox_status("testbox")
        assert status == "created"
    
    def test_get_sandbox_status_not_found(self, manager):
        """Test getting status of a non-existing sandbox."""
        status = manager.get_sandbox_status("nonexistent")
        assert status == "not_found"
    
    def test_update_sandbox_config_success(self, manager):
        """Test updating sandbox configuration."""
        manager.create_sandbox("testbox", memory_limit_mb=128)
        
        result = manager.update_sandbox_config(
            "testbox", 
            memory_limit_mb=256, 
            cpu_limit_seconds=60
        )
        
        assert result is True
        assert manager.sandboxes["testbox"]["memory_limit_mb"] == 256
        assert manager.sandboxes["testbox"]["cpu_limit_seconds"] == 60
    
    def test_update_sandbox_config_not_exists(self, manager):
        """Test updating configuration of a non-existing sandbox."""
        result = manager.update_sandbox_config("nonexistent", memory_limit_mb=256)
        assert result is False
    
    def test_get_sandbox_executor_exists(self, manager):
        """Test getting a sandbox executor for an existing sandbox."""
        manager.create_sandbox("testbox")
        executor = manager.get_sandbox("testbox")
        
        assert executor is not None
        assert executor.sandbox_path == manager.sandboxes["testbox"]["path"]
    
    def test_get_sandbox_executor_not_exists(self, manager):
        """Test getting a sandbox executor for a non-existing sandbox."""
        executor = manager.get_sandbox("nonexistent")
        assert executor is None
    
    def test_cleanup_all(self, manager):
        """Test cleaning up all sandboxes."""
        # Create some sandboxes
        manager.create_sandbox("box1")
        manager.create_sandbox("box2")
        
        assert len(manager.sandboxes) == 2
        
        manager.cleanup_all()
        assert len(manager.sandboxes) == 0
    
    def test_create_sandbox_structure(self, manager):
        """Test that sandbox directory structure is created correctly."""
        manager.create_sandbox("testbox")
        sandbox_path = manager.sandboxes["testbox"]["path"]
        
        # Check essential directories exist
        essential_dirs = ['bin', 'lib', 'lib64', 'usr', 'tmp', 'dev', 'proc', 'sys', 'home', 'etc']
        for dir_name in essential_dirs:
            dir_path = os.path.join(sandbox_path, dir_name)
            assert os.path.exists(dir_path), f"Directory {dir_name} should exist"
            assert os.path.isdir(dir_path), f"{dir_name} should be a directory"
        
        # Check essential files exist
        essential_files = ['etc/passwd', 'etc/group', 'etc/hosts', 'etc/resolv.conf']
        for file_name in essential_files:
            file_path = os.path.join(sandbox_path, file_name)
            assert os.path.exists(file_path), f"File {file_name} should exist"
            assert os.path.isfile(file_path), f"{file_name} should be a file"
    
    @patch('json.load')
    def test_load_sandboxes_file_not_exists(self, mock_json_load, temp_dir):
        """Test loading sandboxes when config file doesn't exist."""
        manager = SandboxManager(base_dir=temp_dir)
        assert manager.sandboxes == {}
        mock_json_load.assert_not_called()
    
    @patch('json.dump')
    def test_save_sandboxes(self, mock_json_dump, manager):
        """Test saving sandbox configurations."""
        manager.create_sandbox("testbox")
        manager._save_sandboxes()
        
        # The method might be called multiple times due to internal calls
        assert mock_json_dump.call_count >= 1
        args, kwargs = mock_json_dump.call_args
        assert "testbox" in args[0]
    
    def test_create_sandbox_with_custom_limits(self, manager):
        """Test creating sandbox with custom resource limits."""
        result = manager.create_sandbox(
            "testbox",
            memory_limit_mb=1024,
            cpu_limit_seconds=120,
            block_network=False
        )
        
        assert result is True
        config = manager.sandboxes["testbox"]
        assert config["memory_limit_mb"] == 1024
        assert config["cpu_limit_seconds"] == 120
        assert config["block_network"] is False 