"""
Sandbox management functionality.
"""

import os
import shutil
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from .executor import SandboxExecutor

logger = logging.getLogger(__name__)


class SandboxManager:
    """Manages sandbox creation, deletion, and configuration."""
    
    def __init__(self, base_dir: str = "/tmp/chroot-lite"):
        """
        Initialize sandbox manager.
        
        Args:
            base_dir: Base directory for sandboxes
        """
        self.base_dir = base_dir
        self.sandboxes: Dict[str, Dict[str, Any]] = {}
        self.config_file = os.path.join(base_dir, "sandboxes.json")
        self._load_sandboxes()
        
    def create_sandbox(self, name: str, memory_limit_mb: int = 128, 
                      cpu_limit_seconds: int = 30, block_network: bool = True) -> bool:
        """
        Create a new sandbox.
        
        Args:
            name: Name of the sandbox
            memory_limit_mb: Memory limit in megabytes
            cpu_limit_seconds: CPU time limit in seconds
            block_network: Whether to block network access
            
        Returns:
            True if successful, False otherwise
        """
        if name in self.sandboxes:
            logger.error(f"Sandbox '{name}' already exists")
            return False
        
        sandbox_path = os.path.join(self.base_dir, name)
        
        try:
            # Create sandbox directory
            os.makedirs(sandbox_path, exist_ok=True)
            
            # Create basic directory structure
            self._create_sandbox_structure(sandbox_path)
            
            # Create sandbox configuration
            sandbox_config = {
                'name': name,
                'path': sandbox_path,
                'created_at': datetime.now().isoformat(),
                'memory_limit_mb': memory_limit_mb,
                'cpu_limit_seconds': cpu_limit_seconds,
                'block_network': block_network,
                'status': 'created'
            }
            
            self.sandboxes[name] = sandbox_config
            self._save_sandboxes()
            
            logger.info(f"Created sandbox '{name}' at {sandbox_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create sandbox '{name}': {e}")
            # Clean up on failure
            if os.path.exists(sandbox_path):
                shutil.rmtree(sandbox_path, ignore_errors=True)
            return False
    
    def _create_sandbox_structure(self, sandbox_path: str) -> None:
        """
        Create the basic directory structure for a sandbox.
        
        Args:
            sandbox_path: Path to the sandbox directory
        """
        # Create essential directories
        directories = [
            'bin',
            'lib',
            'lib64',
            'usr',
            'usr/bin',
            'usr/lib',
            'usr/lib64',
            'tmp',
            'dev',
            'proc',
            'sys',
            'home',
            'etc'
        ]
        
        for directory in directories:
            os.makedirs(os.path.join(sandbox_path, directory), exist_ok=True)
        
        # Create basic files
        self._create_basic_files(sandbox_path)
    
    def _create_basic_files(self, sandbox_path: str) -> None:
        """
        Create basic files needed for the sandbox.
        
        Args:
            sandbox_path: Path to the sandbox directory
        """
        # Create a basic passwd file
        passwd_content = """root:x:0:0:root:/root:/bin/bash
nobody:x:65534:65534:nobody:/:/bin/false
"""
        with open(os.path.join(sandbox_path, 'etc', 'passwd'), 'w') as f:
            f.write(passwd_content)
        
        # Create a basic group file
        group_content = """root:x:0:
nobody:x:65534:
"""
        with open(os.path.join(sandbox_path, 'etc', 'group'), 'w') as f:
            f.write(group_content)
        
        # Create a basic hosts file
        hosts_content = """127.0.0.1 localhost
::1 localhost ip6-localhost ip6-loopback
"""
        with open(os.path.join(sandbox_path, 'etc', 'hosts'), 'w') as f:
            f.write(hosts_content)
        
        # Create a basic resolv.conf
        resolv_content = """nameserver 8.8.8.8
nameserver 8.8.4.4
"""
        with open(os.path.join(sandbox_path, 'etc', 'resolv.conf'), 'w') as f:
            f.write(resolv_content)
    
    def delete_sandbox(self, name: str) -> bool:
        """
        Delete a sandbox.
        
        Args:
            name: Name of the sandbox to delete
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.sandboxes:
            logger.error(f"Sandbox '{name}' does not exist")
            return False
        
        sandbox_config = self.sandboxes[name]
        sandbox_path = sandbox_config['path']
        
        try:
            # Remove sandbox directory
            if os.path.exists(sandbox_path):
                shutil.rmtree(sandbox_path)
            
            # Remove from configuration
            del self.sandboxes[name]
            self._save_sandboxes()
            
            logger.info(f"Deleted sandbox '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete sandbox '{name}': {e}")
            return False
    
    def get_sandbox(self, name: str) -> Optional[SandboxExecutor]:
        """
        Get a sandbox executor for a sandbox.
        
        Args:
            name: Name of the sandbox
            
        Returns:
            SandboxExecutor instance or None if not found
        """
        if name not in self.sandboxes:
            logger.error(f"Sandbox '{name}' does not exist")
            return None
        
        sandbox_config = self.sandboxes[name]
        sandbox_path = sandbox_config['path']
        
        if not os.path.exists(sandbox_path):
            logger.error(f"Sandbox directory for '{name}' does not exist")
            return None
        
        return SandboxExecutor(
            sandbox_path=sandbox_path,
            memory_limit_mb=sandbox_config.get('memory_limit_mb', 128),
            cpu_limit_seconds=sandbox_config.get('cpu_limit_seconds', 30),
            block_network=sandbox_config.get('block_network', True)
        )
    
    def list_sandboxes(self) -> List[Dict[str, Any]]:
        """
        List all sandboxes.
        
        Returns:
            List of sandbox configurations
        """
        return list(self.sandboxes.values())
    
    def get_sandbox_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific sandbox.
        
        Args:
            name: Name of the sandbox
            
        Returns:
            Sandbox configuration or None if not found
        """
        return self.sandboxes.get(name)
    
    def update_sandbox_config(self, name: str, **kwargs) -> bool:
        """
        Update sandbox configuration.
        
        Args:
            name: Name of the sandbox
            **kwargs: Configuration parameters to update
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.sandboxes:
            logger.error(f"Sandbox '{name}' does not exist")
            return False
        
        try:
            # Update configuration
            for key, value in kwargs.items():
                if key in ['memory_limit_mb', 'cpu_limit_seconds', 'block_network']:
                    self.sandboxes[name][key] = value
            
            self._save_sandboxes()
            logger.info(f"Updated configuration for sandbox '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update sandbox '{name}': {e}")
            return False
    
    def _load_sandboxes(self) -> None:
        """Load sandbox configurations from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.sandboxes = json.load(f)
                logger.info(f"Loaded {len(self.sandboxes)} sandbox configurations")
        except Exception as e:
            logger.error(f"Failed to load sandbox configurations: {e}")
            self.sandboxes = {}
    
    def _save_sandboxes(self) -> None:
        """Save sandbox configurations to file."""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.sandboxes, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save sandbox configurations: {e}")
    
    def cleanup_all(self) -> None:
        """Clean up all sandboxes."""
        for name in list(self.sandboxes.keys()):
            self.delete_sandbox(name)
        
        logger.info("Cleaned up all sandboxes")
    
    def get_sandbox_status(self, name: str) -> str:
        """
        Get the status of a sandbox.
        
        Args:
            name: Name of the sandbox
            
        Returns:
            Status string
        """
        if name not in self.sandboxes:
            return "not_found"
        
        sandbox_config = self.sandboxes[name]
        sandbox_path = sandbox_config['path']
        
        if not os.path.exists(sandbox_path):
            return "corrupted"
        
        return sandbox_config.get('status', 'unknown') 