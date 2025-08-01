"""
Shell Task Plugin - Executes shell commands as tasks.
"""

import subprocess
import os
import shlex
from typing import Dict, Any, Tuple


class ShellTaskPlugin:
    """Plugin for executing shell commands."""
    
    def __init__(self):
        self.name = "shell"
        
    def execute(self, task_config: Dict[str, Any]) -> Tuple[str, int]:
        """
        Execute a shell command.
        
        Args:
            task_config: Task configuration dictionary
            
        Returns:
            Tuple of (output, exit_code)
        """
        command = task_config['run']
        cwd = task_config.get('cwd', None)
        env = task_config.get('env', None)
        timeout = task_config.get('timeout', None)
        
        # Parse command if it's a string
        if isinstance(command, str):
            # WARNING: shell=True can be dangerous with untrusted input
            # Ensure commands are validated/sanitized before execution
            # Use shell=True for complex commands
            use_shell = True
            cmd = command
        else:
            # Use shell=False for list of arguments
            use_shell = False
            cmd = command
            
        try:
            # Execute command
            process = subprocess.Popen(
                cmd,
                shell=use_shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                env=env
            )
            
            # Apply timeout to communicate(), not to Popen()
            stdout, stderr = process.communicate(timeout=timeout)
            exit_code = process.returncode
            
            # Combine output
            output = stdout
            if stderr:
                output += f"\nSTDERR:\n{stderr}"
                
            return output, exit_code
            
        except subprocess.TimeoutExpired:
            return "Command timed out", 1
        except Exception as e:
            return f"Command execution failed: {str(e)}", 1
    
    def validate(self, task_config: Dict[str, Any]) -> bool:
        """
        Validate shell task configuration.
        
        Args:
            task_config: Task configuration dictionary
            
        Returns:
            True if valid, False otherwise
        """
        if 'run' not in task_config:
            return False
            
        command = task_config['run']
        if not isinstance(command, (str, list)):
            return False
            
        return True
    
    def get_help(self) -> str:
        """Get help information for shell tasks."""
        return """
Shell Task Configuration:
  run: Command to execute (string or list)
  cwd: Working directory (optional)
  env: Environment variables (optional)
  timeout: Command timeout in seconds (optional)
  
Example:
  run: "echo 'Hello, World!'"
  cwd: "/path/to/working/directory"
  env:
    PATH: "/usr/local/bin:/usr/bin"
  timeout: 30
""" 