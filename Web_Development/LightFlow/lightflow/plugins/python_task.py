"""
Python Task Plugin - Executes Python code as tasks.
"""

import sys
import os
from io import StringIO
from typing import Dict, Any, Tuple


class PythonTaskPlugin:
    """Plugin for executing Python code."""
    
    def __init__(self):
        self.name = "python"
        
    def execute(self, task_config: Dict[str, Any]) -> Tuple[str, int]:
        """
        Execute Python code.
        
        Args:
            task_config: Task configuration dictionary
            
        Returns:
            Tuple of (output, exit_code)
        """
        code = task_config['run']
        cwd = task_config.get('cwd', None)
        
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        output_buffer = StringIO()
        error_buffer = StringIO()
        
        try:
            # Change working directory if specified
            if cwd:
                old_cwd = os.getcwd()
                os.chdir(cwd)
            
            # Redirect stdout and stderr
            sys.stdout = output_buffer
            sys.stderr = error_buffer
            
            # Execute Python code
            exec(code, {})
            
            # Get output
            output = output_buffer.getvalue()
            error_output = error_buffer.getvalue()
            
            if error_output:
                output += f"\nSTDERR:\n{error_output}"
                
            return output, 0
            
        except Exception as e:
            # Get any output before the error
            output = output_buffer.getvalue()
            error_output = error_buffer.getvalue()
            
            if error_output:
                output += f"\nSTDERR:\n{error_output}"
                
            output += f"\nError: {str(e)}"
            return output, 1
            
        finally:
            # Restore stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Restore working directory
            if cwd:
                os.chdir(old_cwd)
    
    def validate(self, task_config: Dict[str, Any]) -> bool:
        """
        Validate Python task configuration.
        
        Args:
            task_config: Task configuration dictionary
            
        Returns:
            True if valid, False otherwise
        """
        if 'run' not in task_config:
            return False
            
        code = task_config['run']
        if not isinstance(code, str):
            return False
            
        return True
    
    def get_help(self) -> str:
        """Get help information for Python tasks."""
        return """
Python Task Configuration:
  run: Python code to execute (string)
  cwd: Working directory (optional)
  
Example:
  run: |
    import os
    print("Current directory:", os.getcwd())
    print("Hello from Python!")
  cwd: "/path/to/working/directory"
""" 