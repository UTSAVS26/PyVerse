"""
Secure code execution within sandboxed environments.
"""

import os
import sys
import subprocess
import signal
import time
import tempfile
import shutil
from typing import Dict, Any, Optional, Tuple
import logging
from .limiter import ResourceLimiter
from .firewall import NetworkFirewall

# Platform-specific imports
try:
    import os
    CHROOT_AVAILABLE = hasattr(os, 'chroot')
    FORK_AVAILABLE = hasattr(os, 'fork')
    WEXITSTATUS_AVAILABLE = hasattr(os, 'WEXITSTATUS')
except AttributeError:
    CHROOT_AVAILABLE = False
    FORK_AVAILABLE = False
    WEXITSTATUS_AVAILABLE = False

# Platform-specific signal constants
try:
    SIGTERM = signal.SIGTERM
    SIGKILL = signal.SIGKILL
except AttributeError:
    SIGTERM = 15
    SIGKILL = 9

logger = logging.getLogger(__name__)


class SandboxExecutor:
    """Executes code securely within a sandboxed environment."""
    
    def __init__(self, sandbox_path: str, memory_limit_mb: int = 128, 
                 cpu_limit_seconds: int = 30, block_network: bool = True):
        """
        Initialize sandbox executor.
        
        Args:
            sandbox_path: Path to the sandbox directory
            memory_limit_mb: Memory limit in megabytes
            cpu_limit_seconds: CPU time limit in seconds
            block_network: Whether to block network access
        """
        self.sandbox_path = sandbox_path
        self.limiter = ResourceLimiter(memory_limit_mb, cpu_limit_seconds)
        self.firewall = NetworkFirewall(block_network)
        self.running_processes: Dict[int, Dict[str, Any]] = {}
        
    def execute_script(self, script_path: str, args: Optional[list] = None, 
                      timeout: int = 60) -> Dict[str, Any]:
        """
        Execute a script within the sandbox.
        
        Args:
            script_path: Path to the script to execute
            args: Additional arguments to pass to the script
            timeout: Maximum execution time in seconds
            
        Returns:
            Dictionary with execution results
        """
        if args is None:
            args = []
            
        # Copy script to sandbox if it's outside
        sandbox_script_path = self._copy_script_to_sandbox(script_path)
        
        # Prepare command
        if script_path.endswith('.py'):
            cmd = ['python', sandbox_script_path] + args
        else:
            cmd = ['/bin/bash', sandbox_script_path] + args
            
        logger.info(f"Executing command in sandbox: {' '.join(cmd)}")
        
        # Execute with resource limits
        result = self._execute_with_limits(cmd, timeout)
        
        # Clean up
        if sandbox_script_path != script_path:
            try:
                os.remove(sandbox_script_path)
            except OSError:
                pass
                
        return result
    
    def _copy_script_to_sandbox(self, script_path: str) -> str:
        """
        Copy a script to the sandbox directory.
        
        Args:
            script_path: Path to the original script
            
        Returns:
            Path to the script within the sandbox
        """
        if script_path.startswith(self.sandbox_path):
            return script_path
            
        # Create a temporary file in the sandbox
        script_name = os.path.basename(script_path)
        sandbox_script_path = os.path.join(self.sandbox_path, 'tmp', script_name)
        
        # Ensure tmp directory exists
        tmp_dir = os.path.join(self.sandbox_path, 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Copy the script
        shutil.copy2(script_path, sandbox_script_path)
        
        # Make it executable
        os.chmod(sandbox_script_path, 0o755)
        
        return sandbox_script_path
    
    def _execute_with_limits(self, cmd: list, timeout: int) -> Dict[str, Any]:
        """
        Execute a command with resource limits.
        
        Args:
            cmd: Command to execute
            timeout: Maximum execution time
            
        Returns:
            Dictionary with execution results
        """
        start_time = time.time()
        
        if not FORK_AVAILABLE:
            # Fallback for platforms without fork
            logger.warning("Fork not available on this platform, using subprocess")
            return self._execute_with_subprocess(cmd, timeout, start_time)
        
        try:
            # Fork a child process
            pid = os.fork()
            
            if pid == 0:
                # Child process
                self._setup_child_environment(cmd)
            else:
                # Parent process
                return self._monitor_child_process(pid, cmd, timeout, start_time)
                
        except OSError as e:
            logger.error(f"Failed to fork process: {e}")
            return {
                'success': False,
                'error': f'Failed to fork process: {e}',
                'return_code': -1,
                'stdout': '',
                'stderr': '',
                'execution_time': 0
            }
    
    def _setup_child_environment(self, cmd: list) -> None:
        """
        Set up the child process environment.
        
        Args:
            cmd: Command to execute
        """
        try:
            # Change to sandbox directory
            os.chdir(self.sandbox_path)
            
            # Change root to sandbox (only if available)
            if CHROOT_AVAILABLE:
                os.chroot(self.sandbox_path)
            else:
                logger.warning("Chroot not available on this platform, using directory isolation only")
            
            # Set resource limits
            self.limiter.set_limits(os.getpid())
            
            # Block network access if enabled
            if self.firewall.block_network:
                self.firewall.block_process_network(os.getpid())
            
            # Execute the command
            os.execvp(cmd[0], cmd)
            
        except Exception as e:
            logger.error(f"Child process setup failed: {e}")
            os._exit(1)
    
    def _monitor_child_process(self, pid: int, cmd: list, timeout: int, 
                              start_time: float) -> Dict[str, Any]:
        """
        Monitor the child process execution.
        
        Args:
            pid: Child process ID
            cmd: Command being executed
            timeout: Maximum execution time
            start_time: Start time of execution
            
        Returns:
            Dictionary with execution results
        """
        self.running_processes[pid] = {
            'cmd': cmd,
            'start_time': start_time,
            'status': 'running'
        }
        
        try:
            # Wait for process to complete
            while True:
                # Check if process is still running
                try:
                    os.waitpid(pid, os.WNOHANG)
                    break
                except ChildProcessError:
                    break
                
                # Check resource limits
                if self.limiter.terminate_if_exceeded(pid):
                    break
                
                # Check timeout
                if time.time() - start_time > timeout:
                    os.kill(pid, SIGTERM)
                    time.sleep(1)
                    try:
                        os.kill(pid, SIGKILL)
                    except ProcessLookupError:
                        pass
                    break
                
                time.sleep(0.1)
            
            # Get final status
            try:
                pid_result, status = os.waitpid(pid, 0)
                if WEXITSTATUS_AVAILABLE:
                    return_code = os.WEXITSTATUS(status)
                else:
                    return_code = 0  # Default for platforms without WEXITSTATUS
            except ChildProcessError:
                return_code = -1
            
            execution_time = time.time() - start_time
            
            # Clean up
            if pid in self.running_processes:
                del self.running_processes[pid]
            
            # Unblock network access
            self.firewall.unblock_process_network(pid)
            
            return {
                'success': return_code == 0,
                'return_code': return_code,
                'execution_time': execution_time,
                'stdout': '',  # Would need to capture stdout/stderr
                'stderr': '',
                'pid': pid
            }
            
        except Exception as e:
            logger.error(f"Error monitoring process {pid}: {e}")
            
            # Clean up
            if pid in self.running_processes:
                del self.running_processes[pid]
            
            self.firewall.unblock_process_network(pid)
            
            return {
                'success': False,
                'error': str(e),
                'return_code': -1,
                'execution_time': time.time() - start_time,
                'stdout': '',
                'stderr': '',
                'pid': pid
            }
    
    def execute_python_code(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Execute Python code within the sandbox.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time
            
        Returns:
            Dictionary with execution results
        """
        # Create temporary file with the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', 
                                        dir=self.sandbox_path, delete=False) as f:
            f.write(code)
            script_path = f.name
        
        try:
            return self.execute_script(script_path, timeout=timeout)
        finally:
            # Clean up
            try:
                os.remove(script_path)
            except OSError:
                pass
    
    def get_running_processes(self) -> Dict[int, Dict[str, Any]]:
        """
        Get information about currently running processes.
        
        Returns:
            Dictionary of running processes
        """
        return self.running_processes.copy()
    
    def terminate_process(self, pid: int) -> bool:
        """
        Terminate a running process.
        
        Args:
            pid: Process ID to terminate
            
        Returns:
            True if successful, False otherwise
        """
        if pid not in self.running_processes:
            return False
        
        try:
            os.kill(pid, SIGTERM)
            time.sleep(1)
            
            # Force kill if still running
            try:
                os.kill(pid, SIGKILL)
            except ProcessLookupError:
                pass
            
            # Clean up
            if pid in self.running_processes:
                del self.running_processes[pid]
            
            self.firewall.unblock_process_network(pid)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to terminate process {pid}: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up all running processes and resources."""
        # Terminate all running processes
        for pid in list(self.running_processes.keys()):
            self.terminate_process(pid)
        
        # Clean up firewall
        self.firewall.cleanup()
        
        logger.info("Cleaned up sandbox executor")
    
    def _execute_with_subprocess(self, cmd: list, timeout: int, start_time: float) -> Dict[str, Any]:
        """
        Execute command using subprocess (fallback for platforms without fork).
        
        Args:
            cmd: Command to execute
            timeout: Maximum execution time
            start_time: Start time of execution
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Execute command using subprocess
            result = subprocess.run(
                cmd,
                cwd=self.sandbox_path,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            return {
                'success': result.returncode == 0,
                'return_code': result.returncode,
                'execution_time': execution_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'pid': result.pid
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Execution timeout',
                'return_code': -1,
                'execution_time': time.time() - start_time,
                'stdout': '',
                'stderr': 'Execution timeout'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'return_code': -1,
                'execution_time': time.time() - start_time,
                'stdout': '',
                'stderr': str(e)
            } 