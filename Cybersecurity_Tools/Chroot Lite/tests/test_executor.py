"""
Unit tests for SandboxExecutor.
"""

import pytest
import os
import tempfile
import shutil
import time
from unittest.mock import patch, MagicMock, mock_open
from sandbox.executor import SandboxExecutor


class TestSandboxExecutor:
    """Test cases for SandboxExecutor."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def executor(self, temp_dir):
        """Create a SandboxExecutor instance for testing."""
        return SandboxExecutor(
            sandbox_path=temp_dir,
            memory_limit_mb=128,
            cpu_limit_seconds=30,
            block_network=True
        )
    
    def test_init(self, temp_dir):
        """Test SandboxExecutor initialization."""
        executor = SandboxExecutor(
            sandbox_path=temp_dir,
            memory_limit_mb=256,
            cpu_limit_seconds=60,
            block_network=False
        )
        
        assert executor.sandbox_path == temp_dir
        assert executor.limiter.memory_limit_mb == 256
        assert executor.limiter.cpu_limit_seconds == 60
        assert executor.firewall.block_network is False
        assert executor.running_processes == {}
    
    def test_copy_script_to_sandbox_already_in_sandbox(self, executor):
        """Test copying script when it's already in sandbox."""
        script_path = os.path.join(executor.sandbox_path, "test.py")
        
        result = executor._copy_script_to_sandbox(script_path)
        
        assert result == script_path
    
    @patch('shutil.copy2')
    @patch('os.chmod')
    def test_copy_script_to_sandbox_from_outside(self, mock_chmod, mock_copy2, executor):
        """Test copying script from outside sandbox."""
        # Create tmp directory
        tmp_dir = os.path.join(executor.sandbox_path, 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        
        external_script = "/tmp/external_script.py"
        expected_sandbox_path = os.path.join(tmp_dir, "external_script.py")
        
        result = executor._copy_script_to_sandbox(external_script)
        
        assert result == expected_sandbox_path
        mock_copy2.assert_called_once_with(external_script, expected_sandbox_path)
        mock_chmod.assert_called_once_with(expected_sandbox_path, 0o755)
    
    @patch('os.fork')
    def test_execute_with_limits_fork_error(self, mock_fork, executor):
        """Test execution when fork fails."""
        if not hasattr(os, 'fork'):
            pytest.skip("Fork not available on this platform")
        
        mock_fork.side_effect = OSError("Fork failed")
        
        result = executor._execute_with_limits(['python', 'test.py'], 60)
        
        assert result['success'] is False
        assert 'Failed to fork process' in result['error']
        assert result['return_code'] == -1
    
    @patch('os.fork')
    @patch('os.waitpid')
    @patch('os.WEXITSTATUS')
    def test_execute_with_limits_success(self, mock_wexitstatus, mock_waitpid, mock_fork, executor):
        """Test successful execution."""
        if not hasattr(os, 'fork'):
            pytest.skip("Fork not available on this platform")
        
        mock_fork.return_value = 12345  # Child PID
        mock_waitpid.return_value = (12345, 0)
        mock_wexitstatus.return_value = 0
        
        # Mock the child process to exit immediately
        with patch.object(executor, '_setup_child_environment') as mock_setup:
            mock_setup.side_effect = SystemExit(0)
            
            result = executor._execute_with_limits(['python', 'test.py'], 60)
        
        assert result['success'] is True
        assert result['return_code'] == 0
        assert result['pid'] == 12345
    
    @patch('os.fork')
    @patch('os.waitpid')
    @patch('os.WEXITSTATUS')
    def test_execute_with_limits_failure(self, mock_wexitstatus, mock_waitpid, mock_fork, executor):
        """Test execution with non-zero return code."""
        if not hasattr(os, 'fork'):
            pytest.skip("Fork not available on this platform")
        
        mock_fork.return_value = 12345  # Child PID
        mock_waitpid.return_value = (12345, 0)
        mock_wexitstatus.return_value = 1
        
        # Mock the child process to exit immediately
        with patch.object(executor, '_setup_child_environment') as mock_setup:
            mock_setup.side_effect = SystemExit(1)
            
            result = executor._execute_with_limits(['python', 'test.py'], 60)
        
        assert result['success'] is False
        assert result['return_code'] == 1
        assert result['pid'] == 12345
    
    @patch('os.chdir')
    @patch('os.chroot')
    @patch('os.execvp')
    def test_setup_child_environment_success(self, mock_execvp, mock_chroot, mock_chdir, executor):
        """Test successful child environment setup."""
        cmd = ['python', 'test.py']
        
        with patch.object(executor.limiter, 'set_limits') as mock_set_limits:
            with patch.object(executor.firewall, 'block_process_network') as mock_block_network:
                executor._setup_child_environment(cmd)
        
        mock_chdir.assert_called_once_with(executor.sandbox_path)
        if hasattr(os, 'chroot'):
            mock_chroot.assert_called_once_with(executor.sandbox_path)
        mock_set_limits.assert_called_once_with(os.getpid())
        mock_block_network.assert_called_once_with(os.getpid())
        mock_execvp.assert_called_once_with('python', cmd)
    
    @patch('os.chdir')
    @patch('os.chroot')
    def test_setup_child_environment_exception(self, mock_chroot, mock_chdir, executor):
        """Test child environment setup when exception occurs."""
        if not hasattr(os, 'chroot'):
            pytest.skip("Chroot not available on this platform")
        
        mock_chroot.side_effect = Exception("Chroot failed")
        
        with patch('os._exit') as mock_exit:
            executor._setup_child_environment(['python', 'test.py'])
        
        mock_exit.assert_called_once_with(1)
    
    @patch('os.fork')
    def test_monitor_child_process_success(self, mock_fork, executor):
        """Test successful child process monitoring."""
        if not hasattr(os, 'fork'):
            pytest.skip("Fork not available on this platform")
        
        mock_fork.return_value = 12345  # Child PID
        
        with patch.object(executor, '_setup_child_environment') as mock_setup:
            mock_setup.side_effect = SystemExit(0)
            
            with patch('os.waitpid') as mock_waitpid:
                mock_waitpid.return_value = (12345, 0)
                
                with patch('os.WEXITSTATUS') as mock_wexitstatus:
                    mock_wexitstatus.return_value = 0
                    
                    result = executor._monitor_child_process(12345, ['python', 'test.py'], 60, time.time())
        
        assert result['success'] is True
        assert result['return_code'] == 0
        assert result['pid'] == 12345
        assert 12345 not in executor.running_processes
    
    @patch('os.fork')
    def test_monitor_child_process_timeout(self, mock_fork, executor):
        """Test child process monitoring with timeout."""
        if not hasattr(os, 'fork'):
            pytest.skip("Fork not available on this platform")
        
        mock_fork.return_value = 12345  # Child PID
        
        with patch.object(executor, '_setup_child_environment') as mock_setup:
            mock_setup.side_effect = SystemExit(0)
            
            with patch('os.waitpid') as mock_waitpid:
                mock_waitpid.side_effect = ChildProcessError()
                
                with patch('os.kill') as mock_kill:
                    with patch('time.time') as mock_time:
                        mock_time.return_value = time.time() + 70  # Exceed timeout
                        
                        result = executor._monitor_child_process(12345, ['python', 'test.py'], 60, time.time())
        
        assert result['success'] is False
        mock_kill.assert_called()
    
    @patch('os.fork')
    def test_monitor_child_process_resource_limit_exceeded(self, mock_fork, executor):
        """Test child process monitoring when resource limits are exceeded."""
        if not hasattr(os, 'fork'):
            pytest.skip("Fork not available on this platform")
        
        mock_fork.return_value = 12345  # Child PID
        
        with patch.object(executor, '_setup_child_environment') as mock_setup:
            mock_setup.side_effect = SystemExit(0)
            
            with patch.object(executor.limiter, 'terminate_if_exceeded') as mock_terminate:
                mock_terminate.return_value = True
                
                with patch('os.waitpid') as mock_waitpid:
                    mock_waitpid.side_effect = ChildProcessError()
                    
                    result = executor._monitor_child_process(12345, ['python', 'test.py'], 60, time.time())
        
        assert result['success'] is False
        mock_terminate.assert_called_with(12345)
    
    def test_execute_script_python(self, executor):
        """Test executing a Python script."""
        script_path = "test.py"
        args = ["arg1", "arg2"]
        
        with patch.object(executor, '_copy_script_to_sandbox') as mock_copy:
            mock_copy.return_value = os.path.join(executor.sandbox_path, "test.py")
            
            with patch.object(executor, '_execute_with_limits') as mock_execute:
                mock_execute.return_value = {
                    'success': True,
                    'return_code': 0,
                    'execution_time': 1.5,
                    'stdout': 'Hello World',
                    'stderr': '',
                    'pid': 12345
                }
                
                result = executor.execute_script(script_path, args)
        
        assert result['success'] is True
        assert result['return_code'] == 0
        assert result['execution_time'] == 1.5
    
    def test_execute_script_shell(self, executor):
        """Test executing a shell script."""
        script_path = "test.sh"
        
        with patch.object(executor, '_copy_script_to_sandbox') as mock_copy:
            mock_copy.return_value = os.path.join(executor.sandbox_path, "test.sh")
            
            with patch.object(executor, '_execute_with_limits') as mock_execute:
                mock_execute.return_value = {
                    'success': True,
                    'return_code': 0,
                    'execution_time': 1.0,
                    'stdout': 'Script output',
                    'stderr': '',
                    'pid': 12345
                }
                
                result = executor.execute_script(script_path)
        
        assert result['success'] is True
        assert result['return_code'] == 0
    
    def test_execute_python_code(self, executor):
        """Test executing Python code."""
        code = "print('Hello World')\nprint(2 + 2)"
        
        with patch.object(executor, 'execute_script') as mock_execute:
            mock_execute.return_value = {
                'success': True,
                'return_code': 0,
                'execution_time': 0.5,
                'stdout': 'Hello World\n4',
                'stderr': '',
                'pid': 12345
            }
            
            result = executor.execute_python_code(code)
        
        assert result['success'] is True
        assert result['return_code'] == 0
        mock_execute.assert_called_once()
    
    def test_get_running_processes(self, executor):
        """Test getting running processes."""
        executor.running_processes = {
            12345: {'cmd': ['python', 'test.py'], 'start_time': time.time(), 'status': 'running'},
            12346: {'cmd': ['bash', 'script.sh'], 'start_time': time.time(), 'status': 'running'}
        }
        
        processes = executor.get_running_processes()
        
        assert len(processes) == 2
        assert 12345 in processes
        assert 12346 in processes
    
    @patch('os.kill')
    def test_terminate_process_success(self, mock_kill, executor):
        """Test successful process termination."""
        executor.running_processes[12345] = {
            'cmd': ['python', 'test.py'],
            'start_time': time.time(),
            'status': 'running'
        }
        
        with patch.object(executor.firewall, 'unblock_process_network') as mock_unblock:
            result = executor.terminate_process(12345)
        
        assert result is True
        assert 12345 not in executor.running_processes
        # Check that kill was called, but don't check the specific signal number
        # since it varies by platform
        mock_kill.assert_called()
        mock_unblock.assert_called_with(12345)
    
    def test_terminate_process_not_running(self, executor):
        """Test terminating a process that's not running."""
        result = executor.terminate_process(12345)
        assert result is False
    
    @patch('os.kill')
    def test_terminate_process_kill_error(self, mock_kill, executor):
        """Test terminating process when kill fails."""
        executor.running_processes[12345] = {
            'cmd': ['python', 'test.py'],
            'start_time': time.time(),
            'status': 'running'
        }
        
        mock_kill.side_effect = Exception("Kill failed")
        
        result = executor.terminate_process(12345)
        
        assert result is False
    
    def test_cleanup(self, executor):
        """Test cleanup of executor."""
        executor.running_processes[12345] = {
            'cmd': ['python', 'test.py'],
            'start_time': time.time(),
            'status': 'running'
        }
        
        with patch.object(executor, 'terminate_process') as mock_terminate:
            with patch.object(executor.firewall, 'cleanup') as mock_firewall_cleanup:
                executor.cleanup()
        
        mock_terminate.assert_called_with(12345)
        mock_firewall_cleanup.assert_called_once()
    
    def test_execute_script_cleanup_on_failure(self, executor):
        """Test that script execution cleans up on failure."""
        script_path = "test.py"
        
        with patch.object(executor, '_copy_script_to_sandbox') as mock_copy:
            mock_copy.return_value = os.path.join(executor.sandbox_path, "test.py")
            
            with patch.object(executor, '_execute_with_limits') as mock_execute:
                mock_execute.return_value = {
                    'success': False,
                    'return_code': 1,
                    'execution_time': 0.1,
                    'stdout': '',
                    'stderr': 'Error occurred',
                    'pid': 12345
                }
                
                with patch('os.remove') as mock_remove:
                    result = executor.execute_script(script_path)
        
        assert result['success'] is False
        assert result['return_code'] == 1 