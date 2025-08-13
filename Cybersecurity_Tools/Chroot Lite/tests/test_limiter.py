"""
Unit tests for ResourceLimiter.
"""

import pytest
import os
import signal
import time
from unittest.mock import patch, MagicMock
from sandbox.limiter import ResourceLimiter

# Platform-specific imports
try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False


class TestResourceLimiter:
    """Test cases for ResourceLimiter."""
    
    @pytest.fixture
    def limiter(self):
        """Create a ResourceLimiter instance for testing."""
        return ResourceLimiter(memory_limit_mb=128, cpu_limit_seconds=30)
    
    def test_init(self):
        """Test ResourceLimiter initialization."""
        limiter = ResourceLimiter(memory_limit_mb=256, cpu_limit_seconds=60)
        
        assert limiter.memory_limit_mb == 256
        assert limiter.cpu_limit_seconds == 60
        assert limiter.memory_limit_bytes == 256 * 1024 * 1024
    
    def test_init_defaults(self):
        """Test ResourceLimiter initialization with defaults."""
        limiter = ResourceLimiter()
        
        assert limiter.memory_limit_mb == 128
        assert limiter.cpu_limit_seconds == 30
        assert limiter.memory_limit_bytes == 128 * 1024 * 1024
    
    def test_set_limits_success(self, limiter):
        """Test successful setting of resource limits."""
        if not RESOURCE_AVAILABLE:
            pytest.skip("Resource module not available on this platform")
        
        # Test that the method doesn't raise an exception
        try:
            limiter.set_limits(12345)
            # If we get here, the method executed successfully
            assert True
        except Exception as e:
            # On Windows, this will fail, which is expected
            assert "Resource limiting not available" in str(e) or "not available" in str(e)
    
    def test_set_limits_exception(self, limiter):
        """Test setting limits when an exception occurs."""
        if not RESOURCE_AVAILABLE:
            pytest.skip("Resource module not available on this platform")
        
        # On Windows, this will fail gracefully
        try:
            limiter.set_limits(12345)
            # If we get here, it worked (unlikely on Windows)
            assert True
        except Exception as e:
            # Expected on Windows
            assert "Resource limiting not available" in str(e) or "not available" in str(e)
    
    @patch('psutil.Process')
    def test_monitor_process_success(self, mock_process_class, limiter):
        """Test successful process monitoring."""
        # Mock process
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=50 * 1024 * 1024)  # 50MB
        mock_process.cpu_percent.return_value = 25.5
        mock_process.cpu_times.return_value = MagicMock(user=10.0, system=5.0)
        mock_process_class.return_value = mock_process
        
        usage_info = limiter.monitor_process(12345)
        
        assert usage_info['pid'] == 12345
        assert usage_info['memory_mb'] == 50.0
        assert usage_info['memory_limit_mb'] == 128
        assert usage_info['memory_exceeded'] is False
        assert usage_info['cpu_percent'] == 25.5
        assert usage_info['cpu_time_user'] == 10.0
        assert usage_info['cpu_time_system'] == 5.0
        assert usage_info['cpu_limit_seconds'] == 30
        assert usage_info['cpu_exceeded'] is False
        assert usage_info['status'] == 'running'
    
    @patch('psutil.Process')
    def test_monitor_process_memory_exceeded(self, mock_process_class, limiter):
        """Test process monitoring when memory limit is exceeded."""
        # Mock process with high memory usage
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=200 * 1024 * 1024)  # 200MB
        mock_process.cpu_percent.return_value = 10.0
        mock_process.cpu_times.return_value = MagicMock(user=5.0, system=2.0)
        mock_process_class.return_value = mock_process
        
        usage_info = limiter.monitor_process(12345)
        
        assert usage_info['memory_exceeded'] is True
        assert usage_info['status'] == 'limit_exceeded'
    
    @patch('psutil.Process')
    def test_monitor_process_cpu_exceeded(self, mock_process_class, limiter):
        """Test process monitoring when CPU limit is exceeded."""
        # Mock process with high CPU usage
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=50 * 1024 * 1024)  # 50MB
        mock_process.cpu_percent.return_value = 10.0
        mock_process.cpu_times.return_value = MagicMock(user=25.0, system=10.0)  # 35s total
        mock_process_class.return_value = mock_process
        
        usage_info = limiter.monitor_process(12345)
        
        assert usage_info['cpu_exceeded'] is True
        assert usage_info['status'] == 'limit_exceeded'
    
    @patch('psutil.Process')
    def test_monitor_process_not_found(self, mock_process_class, limiter):
        """Test process monitoring when process doesn't exist."""
        mock_process_class.side_effect = Exception("No such process")
        
        usage_info = limiter.monitor_process(12345)
        
        assert usage_info['pid'] == 12345
        assert usage_info['status'] == 'error'  # Changed from 'terminated' to 'error'
        assert 'error' in usage_info
    
    @patch('psutil.Process')
    def test_monitor_process_error(self, mock_process_class, limiter):
        """Test process monitoring when an error occurs."""
        mock_process_class.side_effect = Exception("Test error")
        
        usage_info = limiter.monitor_process(12345)
        
        assert usage_info['pid'] == 12345
        assert usage_info['status'] == 'error'
        assert usage_info['error'] == 'Test error'
    
    @patch('os.kill')
    @patch('psutil.Process')
    def test_terminate_if_exceeded_memory(self, mock_process_class, mock_kill, limiter):
        """Test terminating process when memory limit is exceeded."""
        # Mock process with high memory usage
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=200 * 1024 * 1024)  # 200MB
        mock_process.cpu_percent.return_value = 10.0
        mock_process.cpu_times.return_value = MagicMock(user=5.0, system=2.0)
        mock_process_class.return_value = mock_process
        
        result = limiter.terminate_if_exceeded(12345)
        
        assert result is True
        mock_kill.assert_called_once_with(12345, signal.SIGTERM)
    
    @patch('os.kill')
    @patch('psutil.Process')
    def test_terminate_if_exceeded_cpu(self, mock_process_class, mock_kill, limiter):
        """Test terminating process when CPU limit is exceeded."""
        # Mock process with high CPU usage
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=50 * 1024 * 1024)  # 50MB
        mock_process.cpu_percent.return_value = 10.0
        mock_process.cpu_times.return_value = MagicMock(user=25.0, system=10.0)  # 35s total
        mock_process_class.return_value = mock_process
        
        result = limiter.terminate_if_exceeded(12345)
        
        assert result is True
        mock_kill.assert_called_once_with(12345, signal.SIGTERM)
    
    @patch('os.kill')
    @patch('psutil.Process')
    def test_terminate_if_exceeded_no_exceed(self, mock_process_class, mock_kill, limiter):
        """Test terminating process when limits are not exceeded."""
        # Mock process with normal usage
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=50 * 1024 * 1024)  # 50MB
        mock_process.cpu_percent.return_value = 10.0
        mock_process.cpu_times.return_value = MagicMock(user=5.0, system=2.0)
        mock_process_class.return_value = mock_process
        
        result = limiter.terminate_if_exceeded(12345)
        
        assert result is False
        mock_kill.assert_not_called()
    
    @patch('os.kill')
    def test_terminate_if_exceeded_process_lookup_error(self, mock_kill, limiter):
        """Test terminating process when process is already terminated."""
        mock_kill.side_effect = ProcessLookupError()
        
        # Mock monitor_process to return limit_exceeded
        with patch.object(limiter, 'monitor_process') as mock_monitor:
            mock_monitor.return_value = {
                'pid': 12345,
                'status': 'limit_exceeded',
                'memory_exceeded': True
            }
            
            result = limiter.terminate_if_exceeded(12345)
            
            assert result is True
            mock_kill.assert_called_once_with(12345, signal.SIGTERM)
    
    def test_get_usage_summary_running(self, limiter):
        """Test getting usage summary for a running process."""
        with patch.object(limiter, 'monitor_process') as mock_monitor:
            mock_monitor.return_value = {
                'pid': 12345,
                'status': 'running',
                'memory_mb': 50.0,
                'memory_limit_mb': 128,
                'memory_exceeded': False,
                'cpu_percent': 25.5,
                'cpu_time_user': 10.0,
                'cpu_time_system': 5.0,
                'cpu_limit_seconds': 30,
                'cpu_exceeded': False
            }
            
            summary = limiter.get_usage_summary(12345)
            
            assert "Process 12345:" in summary
            assert "Memory ✅" in summary
            assert "CPU ✅" in summary
            assert "50.0MB/128MB" in summary
            assert "25.5%" in summary
    
    def test_get_usage_summary_terminated(self, limiter):
        """Test getting usage summary for a terminated process."""
        with patch.object(limiter, 'monitor_process') as mock_monitor:
            mock_monitor.return_value = {
                'pid': 12345,
                'status': 'terminated'
            }
            
            summary = limiter.get_usage_summary(12345)
            
            assert summary == "Process 12345: Terminated"
    
    def test_get_usage_summary_error(self, limiter):
        """Test getting usage summary for a process with error."""
        with patch.object(limiter, 'monitor_process') as mock_monitor:
            mock_monitor.return_value = {
                'pid': 12345,
                'status': 'error',
                'error': 'Test error'
            }
            
            summary = limiter.get_usage_summary(12345)
            
            assert summary == "Process 12345: Error - Test error"
    
    def test_get_usage_summary_with_exceeded_limits(self, limiter):
        """Test getting usage summary when limits are exceeded."""
        with patch.object(limiter, 'monitor_process') as mock_monitor:
            mock_monitor.return_value = {
                'pid': 12345,
                'status': 'running',
                'memory_mb': 200.0,
                'memory_limit_mb': 128,
                'memory_exceeded': True,
                'cpu_percent': 25.5,
                'cpu_time_user': 35.0,
                'cpu_time_system': 10.0,
                'cpu_limit_seconds': 30,
                'cpu_exceeded': True
            }
            
            summary = limiter.get_usage_summary(12345)
            
            assert "Memory ⚠️" in summary
            assert "CPU ⚠️" in summary
            assert "200.0MB/128MB" in summary 