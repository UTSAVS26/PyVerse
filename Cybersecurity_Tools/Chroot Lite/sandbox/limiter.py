"""
Resource limiting functionality for sandboxed processes.
"""

import os
import signal
import time
import psutil
from typing import Optional, Dict, Any
import logging

# Platform-specific imports
try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False

logger = logging.getLogger(__name__)


class ResourceLimiter:
    """Manages resource limits for sandboxed processes."""
    
    def __init__(self, memory_limit_mb: int = 128, cpu_limit_seconds: int = 30):
        """
        Initialize resource limiter.
        
        Args:
            memory_limit_mb: Memory limit in megabytes
            cpu_limit_seconds: CPU time limit in seconds
        """
        self.memory_limit_mb = memory_limit_mb
        self.cpu_limit_seconds = cpu_limit_seconds
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        
    def set_limits(self, pid: int) -> None:
        """
        Set resource limits for a process.
        
        Args:
            pid: Process ID to set limits for
        """
        if not RESOURCE_AVAILABLE:
            logger.warning("Resource limiting not available on this platform")
            return
            
        try:
            # Set memory limit (RLIMIT_AS - address space limit)
            resource.setrlimit(resource.RLIMIT_AS, (self.memory_limit_bytes, self.memory_limit_bytes))
            
            # Set CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_limit_seconds, self.cpu_limit_seconds))
            
            # Set file size limit (1MB)
            resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
            
            # Set number of open files limit
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
            
            logger.info(f"Set resource limits for PID {pid}: "
                       f"Memory={self.memory_limit_mb}MB, "
                       f"CPU={self.cpu_limit_seconds}s")
            
        except Exception as e:
            logger.error(f"Failed to set resource limits for PID {pid}: {e}")
            raise
    
    def monitor_process(self, pid: int) -> Dict[str, Any]:
        """
        Monitor a process for resource usage.
        
        Args:
            pid: Process ID to monitor
            
        Returns:
            Dictionary with resource usage information
        """
        try:
            process = psutil.Process(pid)
            
            # Get memory usage
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # Get CPU usage
            cpu_percent = process.cpu_percent()
            cpu_times = process.cpu_times()
            
            # Check if limits are exceeded
            memory_exceeded = memory_mb > self.memory_limit_mb
            cpu_exceeded = cpu_times.user + cpu_times.system > self.cpu_limit_seconds
            
            usage_info = {
                'pid': pid,
                'memory_mb': round(memory_mb, 2),
                'memory_limit_mb': self.memory_limit_mb,
                'memory_exceeded': memory_exceeded,
                'cpu_percent': round(cpu_percent, 2),
                'cpu_time_user': round(cpu_times.user, 2),
                'cpu_time_system': round(cpu_times.system, 2),
                'cpu_limit_seconds': self.cpu_limit_seconds,
                'cpu_exceeded': cpu_exceeded,
                'status': 'running'
            }
            
            if memory_exceeded or cpu_exceeded:
                usage_info['status'] = 'limit_exceeded'
                logger.warning(f"Resource limits exceeded for PID {pid}: "
                             f"Memory={memory_mb:.2f}MB/{self.memory_limit_mb}MB, "
                             f"CPU={cpu_times.user + cpu_times.system:.2f}s/{self.cpu_limit_seconds}s")
            
            return usage_info
            
        except psutil.NoSuchProcess:
            return {
                'pid': pid,
                'status': 'terminated',
                'error': 'Process not found'
            }
        except Exception as e:
            logger.error(f"Error monitoring process {pid}: {e}")
            return {
                'pid': pid,
                'status': 'error',
                'error': str(e)
            }
    
    def terminate_if_exceeded(self, pid: int) -> bool:
        """
        Terminate process if resource limits are exceeded.
        
        Args:
            pid: Process ID to check and potentially terminate
            
        Returns:
            True if process was terminated, False otherwise
        """
        usage_info = self.monitor_process(pid)
        
        if usage_info.get('status') == 'limit_exceeded':
            try:
                os.kill(pid, signal.SIGTERM)
                logger.info(f"Terminated process {pid} due to resource limit violation")
                return True
            except ProcessLookupError:
                logger.info(f"Process {pid} already terminated")
                return True
            except Exception as e:
                logger.error(f"Failed to terminate process {pid}: {e}")
                return False
        
        return False
    
    def get_usage_summary(self, pid: int) -> str:
        """
        Get a human-readable summary of process resource usage.
        
        Args:
            pid: Process ID to get summary for
            
        Returns:
            Formatted usage summary string
        """
        usage_info = self.monitor_process(pid)
        
        if usage_info.get('status') == 'terminated':
            return f"Process {pid}: Terminated"
        elif usage_info.get('status') == 'error':
            return f"Process {pid}: Error - {usage_info.get('error', 'Unknown error')}"
        
        memory_status = "⚠️" if usage_info.get('memory_exceeded') else "✅"
        cpu_status = "⚠️" if usage_info.get('cpu_exceeded') else "✅"
        
        return (f"Process {pid}: "
                f"Memory {memory_status} {usage_info.get('memory_mb', 0):.1f}MB/{self.memory_limit_mb}MB, "
                f"CPU {cpu_status} {usage_info.get('cpu_percent', 0):.1f}% "
                f"({usage_info.get('cpu_time_user', 0):.1f}s user, "
                f"{usage_info.get('cpu_time_system', 0):.1f}s system)") 