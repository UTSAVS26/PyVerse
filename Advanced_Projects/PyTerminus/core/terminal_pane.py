"""
Terminal pane implementation using pseudo-terminals.
"""

import os
import struct
import select
import threading
import time
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# Platform-specific imports
try:
    import pty
    import fcntl
    import termios
    PLATFORM_SUPPORTED = True
except ImportError:
    # Windows doesn't have these modules
    PLATFORM_SUPPORTED = False
    pty = None


@dataclass
class PaneConfig:
    """Configuration for a terminal pane."""
    name: str
    command: str
    cwd: Optional[str] = None
    shell: str = 'bash'
    width: int = 80
    height: int = 24


class TerminalPane:
    """A terminal pane that manages a pseudo-terminal session."""
    
    def __init__(self, config: PaneConfig, output_callback: Optional[Callable] = None):
        self.config = config
        self.output_callback = output_callback
        self.master_fd = None
        self.slave_fd = None
        self.pid = None
        self.is_running = False
        self.output_buffer = []
        self.max_buffer_size = 1000
        
        # Threading
        self._output_thread = None
        self._input_thread = None
        self._lock = threading.Lock()
        
    def start(self) -> bool:
        """Start the terminal pane."""
        if not PLATFORM_SUPPORTED or pty is None:
            print("Terminal panes are not supported on this platform")
            return False
            
        try:
            # Create pseudo-terminal
            self.pid, self.master_fd = pty.fork()
            
            if self.pid == 0:
                # Child process
                self._setup_child_process()
            else:
                # Parent process
                self._setup_parent_process()
                return True
                
        except Exception as e:
            print(f"Error starting terminal pane: {e}")
            return False
            
        return False
    
    def _setup_child_process(self):
        """Setup the child process (shell)."""
        try:
            # Set working directory
            if self.config.cwd:
                os.chdir(self.config.cwd)
            
            # Set environment variables
            env = os.environ.copy()
            env['TERM'] = 'xterm-256color'
            env['COLUMNS'] = str(self.config.width)
            env['LINES'] = str(self.config.height)
            
            # Execute the command
            if self.config.command != self.config.shell:
                # Execute specific command
                os.execvpe(self.config.command, [self.config.command], env)
            else:
                # Start shell
                os.execvpe(self.config.shell, [self.config.shell], env)
                
        except Exception as e:
            print(f"Error in child process: {e}")
            os._exit(1)
    
    def _setup_parent_process(self):
        """Setup the parent process (terminal manager)."""
        if PLATFORM_SUPPORTED:
            # Set non-blocking mode
            fcntl.fcntl(self.master_fd, fcntl.F_SETFL, os.O_NONBLOCK)
        
        # Start output monitoring thread
        self.is_running = True
        self._output_thread = threading.Thread(target=self._monitor_output, daemon=True)
        self._output_thread.start()
    
    def _monitor_output(self):
        """Monitor output from the terminal."""
        while self.is_running:
            try:
                # Check if there's data to read
                ready, _, _ = select.select([self.master_fd], [], [], 0.1)
                
                if ready:
                    data = os.read(self.master_fd, 1024)
                    if data:
                        output = data.decode('utf-8', errors='replace')
                        
                        with self._lock:
                            self.output_buffer.append(output)
                            if len(self.output_buffer) > self.max_buffer_size:
                                self.output_buffer.pop(0)
                        
                        if self.output_callback:
                            self.output_callback(self.config.name, output)
                            
            except (OSError, IOError):
                # Process might have terminated
                break
            except Exception as e:
                print(f"Error monitoring output: {e}")
                break
        
        self.is_running = False
    
    def write_input(self, data: str):
        """Write input to the terminal."""
        if not self.is_running or self.master_fd is None:
            return False
        
        try:
            os.write(self.master_fd, data.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Error writing input: {e}")
            return False
    
    def resize(self, width: int, height: int):
        """Resize the terminal."""
        if not self.is_running or self.master_fd is None or not PLATFORM_SUPPORTED:
            return
        
        try:
            winsize = struct.pack("HHHH", height, width, 0, 0)
            fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, winsize)
            self.config.width = width
            self.config.height = height
        except Exception as e:
            print(f"Error resizing terminal: {e}")
    
    def get_output(self, lines: int = -1) -> str:
        """Get recent output from the terminal."""
        with self._lock:
            if lines == -1:
                return ''.join(self.output_buffer)
            else:
                return ''.join(self.output_buffer[-lines:])
    
    def stop(self):
        """Stop the terminal pane."""
        self.is_running = False
        
        if self.pid and self.master_fd:
            try:
                # Send SIGTERM to the process
                os.kill(self.pid, 15)
                
                # Wait a bit for graceful shutdown
                time.sleep(0.1)
                
                # Force kill if still running
                try:
                    os.kill(self.pid, 9)
                except ProcessLookupError:
                    pass
                
                # Close file descriptors
                os.close(self.master_fd)
                
            except Exception as e:
                print(f"Error stopping terminal pane: {e}")
    
    def is_alive(self) -> bool:
        """Check if the terminal process is still alive."""
        if not self.pid:
            return False
        
        try:
            os.kill(self.pid, 0)
            return True
        except (ProcessLookupError, OSError):
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information about the pane."""
        return {
            'name': self.config.name,
            'command': self.config.command,
            'cwd': self.config.cwd,
            'is_running': self.is_running,
            'is_alive': self.is_alive(),
            'width': self.config.width,
            'height': self.config.height,
            'buffer_size': len(self.output_buffer)
        } 