"""
Pytest configuration and fixtures for PyTerminus tests.
"""

import pytest
import sys
from unittest.mock import Mock, patch

# Mock platform-specific modules for Windows testing
if sys.platform == 'win32':
    # Mock pty and related modules
    sys.modules['pty'] = Mock()
    sys.modules['fcntl'] = Mock()
    sys.modules['termios'] = Mock()
    
    # Mock the pty.fork function
    mock_pty = Mock()
    mock_pty.fork = Mock(return_value=(123, 456))  # (pid, master_fd)
    sys.modules['pty'] = mock_pty 