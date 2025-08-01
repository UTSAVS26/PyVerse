"""
Simplified tests for the Logger class that work on Windows.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock
from datetime import datetime

# Import logger directly without importing the problematic modules
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the problematic modules before importing
if sys.platform == 'win32':
    sys.modules['pty'] = Mock()
    sys.modules['fcntl'] = Mock()
    sys.modules['termios'] = Mock()

from core.logger import Logger


class TestLoggerSimple:
    """Simplified test cases for Logger."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def logger(self, temp_dir):
        """Create a Logger instance for testing."""
        return Logger(temp_dir)
    
    def test_init(self, logger, temp_dir):
        """Test Logger initialization."""
        assert logger.log_dir == temp_dir
        assert logger.session_log.exists()
        assert logger.output_log.exists()
        assert logger.command_log.exists()
    
    def test_init_log_files(self, temp_dir):
        """Test log file initialization."""
        logger = Logger(temp_dir)
        
        # Check that log files have headers
        with open(logger.session_log, 'r') as f:
            content = f.read()
            assert "# PyTerminus Session Log" in content
        
        with open(logger.output_log, 'r') as f:
            content = f.read()
            assert "# PyTerminus Output Log" in content
        
        with open(logger.command_log, 'r') as f:
            content = f.read()
            assert "# PyTerminus Command Log" in content
    
    def test_log_session_started(self, logger):
        """Test logging session start."""
        logger.log_session_started("test-session", 3)
        
        logs = logger.get_recent_logs('session', limit=1)
        assert len(logs) == 1
        
        log_entry = logs[0]
        assert log_entry['type'] == 'session_started'
        assert log_entry['data']['session_name'] == 'test-session'
        assert log_entry['data']['panes_count'] == 3
    
    def test_log_session_ended(self, logger):
        """Test logging session end."""
        logger.log_session_ended("test-session", 120.5)
        
        logs = logger.get_recent_logs('session', limit=1)
        assert len(logs) == 1
        
        log_entry = logs[0]
        assert log_entry['type'] == 'session_ended'
        assert log_entry['data']['session_name'] == 'test-session'
        assert log_entry['data']['duration_seconds'] == 120.5
    
    def test_log_pane_created(self, logger):
        """Test logging pane creation."""
        logger.log_pane_created("test_pane", "bash", "/home/user")
        
        logs = logger.get_recent_logs('session', limit=1)
        assert len(logs) == 1
        
        log_entry = logs[0]
        assert log_entry['type'] == 'pane_created'
        assert log_entry['data']['pane_name'] == 'test_pane'
        assert log_entry['data']['command'] == 'bash'
        assert log_entry['data']['cwd'] == '/home/user'
    
    def test_log_pane_output(self, logger):
        """Test logging pane output."""
        output = "Hello, World!\n"
        logger.log_pane_output("test_pane", output)
        
        logs = logger.get_recent_logs('output', limit=1)
        assert len(logs) == 1
        
        log_entry = logs[0]
        assert log_entry['type'] == 'pane_output'
        assert log_entry['data']['pane_name'] == 'test_pane'
        assert log_entry['data']['output'] == output
        assert log_entry['data']['output_length'] == len(output)
    
    def test_get_recent_logs(self, logger):
        """Test getting recent logs."""
        # Add some test logs
        logger.log_session_started("test1", 1)
        logger.log_session_started("test2", 2)
        logger.log_session_started("test3", 3)
        
        logs = logger.get_recent_logs('session', limit=2)
        assert len(logs) == 2
        
        # Should get the most recent logs
        assert logs[-1]['data']['session_name'] == 'test3'
        assert logs[-2]['data']['session_name'] == 'test2'
    
    def test_search_logs(self, logger):
        """Test searching logs."""
        # Add some test logs
        logger.log_session_started("test-session", 1)
        logger.log_pane_created("pane1", "bash", "/home")
        logger.log_pane_created("pane2", "zsh", "/tmp")
        
        # Search for specific content
        results = logger.search_logs("test-session", 'session')
        assert len(results) == 1
        assert results[0]['data']['session_name'] == 'test-session'
        
        results = logger.search_logs("pane", 'session')
        assert len(results) == 3  # session_started + 2 pane_created logs
        
        results = logger.search_logs("nonexistent", 'session')
        assert len(results) == 0 