"""
Tests for the Logger class.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock
from datetime import datetime

from core.logger import Logger


class TestLogger:
    """Test cases for Logger."""
    
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
    
    def test_log_session_saved(self, logger):
        """Test logging session save."""
        logger.log_session_saved("test_session.yaml", 2)
        
        logs = logger.get_recent_logs('session', limit=1)
        assert len(logs) == 1
        
        log_entry = logs[0]
        assert log_entry['type'] == 'session_saved'
        assert log_entry['data']['filename'] == 'test_session.yaml'
        assert log_entry['data']['panes_count'] == 2
    
    def test_log_session_loaded(self, logger):
        """Test logging session load."""
        logger.log_session_loaded("test_session.yaml", 3)
        
        logs = logger.get_recent_logs('session', limit=1)
        assert len(logs) == 1
        
        log_entry = logs[0]
        assert log_entry['type'] == 'session_loaded'
        assert log_entry['data']['filename'] == 'test_session.yaml'
        assert log_entry['data']['panes_count'] == 3
    
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
    
    def test_log_pane_removed(self, logger):
        """Test logging pane removal."""
        logger.log_pane_removed("test_pane")
        
        logs = logger.get_recent_logs('session', limit=1)
        assert len(logs) == 1
        
        log_entry = logs[0]
        assert log_entry['type'] == 'pane_removed'
        assert log_entry['data']['pane_name'] == 'test_pane'
    
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
    
    def test_log_pane_output_truncation(self, logger):
        """Test that long outputs are truncated."""
        long_output = "x" * 2000  # Very long output
        logger.log_pane_output("test_pane", long_output)
        
        logs = logger.get_recent_logs('output', limit=1)
        assert len(logs) == 1
        
        log_entry = logs[0]
        assert log_entry['data']['output_length'] <= 1000
        assert "... [truncated]" in log_entry['data']['output']
    
    def test_log_command_executed(self, logger):
        """Test logging command execution."""
        logger.log_command_executed("test_pane", "ls -la", True)
        
        logs = logger.get_recent_logs('command', limit=1)
        assert len(logs) == 1
        
        log_entry = logs[0]
        assert log_entry['type'] == 'command_executed'
        assert log_entry['data']['pane_name'] == 'test_pane'
        assert log_entry['data']['command'] == 'ls -la'
        assert log_entry['data']['success'] is True
    
    def test_log_error(self, logger):
        """Test logging errors."""
        context = {'pane': 'test_pane', 'command': 'invalid_cmd'}
        logger.log_error("command_error", "Command not found", context)
        
        logs = logger.get_recent_logs('session', limit=1)
        assert len(logs) == 1
        
        log_entry = logs[0]
        assert log_entry['type'] == 'error'
        assert log_entry['data']['error_type'] == 'command_error'
        assert log_entry['data']['error_message'] == 'Command not found'
        assert log_entry['data']['context'] == context
    
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
    
    def test_get_recent_logs_empty(self, temp_dir):
        """Test getting logs from empty log file."""
        logger = Logger(temp_dir)
        logs = logger.get_recent_logs('session')
        assert logs == []
    
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
        assert len(results) == 2
        
        results = logger.search_logs("nonexistent", 'session')
        assert len(results) == 0
    
    def test_get_session_summary(self, logger):
        """Test getting session summary."""
        # Add various types of logs
        logger.log_session_started("test1", 2)
        logger.log_session_started("test2", 1)
        logger.log_pane_created("pane1", "bash")
        logger.log_pane_created("pane2", "zsh")
        logger.log_command_executed("pane1", "ls")
        logger.log_command_executed("pane2", "pwd")
        logger.log_error("test_error", "Something went wrong")
        
        summary = logger.get_session_summary()
        
        assert summary['total_sessions'] == 2
        assert summary['total_panes_created'] == 2
        assert summary['total_commands'] == 2
        assert summary['errors'] == 1
        assert summary['last_activity'] is not None
    
    def test_get_session_summary_empty(self, temp_dir):
        """Test getting summary from empty logs."""
        logger = Logger(temp_dir)
        summary = logger.get_session_summary()
        
        assert summary['total_sessions'] == 0
        assert summary['total_panes_created'] == 0
        assert summary['total_commands'] == 0
        assert summary['errors'] == 0
        assert summary['last_activity'] is None 