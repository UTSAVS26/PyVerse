"""
Tests for the SessionManager class.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import yaml

from core.session_manager import SessionManager
from core.terminal_pane import PaneConfig


class TestSessionManager:
    """Test cases for SessionManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def session_manager(self, temp_dir):
        """Create a SessionManager instance for testing."""
        return SessionManager(
            default_shell='bash',
            log_dir=temp_dir,
            logger=Mock()
        )
    
    def test_init(self, session_manager):
        """Test SessionManager initialization."""
        assert session_manager.default_shell == 'bash'
        assert session_manager.panes == {}
        assert session_manager.active_pane is None
        assert session_manager.session_name == "default"
    
    def test_create_pane_success(self, session_manager):
        """Test successful pane creation."""
        with patch('core.session_manager.TerminalPane') as mock_pane_class:
            mock_pane = Mock()
            mock_pane.start.return_value = True
            mock_pane_class.return_value = mock_pane
            
            result = session_manager.create_pane("test_pane", "bash")
            
            assert result is True
            assert "test_pane" in session_manager.panes
            assert session_manager.active_pane == "test_pane"
    
    def test_create_pane_failure(self, session_manager):
        """Test pane creation failure."""
        with patch('core.session_manager.TerminalPane') as mock_pane_class:
            mock_pane = Mock()
            mock_pane.start.return_value = False
            mock_pane_class.return_value = mock_pane
            
            result = session_manager.create_pane("test_pane", "bash")
            
            assert result is False
            assert "test_pane" not in session_manager.panes
    
    def test_create_pane_duplicate(self, session_manager):
        """Test creating a pane with duplicate name."""
        with patch('core.session_manager.TerminalPane') as mock_pane_class:
            mock_pane = Mock()
            mock_pane.start.return_value = True
            mock_pane_class.return_value = mock_pane
            
            # Create first pane
            session_manager.create_pane("test_pane", "bash")
            
            # Try to create duplicate
            result = session_manager.create_pane("test_pane", "zsh")
            
            assert result is False
    
    def test_remove_pane(self, session_manager):
        """Test removing a pane."""
        with patch('core.session_manager.TerminalPane') as mock_pane_class:
            mock_pane = Mock()
            mock_pane.start.return_value = True
            mock_pane_class.return_value = mock_pane
            
            # Create pane
            session_manager.create_pane("test_pane", "bash")
            assert "test_pane" in session_manager.panes
            
            # Remove pane
            result = session_manager.remove_pane("test_pane")
            
            assert result is True
            assert "test_pane" not in session_manager.panes
    
    def test_remove_nonexistent_pane(self, session_manager):
        """Test removing a non-existent pane."""
        result = session_manager.remove_pane("nonexistent")
        assert result is False
    
    def test_get_pane(self, session_manager):
        """Test getting a pane by name."""
        with patch('core.session_manager.TerminalPane') as mock_pane_class:
            mock_pane = Mock()
            mock_pane.start.return_value = True
            mock_pane_class.return_value = mock_pane
            
            session_manager.create_pane("test_pane", "bash")
            
            pane = session_manager.get_pane("test_pane")
            assert pane is not None
            
            nonexistent = session_manager.get_pane("nonexistent")
            assert nonexistent is None
    
    def test_set_active_pane(self, session_manager):
        """Test setting the active pane."""
        with patch('core.session_manager.TerminalPane') as mock_pane_class:
            mock_pane = Mock()
            mock_pane.start.return_value = True
            mock_pane_class.return_value = mock_pane
            
            session_manager.create_pane("pane1", "bash")
            session_manager.create_pane("pane2", "zsh")
            
            # Set active pane
            result = session_manager.set_active_pane("pane2")
            assert result is True
            assert session_manager.active_pane == "pane2"
            
            # Try to set non-existent pane
            result = session_manager.set_active_pane("nonexistent")
            assert result is False
    
    def test_write_to_active_pane(self, session_manager):
        """Test writing to active pane."""
        with patch('core.session_manager.TerminalPane') as mock_pane_class:
            mock_pane = Mock()
            mock_pane.start.return_value = True
            mock_pane.write_input.return_value = True
            mock_pane_class.return_value = mock_pane
            
            session_manager.create_pane("test_pane", "bash")
            
            result = session_manager.write_to_active_pane("test input")
            assert result is True
            mock_pane.write_input.assert_called_with("test input")
    
    def test_write_to_no_active_pane(self, session_manager):
        """Test writing when no active pane."""
        result = session_manager.write_to_active_pane("test input")
        assert result is False
    
    def test_save_session(self, session_manager, temp_dir):
        """Test saving a session."""
        with patch('core.session_manager.TerminalPane') as mock_pane_class:
            mock_pane = Mock()
            mock_pane.start.return_value = True
            mock_pane.config.command = "bash"
            mock_pane.config.cwd = "/home/user"
            mock_pane.config.shell = "bash"
            mock_pane_class.return_value = mock_pane
            
            session_manager.create_pane("pane1", "bash", "/home/user")
            session_manager.create_pane("pane2", "zsh", "/tmp")
            
            session_file = temp_dir / "test_session.yaml"
            result = session_manager.save_session(str(session_file))
            
            assert result is True
            assert session_file.exists()
            
            # Verify saved content
            with open(session_file, 'r') as f:
                data = yaml.safe_load(f)
                assert data['name'] == "default"
                assert len(data['panes']) == 2
                assert data['panes'][0]['name'] == "pane1"
                assert data['panes'][1]['name'] == "pane2"
    
    def test_load_session(self, session_manager, temp_dir):
        """Test loading a session."""
        # Create a test session file
        session_data = {
            'name': 'test-session',
            'panes': [
                {
                    'name': 'pane1',
                    'command': 'bash',
                    'cwd': '/home/user',
                    'shell': 'bash'
                },
                {
                    'name': 'pane2',
                    'command': 'zsh',
                    'cwd': '/tmp',
                    'shell': 'zsh'
                }
            ]
        }
        
        session_file = temp_dir / "test_session.yaml"
        with open(session_file, 'w') as f:
            yaml.dump(session_data, f)
        
        with patch('core.session_manager.TerminalPane') as mock_pane_class:
            mock_pane = Mock()
            mock_pane.start.return_value = True
            mock_pane_class.return_value = mock_pane
            
            result = session_manager.load_session(str(session_file))
            
            assert result is True
            assert session_manager.session_name == "test-session"
            assert len(session_manager.panes) == 2
            assert "pane1" in session_manager.panes
            assert "pane2" in session_manager.panes
    
    def test_load_nonexistent_session(self, session_manager):
        """Test loading a non-existent session file."""
        result = session_manager.load_session("nonexistent.yaml")
        assert result is False
    
    def test_get_session_info(self, session_manager):
        """Test getting session information."""
        with patch('core.session_manager.TerminalPane') as mock_pane_class:
            mock_pane = Mock()
            mock_pane.start.return_value = True
            mock_pane.is_alive.return_value = True
            mock_pane_class.return_value = mock_pane
            
            session_manager.create_pane("pane1", "bash")
            session_manager.create_pane("pane2", "zsh")
            
            info = session_manager.get_session_info()
            assert "2/2 terminals" in info
            assert "(default)" in info
    
    def test_cleanup(self, session_manager):
        """Test cleanup of session manager."""
        with patch('core.session_manager.TerminalPane') as mock_pane_class:
            mock_pane = Mock()
            mock_pane.start.return_value = True
            mock_pane_class.return_value = mock_pane
            
            session_manager.create_pane("pane1", "bash")
            session_manager.create_pane("pane2", "zsh")
            
            assert len(session_manager.panes) == 2
            
            session_manager.cleanup()
            
            assert len(session_manager.panes) == 0
            assert session_manager.active_pane is None
            
            # Verify stop was called on all panes
            for pane in session_manager.panes.values():
                pane.stop.assert_called_once() 