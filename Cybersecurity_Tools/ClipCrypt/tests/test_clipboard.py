"""
Tests for the clipboard monitoring module.
"""

import pytest
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
from clipcrypt.clipboard import ClipboardMonitor
from clipcrypt.storage import StorageManager


class TestClipboardMonitor:
    """Test cases for ClipboardMonitor."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def storage_manager(self, temp_dir):
        """Create a StorageManager instance for testing."""
        return StorageManager(temp_dir)
    
    @pytest.fixture
    def clipboard_monitor(self, storage_manager):
        """Create a ClipboardMonitor instance for testing."""
        return ClipboardMonitor(storage_manager)
    
    def test_initialization(self, clipboard_monitor, storage_manager):
        """Test ClipboardMonitor initialization."""
        assert clipboard_monitor.storage_manager == storage_manager
        assert clipboard_monitor.callback is None
        assert clipboard_monitor._monitoring is False
        assert clipboard_monitor._monitor_thread is None
        assert clipboard_monitor._last_content == ""
        assert clipboard_monitor._check_interval == 1.0
    
    def test_initialization_with_callback(self, storage_manager):
        """Test ClipboardMonitor initialization with callback."""
        callback = MagicMock()
        monitor = ClipboardMonitor(storage_manager, callback)
        
        assert monitor.callback == callback
    
    def test_start_monitoring(self, clipboard_monitor):
        """Test starting clipboard monitoring."""
        assert clipboard_monitor._monitoring is False
        
        clipboard_monitor.start_monitoring()
        
        assert clipboard_monitor._monitoring is True
        assert clipboard_monitor._monitor_thread is not None
        assert clipboard_monitor._monitor_thread.is_alive()
    
    def test_start_monitoring_already_monitoring(self, clipboard_monitor):
        """Test starting monitoring when already monitoring."""
        clipboard_monitor._monitoring = True
        
        clipboard_monitor.start_monitoring()
        
        # Should not start another thread
        assert clipboard_monitor._monitoring is True
    
    def test_stop_monitoring(self, clipboard_monitor):
        """Test stopping clipboard monitoring."""
        clipboard_monitor._monitoring = True
        clipboard_monitor._monitor_thread = MagicMock()
        
        clipboard_monitor.stop_monitoring()
        
        assert clipboard_monitor._monitoring is False
    
    def test_is_monitoring(self, clipboard_monitor):
        """Test is_monitoring method."""
        assert clipboard_monitor.is_monitoring() is False
        
        clipboard_monitor._monitoring = True
        assert clipboard_monitor.is_monitoring() is True
    
    def test_set_check_interval(self, clipboard_monitor):
        """Test setting check interval."""
        clipboard_monitor.set_check_interval(2.5)
        assert clipboard_monitor._check_interval == 2.5
    
    def test_set_check_interval_invalid(self, clipboard_monitor):
        """Test setting invalid check interval."""
        original_interval = clipboard_monitor._check_interval
        
        clipboard_monitor.set_check_interval(-1)
        assert clipboard_monitor._check_interval == original_interval
        
        clipboard_monitor.set_check_interval(0)
        assert clipboard_monitor._check_interval == original_interval
    
    def test_get_current_content(self, clipboard_monitor):
        """Test getting current clipboard content."""
        with patch('clipcrypt.clipboard.pyperclip.paste', return_value="test content"):
            content = clipboard_monitor.get_current_content()
            assert content == "test content"
    
    def test_get_current_content_error(self, clipboard_monitor):
        """Test getting clipboard content when error occurs."""
        with patch('clipcrypt.clipboard.pyperclip.paste', side_effect=Exception("Clipboard error")):
            with patch('builtins.print') as mock_print:
                content = clipboard_monitor.get_current_content()
                assert content == ""
                mock_print.assert_called()
    
    def test_set_content(self, clipboard_monitor):
        """Test setting clipboard content."""
        with patch('clipcrypt.clipboard.pyperclip.copy') as mock_copy:
            result = clipboard_monitor.set_content("test content")
            
            assert result is True
            mock_copy.assert_called_with("test content")
    
    def test_set_content_error(self, clipboard_monitor):
        """Test setting clipboard content when error occurs."""
        with patch('clipcrypt.clipboard.pyperclip.copy', side_effect=Exception("Clipboard error")):
            with patch('builtins.print') as mock_print:
                result = clipboard_monitor.set_content("test content")
                
                assert result is False
                mock_print.assert_called()
    
    def test_monitor_loop_new_content(self, clipboard_monitor):
        """Test monitoring loop with new content."""
        # Mock clipboard content changes
        content_sequence = ["", "old content", "new content"]
        content_index = 0
        
        def mock_paste():
            nonlocal content_index
            content = content_sequence[content_index]
            content_index += 1
            return content
        
        with patch('clipcrypt.clipboard.pyperclip.paste', side_effect=mock_paste):
            with patch('clipcrypt.clipboard.time.sleep'):
                with patch('builtins.print') as mock_print:
                    clipboard_monitor._monitoring = True
                    clipboard_monitor._monitor_loop()
                    
                    # Should have printed detection messages
                    assert mock_print.called
    
    def test_monitor_loop_no_change(self, clipboard_monitor):
        """Test monitoring loop with no content change."""
        with patch('clipcrypt.clipboard.pyperclip.paste', return_value="same content"):
            with patch('clipcrypt.clipboard.time.sleep'):
                clipboard_monitor._last_content = "same content"
                clipboard_monitor._monitoring = True
                
                # Should not add any entries
                initial_count = len(clipboard_monitor.storage_manager._entries)
                clipboard_monitor._monitor_loop()
                
                # Give a moment for the loop to process
                time.sleep(0.1)
                
                # Should not have added any entries
                assert len(clipboard_monitor.storage_manager._entries) == initial_count
    
    def test_monitor_loop_empty_content(self, clipboard_monitor):
        """Test monitoring loop with empty content."""
        with patch('clipcrypt.clipboard.pyperclip.paste', return_value=""):
            with patch('clipcrypt.clipboard.time.sleep'):
                clipboard_monitor._monitoring = True
                
                # Should not add empty content
                initial_count = len(clipboard_monitor.storage_manager._entries)
                clipboard_monitor._monitor_loop()
                
                # Give a moment for the loop to process
                time.sleep(0.1)
                
                # Should not have added any entries
                assert len(clipboard_monitor.storage_manager._entries) == initial_count
    
    def test_monitor_loop_single_character(self, clipboard_monitor):
        """Test monitoring loop with single character content."""
        with patch('clipcrypt.clipboard.pyperclip.paste', return_value="a"):
            with patch('clipcrypt.clipboard.time.sleep'):
                clipboard_monitor._monitoring = True
                
                # Should not add single character content
                initial_count = len(clipboard_monitor.storage_manager._entries)
                clipboard_monitor._monitor_loop()
                
                # Give a moment for the loop to process
                time.sleep(0.1)
                
                # Should not have added any entries
                assert len(clipboard_monitor.storage_manager._entries) == initial_count
    
    def test_monitor_loop_with_callback(self, storage_manager):
        """Test monitoring loop with callback function."""
        callback = MagicMock()
        monitor = ClipboardMonitor(storage_manager, callback)
        
        with patch('clipcrypt.clipboard.pyperclip.paste', return_value="test content"):
            with patch('clipcrypt.clipboard.time.sleep'):
                monitor._monitoring = True
                monitor._monitor_loop()
                
                # Give a moment for the loop to process
                time.sleep(0.1)
                
                # Should have called callback
                callback.assert_called()
    
    def test_monitor_loop_error_handling(self, clipboard_monitor):
        """Test monitoring loop error handling."""
        with patch('clipcrypt.clipboard.pyperclip.paste', side_effect=Exception("Clipboard error")):
            with patch('clipcrypt.clipboard.time.sleep'):
                with patch('builtins.print') as mock_print:
                    clipboard_monitor._monitoring = True
                    clipboard_monitor._monitor_loop()
                    
                    # Should have printed error message
                    assert mock_print.called 