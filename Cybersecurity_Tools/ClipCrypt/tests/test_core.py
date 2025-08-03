"""
Tests for the core ClipCrypt module.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from clipcrypt.core import ClipCrypt


class TestClipCrypt:
    """Test cases for ClipCrypt core functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def clipcrypt(self, temp_dir):
        """Create a ClipCrypt instance for testing."""
        return ClipCrypt(temp_dir)
    
    def test_initialization(self, clipcrypt, temp_dir):
        """Test ClipCrypt initialization."""
        assert clipcrypt.config_dir == temp_dir
        assert clipcrypt.storage_manager is not None
        assert clipcrypt.clipboard_monitor is not None
        assert clipcrypt.console is not None
    
    def test_get_default_config_dir_windows(self):
        """Test default config directory on Windows."""
        with patch('clipcrypt.core.sys.platform', 'win32'):
            with patch('clipcrypt.core.os.environ', {'APPDATA': 'C:\\AppData'}):
                clipcrypt = ClipCrypt()
                expected = Path('C:\\AppData') / "ClipCrypt"
                assert clipcrypt.config_dir == expected
    
    def test_get_default_config_dir_macos(self):
        """Test default config directory on macOS."""
        with patch('clipcrypt.core.sys.platform', 'darwin'):
            with patch('clipcrypt.core.Path.home') as mock_home:
                mock_home.return_value = Path('/home/user')
                clipcrypt = ClipCrypt()
                expected = Path('/home/user/Library/Application Support/ClipCrypt')
                assert clipcrypt.config_dir == expected
    
    def test_get_default_config_dir_linux(self):
        """Test default config directory on Linux."""
        with patch('clipcrypt.core.sys.platform', 'linux'):
            with patch('clipcrypt.core.Path.home') as mock_home:
                mock_home.return_value = Path('/home/user')
                clipcrypt = ClipCrypt()
                expected = Path('/home/user/.config/ClipCrypt')
                assert clipcrypt.config_dir == expected
    
    def test_list_entries(self, clipcrypt):
        """Test listing entries."""
        # Add some test entries
        clipcrypt.storage_manager.add_entry("First entry")
        clipcrypt.storage_manager.add_entry("Second entry")
        
        # Mock console.print to capture output
        with patch.object(clipcrypt.console, 'print') as mock_print:
            clipcrypt.list_entries()
            
            # Should have called print at least once
            assert mock_print.called
    
    def test_list_entries_empty(self, clipcrypt):
        """Test listing entries when none exist."""
        with patch.object(clipcrypt.console, 'print') as mock_print:
            clipcrypt.list_entries()
            
            # Should print "No clipboard entries found."
            mock_print.assert_called_with("No clipboard entries found.")
    
    def test_get_entry(self, clipcrypt):
        """Test getting a specific entry."""
        # Add a test entry
        entry_id = clipcrypt.storage_manager.add_entry("Test content")
        
        with patch.object(clipcrypt.console, 'print') as mock_print:
            clipcrypt.get_entry(entry_id)
            
            # Should have called print for content and metadata panels
            assert mock_print.call_count >= 2
    
    def test_get_nonexistent_entry(self, clipcrypt):
        """Test getting a non-existent entry."""
        with patch.object(clipcrypt.console, 'print') as mock_print:
            clipcrypt.get_entry(999)
            
            # Should print error message
            mock_print.assert_called_with("[red]Entry 999 not found.[/red]")
    
    def test_search_entries(self, clipcrypt):
        """Test searching entries."""
        # Add test entries
        clipcrypt.storage_manager.add_entry("Hello world")
        clipcrypt.storage_manager.add_entry("Python programming")
        
        with patch.object(clipcrypt.console, 'print') as mock_print:
            clipcrypt.search_entries("world")
            
            # Should have called print for results
            assert mock_print.called
    
    def test_search_no_results(self, clipcrypt):
        """Test searching with no results."""
        with patch.object(clipcrypt.console, 'print') as mock_print:
            clipcrypt.search_entries("nonexistent")
            
            # Should print no results message
            mock_print.assert_called_with("No entries found matching 'nonexistent'.")
    
    def test_delete_entry(self, clipcrypt):
        """Test deleting an entry."""
        # Add a test entry
        entry_id = clipcrypt.storage_manager.add_entry("Test content")
        
        with patch.object(clipcrypt.console, 'print') as mock_print:
            clipcrypt.delete_entry(entry_id)
            
            # Should print success message
            mock_print.assert_called_with(f"[green]Entry {entry_id} deleted successfully.[/green]")
    
    def test_delete_nonexistent_entry(self, clipcrypt):
        """Test deleting a non-existent entry."""
        with patch.object(clipcrypt.console, 'print') as mock_print:
            clipcrypt.delete_entry(999)
            
            # Should print error message
            mock_print.assert_called_with("[red]Entry 999 not found.[/red]")
    
    def test_add_tag(self, clipcrypt):
        """Test adding a tag to an entry."""
        # Add a test entry
        entry_id = clipcrypt.storage_manager.add_entry("Test content")
        
        with patch.object(clipcrypt.console, 'print') as mock_print:
            clipcrypt.add_tag(entry_id, "test")
            
            # Should print success message
            mock_print.assert_called_with("[green]Tag 'test' added to entry 1.[/green]")
    
    def test_remove_tag(self, clipcrypt):
        """Test removing a tag from an entry."""
        # Add a test entry with a tag
        entry_id = clipcrypt.storage_manager.add_entry("Test content", tags=["test"])
        
        with patch.object(clipcrypt.console, 'print') as mock_print:
            clipcrypt.remove_tag(entry_id, "test")
            
            # Should print success message
            mock_print.assert_called_with("[green]Tag 'test' removed from entry 1.[/green]")
    
    def test_get_entries_by_tag(self, clipcrypt):
        """Test getting entries by tag."""
        # Add test entries with tags
        clipcrypt.storage_manager.add_entry("First entry", tags=["code"])
        clipcrypt.storage_manager.add_entry("Second entry", tags=["test"])
        clipcrypt.storage_manager.add_entry("Third entry", tags=["code", "test"])
        
        with patch.object(clipcrypt.console, 'print') as mock_print:
            clipcrypt.get_entries_by_tag("code")
            
            # Should have called print for results
            assert mock_print.called
    
    def test_clear_all(self, clipcrypt):
        """Test clearing all entries."""
        # Add test entries
        clipcrypt.storage_manager.add_entry("First entry")
        clipcrypt.storage_manager.add_entry("Second entry")
        
        # Mock input to return 'y'
        with patch('builtins.input', return_value='y'):
            with patch.object(clipcrypt.console, 'print') as mock_print:
                clipcrypt.clear_all()
                
                # Should print confirmation and success messages
                assert mock_print.call_count >= 2
    
    def test_clear_all_cancelled(self, clipcrypt):
        """Test clearing all entries when cancelled."""
        # Add test entries
        clipcrypt.storage_manager.add_entry("First entry")
        
        # Mock input to return 'n'
        with patch('builtins.input', return_value='n'):
            with patch.object(clipcrypt.console, 'print') as mock_print:
                clipcrypt.clear_all()
                
                # Should print cancellation message
                mock_print.assert_called_with("[blue]Operation cancelled.[/blue]")
    
    def test_show_stats(self, clipcrypt):
        """Test showing statistics."""
        # Add test entries
        clipcrypt.storage_manager.add_entry("First entry", tags=["code"])
        clipcrypt.storage_manager.add_entry("Second entry", tags=["test"])
        
        with patch.object(clipcrypt.console, 'print') as mock_print:
            clipcrypt.show_stats()
            
            # Should have called print for stats panel
            assert mock_print.called
    
    def test_copy_to_clipboard(self, clipcrypt):
        """Test copying an entry to clipboard."""
        # Add a test entry
        entry_id = clipcrypt.storage_manager.add_entry("Test content")
        
        # Mock clipboard monitor
        clipcrypt.clipboard_monitor.set_content = MagicMock(return_value=True)
        
        with patch.object(clipcrypt.console, 'print') as mock_print:
            clipcrypt.copy_to_clipboard(entry_id)
            
            # Should print success message
            mock_print.assert_called_with(f"[green]Entry {entry_id} copied to clipboard.[/green]")
    
    def test_copy_nonexistent_to_clipboard(self, clipcrypt):
        """Test copying a non-existent entry to clipboard."""
        with patch.object(clipcrypt.console, 'print') as mock_print:
            clipcrypt.copy_to_clipboard(999)
            
            # Should print error message
            mock_print.assert_called_with("[red]Entry 999 not found.[/red]")
    
    def test_start_monitoring(self, clipcrypt):
        """Test starting clipboard monitoring."""
        # Mock the monitoring loop to avoid infinite loop
        with patch.object(clipcrypt.clipboard_monitor, 'start_monitoring'):
            with patch.object(clipcrypt.clipboard_monitor, 'is_monitoring', return_value=False):
                clipcrypt.start_monitoring()
                
                # Should have called start_monitoring
                clipcrypt.clipboard_monitor.start_monitoring.assert_called_once()
    
    def test_start_monitoring_keyboard_interrupt(self, clipcrypt):
        """Test handling KeyboardInterrupt during monitoring."""
        # Mock the monitoring loop to raise KeyboardInterrupt
        with patch.object(clipcrypt.clipboard_monitor, 'start_monitoring'):
            with patch.object(clipcrypt.clipboard_monitor, 'is_monitoring', side_effect=KeyboardInterrupt):
                with patch.object(clipcrypt.clipboard_monitor, 'stop_monitoring'):
                    clipcrypt.start_monitoring()
                    
                    # Should have called stop_monitoring
                    clipcrypt.clipboard_monitor.stop_monitoring.assert_called_once() 