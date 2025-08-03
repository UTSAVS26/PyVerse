"""
Tests for the KeyBindings class.
"""

import pytest
from tui.keybindings import KeyBindings


class TestKeyBindings:
    """Test cases for KeyBindings."""
    
    @pytest.fixture
    def keybindings(self):
        """Create a KeyBindings instance for testing."""
        return KeyBindings()
    
    def test_init(self, keybindings):
        """Test KeyBindings initialization."""
        assert keybindings.bindings is not None
        assert len(keybindings.bindings) > 0
    
    def test_default_bindings(self, keybindings):
        """Test that default bindings are set up."""
        # Check some key bindings exist
        assert 'ctrl n' in keybindings.bindings
        assert 'ctrl tab' in keybindings.bindings
        assert 'ctrl x' in keybindings.bindings
        assert 'ctrl q' in keybindings.bindings
        assert 'f1' in keybindings.bindings
    
    def test_get_binding(self, keybindings):
        """Test getting a binding."""
        assert keybindings.get_binding('ctrl n') == 'new_pane'
        assert keybindings.get_binding('ctrl q') == 'quit'
        assert keybindings.get_binding('nonexistent') == 'unknown'
    
    def test_set_binding(self, keybindings):
        """Test setting a binding."""
        keybindings.set_binding('ctrl a', 'custom_action')
        assert keybindings.get_binding('ctrl a') == 'custom_action'
    
    def test_remove_binding(self, keybindings):
        """Test removing a binding."""
        # Set a custom binding
        keybindings.set_binding('ctrl a', 'custom_action')
        assert keybindings.get_binding('ctrl a') == 'custom_action'
        
        # Remove it
        keybindings.remove_binding('ctrl a')
        assert keybindings.get_binding('ctrl a') == 'unknown'
    
    def test_remove_nonexistent_binding(self, keybindings):
        """Test removing a non-existent binding."""
        # Should not raise an exception
        keybindings.remove_binding('nonexistent')
    
    def test_get_all_bindings(self, keybindings):
        """Test getting all bindings."""
        bindings = keybindings.get_all_bindings()
        assert isinstance(bindings, dict)
        assert len(bindings) > 0
        assert 'ctrl n' in bindings
        assert 'ctrl q' in bindings
    
    def test_get_help_text(self, keybindings):
        """Test getting help text."""
        help_text = keybindings.get_help_text()
        assert isinstance(help_text, str)
        assert len(help_text) > 0
        assert "PyTerminus Key Bindings" in help_text
        assert "Navigation" in help_text
        assert "Pane Management" in help_text
        assert "Layout" in help_text
        assert "Search" in help_text
        assert "Session" in help_text
        assert "Help" in help_text
        assert "Exit" in help_text
    
    def test_parse_key(self, keybindings):
        """Test parsing keys."""
        # Test special keys
        assert keybindings.parse_key('tab') == 'next_pane'
        assert keybindings.parse_key('shift tab') == 'prev_pane'
        
        # Test ctrl keys
        assert keybindings.parse_key('ctrl n') == 'new_pane'
        assert keybindings.parse_key('ctrl q') == 'quit'
        
        # Test function keys
        assert keybindings.parse_key('f1') == 'show_help'
        assert keybindings.parse_key('f2') == 'show_status'
        
        # Test unknown keys
        assert keybindings.parse_key('unknown') == 'unknown'
        assert keybindings.parse_key('ctrl unknown') == 'unknown'
    
    def test_is_navigation_key(self, keybindings):
        """Test navigation key detection."""
        assert keybindings.is_navigation_key('ctrl tab') is True
        assert keybindings.is_navigation_key('ctrl shift tab') is True
        assert keybindings.is_navigation_key('ctrl 1') is True
        assert keybindings.is_navigation_key('ctrl 9') is True
        
        assert keybindings.is_navigation_key('ctrl n') is False
        assert keybindings.is_navigation_key('ctrl q') is False
        assert keybindings.is_navigation_key('unknown') is False
    
    def test_is_pane_management_key(self, keybindings):
        """Test pane management key detection."""
        assert keybindings.is_pane_management_key('ctrl n') is True
        assert keybindings.is_pane_management_key('ctrl x') is True
        assert keybindings.is_pane_management_key('ctrl r') is True
        assert keybindings.is_pane_management_key('ctrl s') is True
        assert keybindings.is_pane_management_key('ctrl l') is True
        
        assert keybindings.is_pane_management_key('ctrl tab') is False
        assert keybindings.is_pane_management_key('ctrl q') is False
        assert keybindings.is_pane_management_key('unknown') is False
    
    def test_is_layout_key(self, keybindings):
        """Test layout key detection."""
        assert keybindings.is_layout_key('ctrl h') is True
        assert keybindings.is_layout_key('ctrl v') is True
        assert keybindings.is_layout_key('ctrl +') is True
        assert keybindings.is_layout_key('ctrl -') is True
        
        assert keybindings.is_layout_key('ctrl n') is False
        assert keybindings.is_layout_key('ctrl q') is False
        assert keybindings.is_layout_key('unknown') is False
    
    def test_binding_categories(self, keybindings):
        """Test that bindings are properly categorized."""
        # Navigation bindings
        navigation_bindings = [
            'ctrl n', 'ctrl tab', 'ctrl shift tab',
            'ctrl 1', 'ctrl 2', 'ctrl 3', 'ctrl 4', 'ctrl 5',
            'ctrl 6', 'ctrl 7', 'ctrl 8', 'ctrl 9'
        ]
        
        # Pane management bindings
        management_bindings = [
            'ctrl x', 'ctrl r', 'ctrl s', 'ctrl l'
        ]
        
        # Layout bindings
        layout_bindings = [
            'ctrl h', 'ctrl v', 'ctrl +', 'ctrl -'
        ]
        
        # Search bindings
        search_bindings = [
            'ctrl /', 'ctrl f', 'ctrl g'
        ]
        
        # Session bindings
        session_bindings = [
            'ctrl d', 'ctrl z', 'ctrl c'
        ]
        
        # Help bindings
        help_bindings = [
            'f1', 'f2', 'f3'
        ]
        
        # Exit bindings
        exit_bindings = [
            'ctrl q', 'ctrl \\'
        ]
        
        # Verify all bindings exist
        all_bindings = (
            navigation_bindings + management_bindings + 
            layout_bindings + search_bindings + 
            session_bindings + help_bindings + exit_bindings
        )
        
        for binding in all_bindings:
            assert binding in keybindings.bindings, f"Missing binding: {binding}"
    
    def test_help_text_structure(self, keybindings):
        """Test that help text has proper structure."""
        help_text = keybindings.get_help_text()
        
        # Should contain all categories
        categories = [
            'Navigation:', 'Pane Management:', 'Layout:', 
            'Search:', 'Session:', 'Help:', 'Exit:'
        ]
        
        for category in categories:
            assert category in help_text
        
        # Should contain key descriptions
        assert 'New pane' in help_text
        assert 'Next pane' in help_text
        assert 'Previous pane' in help_text
        assert 'Close pane' in help_text
        assert 'Save session' in help_text
        assert 'Quit' in help_text 