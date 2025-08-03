"""
Tests for the LayoutManager class.
"""

import pytest
import urwid
from unittest.mock import Mock
from tui.layout import LayoutManager, PaneInfo


class TestLayoutManager:
    """Test cases for LayoutManager."""
    
    @pytest.fixture
    def layout_manager(self):
        """Create a LayoutManager instance for testing."""
        return LayoutManager()
    
    @pytest.fixture
    def mock_widget(self):
        """Create a mock widget for testing."""
        widget = Mock(spec=urwid.Widget)
        widget.render.return_value = (None, [])
        return widget
    
    def test_init(self, layout_manager):
        """Test LayoutManager initialization."""
        assert layout_manager.panes == {}
        assert layout_manager.layout_widget is None
        assert layout_manager.active_pane is None
        assert layout_manager.layout_type == 'horizontal'
    
    def test_add_pane(self, layout_manager, mock_widget):
        """Test adding a pane."""
        layout_manager.add_pane("test_pane", mock_widget, 1.5)
        
        assert "test_pane" in layout_manager.panes
        pane_info = layout_manager.panes["test_pane"]
        assert pane_info.name == "test_pane"
        assert pane_info.widget == mock_widget
        assert pane_info.size_ratio == 1.5
        assert pane_info.is_active is True
        assert layout_manager.active_pane == "test_pane"
    
    def test_add_multiple_panes(self, layout_manager, mock_widget):
        """Test adding multiple panes."""
        layout_manager.add_pane("pane1", mock_widget, 1.0)
        layout_manager.add_pane("pane2", mock_widget, 2.0)
        
        assert len(layout_manager.panes) == 2
        assert "pane1" in layout_manager.panes
        assert "pane2" in layout_manager.panes
        assert layout_manager.active_pane == "pane1"  # First pane should be active
    
    def test_remove_pane(self, layout_manager, mock_widget):
        """Test removing a pane."""
        layout_manager.add_pane("pane1", mock_widget)
        layout_manager.add_pane("pane2", mock_widget)
        
        assert len(layout_manager.panes) == 2
        assert layout_manager.active_pane == "pane1"
        
        layout_manager.remove_pane("pane1")
        
        assert len(layout_manager.panes) == 1
        assert "pane1" not in layout_manager.panes
        assert layout_manager.active_pane == "pane2"  # Should switch to remaining pane
    
    def test_remove_nonexistent_pane(self, layout_manager):
        """Test removing a non-existent pane."""
        # Should not raise an exception
        layout_manager.remove_pane("nonexistent")
    
    def test_set_active_pane(self, layout_manager, mock_widget):
        """Test setting the active pane."""
        layout_manager.add_pane("pane1", mock_widget)
        layout_manager.add_pane("pane2", mock_widget)
        
        # Initially pane1 should be active
        assert layout_manager.active_pane == "pane1"
        assert layout_manager.panes["pane1"].is_active is True
        assert layout_manager.panes["pane2"].is_active is False
        
        # Set pane2 as active
        layout_manager.set_active_pane("pane2")
        
        assert layout_manager.active_pane == "pane2"
        assert layout_manager.panes["pane1"].is_active is False
        assert layout_manager.panes["pane2"].is_active is True
    
    def test_set_active_nonexistent_pane(self, layout_manager):
        """Test setting a non-existent pane as active."""
        # Should not raise an exception
        layout_manager.set_active_pane("nonexistent")
    
    def test_get_active_pane(self, layout_manager, mock_widget):
        """Test getting the active pane."""
        assert layout_manager.get_active_pane() is None
        
        layout_manager.add_pane("test_pane", mock_widget)
        assert layout_manager.get_active_pane() == "test_pane"
    
    def test_get_pane_widget(self, layout_manager, mock_widget):
        """Test getting a pane widget."""
        layout_manager.add_pane("test_pane", mock_widget)
        
        widget = layout_manager.get_pane_widget("test_pane")
        assert widget == mock_widget
        
        nonexistent = layout_manager.get_pane_widget("nonexistent")
        assert nonexistent is None
    
    def test_resize_pane(self, layout_manager, mock_widget):
        """Test resizing a pane."""
        layout_manager.add_pane("test_pane", mock_widget, 1.0)
        
        # Resize the pane
        layout_manager.resize_pane("test_pane", 2.5)
        
        assert layout_manager.panes["test_pane"].size_ratio == 2.5
    
    def test_resize_nonexistent_pane(self, layout_manager):
        """Test resizing a non-existent pane."""
        # Should not raise an exception
        layout_manager.resize_pane("nonexistent", 2.0)
    
    def test_resize_pane_minimum_ratio(self, layout_manager, mock_widget):
        """Test that pane size ratio has a minimum value."""
        layout_manager.add_pane("test_pane", mock_widget, 1.0)
        
        # Try to set a very small ratio
        layout_manager.resize_pane("test_pane", 0.05)
        
        # Should be clamped to minimum
        assert layout_manager.panes["test_pane"].size_ratio == 0.1
    
    def test_split_horizontal(self, layout_manager, mock_widget):
        """Test horizontal splitting."""
        layout_manager.add_pane("pane1", mock_widget)
        
        layout_manager.split_horizontal()
        
        assert len(layout_manager.panes) == 2
        assert "pane_2" in layout_manager.panes
        assert layout_manager.active_pane == "pane_2"
    
    def test_split_vertical(self, layout_manager, mock_widget):
        """Test vertical splitting."""
        layout_manager.add_pane("pane1", mock_widget)
        
        layout_manager.split_vertical()
        
        assert len(layout_manager.panes) == 2
        assert "pane_2" in layout_manager.panes
        assert layout_manager.active_pane == "pane_2"
    
    def test_split_without_active_pane(self, layout_manager):
        """Test splitting when no active pane."""
        # Should not raise an exception
        layout_manager.split_horizontal()
        layout_manager.split_vertical()
    
    def test_increase_pane_size(self, layout_manager, mock_widget):
        """Test increasing pane size."""
        layout_manager.add_pane("test_pane", mock_widget, 1.0)
        
        original_ratio = layout_manager.panes["test_pane"].size_ratio
        layout_manager.increase_pane_size()
        
        new_ratio = layout_manager.panes["test_pane"].size_ratio
        assert new_ratio > original_ratio
    
    def test_decrease_pane_size(self, layout_manager, mock_widget):
        """Test decreasing pane size."""
        layout_manager.add_pane("test_pane", mock_widget, 2.0)
        
        original_ratio = layout_manager.panes["test_pane"].size_ratio
        layout_manager.decrease_pane_size()
        
        new_ratio = layout_manager.panes["test_pane"].size_ratio
        assert new_ratio < original_ratio
    
    def test_get_layout_widget_single_pane(self, layout_manager, mock_widget):
        """Test getting layout widget with single pane."""
        layout_manager.add_pane("test_pane", mock_widget)
        
        layout_widget = layout_manager.get_layout_widget()
        assert layout_widget == mock_widget
    
    def test_get_layout_widget_multiple_panes(self, layout_manager, mock_widget):
        """Test getting layout widget with multiple panes."""
        layout_manager.add_pane("pane1", mock_widget)
        layout_manager.add_pane("pane2", mock_widget)
        
        layout_widget = layout_manager.get_layout_widget()
        assert layout_widget is not None
        assert isinstance(layout_widget, urwid.Columns)
    
    def test_get_layout_widget_no_panes(self, layout_manager):
        """Test getting layout widget with no panes."""
        layout_widget = layout_manager.get_layout_widget()
        assert isinstance(layout_widget, urwid.Text)
        assert "No panes available" in layout_widget.text
    
    def test_get_pane_names(self, layout_manager, mock_widget):
        """Test getting pane names."""
        assert layout_manager.get_pane_names() == []
        
        layout_manager.add_pane("pane1", mock_widget)
        layout_manager.add_pane("pane2", mock_widget)
        
        names = layout_manager.get_pane_names()
        assert "pane1" in names
        assert "pane2" in names
        assert len(names) == 2
    
    def test_get_pane_info(self, layout_manager, mock_widget):
        """Test getting pane info."""
        layout_manager.add_pane("test_pane", mock_widget, 1.5)
        
        pane_info = layout_manager.get_pane_info("test_pane")
        assert pane_info is not None
        assert pane_info.name == "test_pane"
        assert pane_info.widget == mock_widget
        assert pane_info.size_ratio == 1.5
        assert pane_info.is_active is True
        
        nonexistent = layout_manager.get_pane_info("nonexistent")
        assert nonexistent is None
    
    def test_set_layout_type(self, layout_manager):
        """Test setting layout type."""
        assert layout_manager.layout_type == 'horizontal'
        
        layout_manager.set_layout_type('vertical')
        assert layout_manager.layout_type == 'vertical'
        
        layout_manager.set_layout_type('grid')
        assert layout_manager.layout_type == 'grid'
        
        # Invalid layout type should not change
        layout_manager.set_layout_type('invalid')
        assert layout_manager.layout_type == 'grid'
    
    def test_get_status_bar_text(self, layout_manager, mock_widget):
        """Test getting status bar text."""
        # No panes
        status = layout_manager.get_status_bar_text()
        assert status == "No panes"
        
        # With panes
        layout_manager.add_pane("pane1", mock_widget)
        layout_manager.add_pane("pane2", mock_widget)
        
        status = layout_manager.get_status_bar_text()
        assert "Panes: 2" in status
        assert "Active: pane1" in status
        assert "Layout: horizontal" in status
    
    def test_update_layout_with_active_pane(self, layout_manager, mock_widget):
        """Test layout update with active pane highlighting."""
        layout_manager.add_pane("pane1", mock_widget)
        layout_manager.add_pane("pane2", mock_widget)
        
        # Set pane2 as active
        layout_manager.set_active_pane("pane2")
        
        layout_widget = layout_manager.get_layout_widget()
        assert isinstance(layout_widget, urwid.Columns)
        
        # The active pane should have a different title
        # (This is tested indirectly through the layout widget structure)
        assert layout_widget is not None 