"""
Layout manager for handling split panes and terminal layout.
"""

import urwid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class PaneInfo:
    """Information about a pane in the layout."""
    name: str
    widget: urwid.Widget
    is_active: bool = False
    size_ratio: float = 1.0


class LayoutManager:
    """Manages the layout of terminal panes."""
    
    def __init__(self):
        self.panes: Dict[str, PaneInfo] = {}
        self.layout_widget = None
        self.active_pane = None
        self.layout_type = 'horizontal'  # 'horizontal', 'vertical', 'grid'
        
    def add_pane(self, name: str, widget: urwid.Widget, size_ratio: float = 1.0):
        """Add a pane to the layout."""
        pane_info = PaneInfo(
            name=name,
            widget=widget,
            size_ratio=size_ratio
        )
        
        self.panes[name] = pane_info
        
        if not self.active_pane:
            self.active_pane = name
            pane_info.is_active = True
        
        self._update_layout()
    
    def remove_pane(self, name: str):
        """Remove a pane from the layout."""
        if name in self.panes:
            del self.panes[name]
            
            # Update active pane if needed
            if self.active_pane == name:
                self.active_pane = next(iter(self.panes.keys()), None)
                if self.active_pane:
                    self.panes[self.active_pane].is_active = True
            
            self._update_layout()
    
    def set_active_pane(self, name: str):
        """Set the active pane."""
        if name in self.panes:
            # Clear previous active pane
            if self.active_pane and self.active_pane in self.panes:
                self.panes[self.active_pane].is_active = False
            
            self.active_pane = name
            self.panes[name].is_active = True
            self._update_layout()
    
    def get_active_pane(self) -> Optional[str]:
        """Get the name of the active pane."""
        return self.active_pane
    
    def get_pane_widget(self, name: str) -> Optional[urwid.Widget]:
        """Get the widget for a specific pane."""
        if name in self.panes:
            return self.panes[name].widget
        return None
    
    def resize_pane(self, name: str, size_ratio: float):
        """Resize a pane by changing its size ratio."""
        if name in self.panes:
            self.panes[name].size_ratio = max(0.1, size_ratio)
            self._update_layout()
    
    def split_horizontal(self):
        """Split the active pane horizontally."""
        if not self.active_pane:
            return
        
        # Create a new pane name
        new_pane_name = f"pane_{len(self.panes) + 1}"
        
        # Create a simple text widget for the new pane
        new_widget = urwid.Text(f"New pane: {new_pane_name}")
        
        # Add the new pane
        self.add_pane(new_pane_name, new_widget, 1.0)
        
        # Set it as active
        self.set_active_pane(new_pane_name)
    
    def split_vertical(self):
        """Split the active pane vertically."""
        if not self.active_pane:
            return
        
        # Create a new pane name
        new_pane_name = f"pane_{len(self.panes) + 1}"
        
        # Create a simple text widget for the new pane
        new_widget = urwid.Text(f"New pane: {new_pane_name}")
        
        # Add the new pane
        self.add_pane(new_pane_name, new_widget, 1.0)
        
        # Set it as active
        self.set_active_pane(new_pane_name)
    
    def increase_pane_size(self):
        """Increase the size of the active pane."""
        if self.active_pane:
            current_ratio = self.panes[self.active_pane].size_ratio
            self.resize_pane(self.active_pane, current_ratio * 1.2)
    
    def decrease_pane_size(self):
        """Decrease the size of the active pane."""
        if self.active_pane:
            current_ratio = self.panes[self.active_pane].size_ratio
            self.resize_pane(self.active_pane, current_ratio * 0.8)
    
    def _update_layout(self):
        """Update the layout widget based on current panes."""
        if not self.panes:
            # No panes, create empty layout
            self.layout_widget = urwid.Text("No panes available")
            return
        
        if len(self.panes) == 1:
            # Single pane
            pane_name = list(self.panes.keys())[0]
            self.layout_widget = self.panes[pane_name].widget
        else:
            # Multiple panes - create columns or rows
            widgets = []
            weights = []
            
            for name, pane_info in self.panes.items():
                # Add border to active pane
                if pane_info.is_active:
                    widget = urwid.LineBox(pane_info.widget, title=f"*{name}*")
                else:
                    widget = urwid.LineBox(pane_info.widget, title=name)
                
                widgets.append(widget)
                weights.append(int(pane_info.size_ratio * 10))
            
            if self.layout_type == 'horizontal':
                self.layout_widget = urwid.Columns(widgets, weights)
            elif self.layout_type == 'vertical':
                self.layout_widget = urwid.Pile(widgets)
            else:
                # Grid layout (simple 2x2 for now)
                if len(widgets) <= 2:
                    self.layout_widget = urwid.Columns(widgets, weights)
                else:
                    # Create a 2x2 grid
                    top_row = urwid.Columns(widgets[:2], weights[:2])
                    bottom_row = urwid.Columns(widgets[2:4], weights[2:4]) if len(widgets) > 2 else urwid.Text("")
                    self.layout_widget = urwid.Pile([top_row, bottom_row])
    
    def get_layout_widget(self) -> urwid.Widget:
        """Get the current layout widget."""
        if not self.layout_widget:
            self._update_layout()
        return self.layout_widget
    
    def get_pane_names(self) -> List[str]:
        """Get list of all pane names."""
        return list(self.panes.keys())
    
    def get_pane_info(self, name: str) -> Optional[PaneInfo]:
        """Get information about a specific pane."""
        return self.panes.get(name)
    
    def set_layout_type(self, layout_type: str):
        """Set the layout type."""
        if layout_type in ['horizontal', 'vertical', 'grid']:
            self.layout_type = layout_type
            self._update_layout()
    
    def get_status_bar_text(self) -> str:
        """Get status bar text showing pane information."""
        if not self.panes:
            return "No panes"
        
        pane_names = list(self.panes.keys())
        active_index = pane_names.index(self.active_pane) if self.active_pane else 0
        
        status = f"Panes: {len(self.panes)} | Active: {self.active_pane or 'None'}"
        status += f" | Layout: {self.layout_type}"
        
        return status 