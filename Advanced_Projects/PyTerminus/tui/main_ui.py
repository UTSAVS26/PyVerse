"""
Main UI component for PyTerminus.
"""

import urwid
import sys
import time
from typing import Optional, Dict, Any
from .layout import LayoutManager
from .keybindings import KeyBindings


class TerminalWidget(urwid.Widget):
    """Widget for displaying terminal output."""
    
    def __init__(self, pane_name: str, session_manager):
        self.pane_name = pane_name
        self.session_manager = session_manager
        self.output_text = ""
        self.max_lines = 1000
        self.lines = []
        
        # Create the text widget
        self.text_widget = urwid.Text("")
        super().__init__()
    
    def update_output(self, output: str):
        """Update the terminal output."""
        self.output_text += output
        
        # Split into lines and keep only recent ones
        self.lines = self.output_text.split('\n')
        if len(self.lines) > self.max_lines:
            self.lines = self.lines[-self.max_lines:]
        
        # Update the text widget
        display_text = '\n'.join(self.lines[-50:])  # Show last 50 lines
        self.text_widget.set_text(display_text)
    
    def render(self, size, focus=False):
        """Render the widget."""
        return self.text_widget.render(size, focus)
    
    def selectable(self):
        """Make the widget selectable."""
        return True
    
    def keypress(self, size, key):
        """Handle keypress events."""
        # Send the key to the terminal pane
        if self.session_manager.write_to_active_pane(key):
            return None
        return key


class MainUI:
    """Main UI class for PyTerminus."""
    
    def __init__(self, session_manager, theme='dark', logger=None):
        self.session_manager = session_manager
        self.logger = logger
        self.theme = theme
        
        # Initialize components
        self.layout_manager = LayoutManager()
        self.keybindings = KeyBindings()
        
        # UI state
        self.current_mode = 'normal'  # 'normal', 'search', 'help'
        self.search_query = ""
        self.help_visible = False
        self.status_visible = False
        
        # Create main UI components
        self._create_ui()
        
        # Start with a default pane
        self._create_default_pane()
    
    def _create_ui(self):
        """Create the main UI components."""
        # Create status bar
        self.status_bar = urwid.Text("PyTerminus - Ready")
        
        # Create help overlay
        self.help_overlay = urwid.Overlay(
            urwid.LineBox(
                urwid.Text(self.keybindings.get_help_text()),
                title="Help"
            ),
            urwid.SolidFill(),
            'center', ('relative', 80),
            'middle', ('relative', 80)
        )
        
        # Create search overlay
        self.search_overlay = urwid.Overlay(
            urwid.LineBox(
                urwid.Edit("Search: "),
                title="Search"
            ),
            urwid.SolidFill(),
            'center', ('relative', 60),
            'middle', ('relative', 10)
        )
        
        # Create main layout
        self.main_widget = urwid.Frame(
            body=urwid.SolidFill(),
            footer=self.status_bar
        )
        
        # Create the main loop
        self.loop = urwid.MainLoop(
            self.main_widget,
            self._get_palette(),
            unhandled_input=self._handle_input
        )
    
    def _get_palette(self):
        """Get the color palette based on theme."""
        if self.theme == 'dark':
            return [
                ('default', 'white', 'black'),
                ('status', 'white', 'dark blue'),
                ('active', 'white', 'dark green'),
                ('inactive', 'light gray', 'black'),
                ('error', 'white', 'dark red'),
                ('help', 'white', 'dark blue'),
                ('search', 'white', 'dark magenta'),
            ]
        else:
            return [
                ('default', 'black', 'white'),
                ('status', 'black', 'light blue'),
                ('active', 'black', 'light green'),
                ('inactive', 'dark gray', 'white'),
                ('error', 'white', 'red'),
                ('help', 'black', 'light blue'),
                ('search', 'black', 'light magenta'),
            ]
    
    def _create_default_pane(self):
        """Create a default pane if none exists."""
        if not self.session_manager.get_all_panes():
            # Create a default shell pane
            self.session_manager.create_pane("shell", "bash")
            
            # Create terminal widget for the pane
            terminal_widget = TerminalWidget("shell", self.session_manager)
            
            # Add to layout
            self.layout_manager.add_pane("shell", terminal_widget)
            
            # Update main widget
            self.main_widget.body = self.layout_manager.get_layout_widget()
    
    def _handle_input(self, key):
        """Handle input events."""
        if key in ('q', 'Q'):
            raise urwid.ExitMainLoop()
        
        # Handle special keys
        if key == 'tab':
            action = self.keybindings.get_binding('ctrl tab')
        elif key == 'shift tab':
            action = self.keybindings.get_binding('ctrl shift tab')
        elif key.startswith('ctrl '):
            action = self.keybindings.get_binding(key)
        elif key.startswith('f') and key[1:].isdigit():
            action = self.keybindings.get_binding(key)
        else:
            # Pass to active terminal
            if self.session_manager.write_to_active_pane(key):
                return
            action = 'unknown'
        
        # Handle actions
        self._handle_action(action)
    
    def _handle_action(self, action: str):
        """Handle UI actions."""
        if action == 'new_pane':
            self._create_new_pane()
        elif action == 'next_pane':
            self._switch_to_next_pane()
        elif action == 'prev_pane':
            self._switch_to_prev_pane()
        elif action == 'close_pane':
            self._close_active_pane()
        elif action == 'save_session':
            self._save_session()
        elif action == 'load_session':
            self._load_session()
        elif action == 'split_horizontal':
            self._split_horizontal()
        elif action == 'split_vertical':
            self._split_vertical()
        elif action == 'increase_pane_size':
            self._increase_pane_size()
        elif action == 'decrease_pane_size':
            self._decrease_pane_size()
        elif action == 'search_history':
            self._show_search()
        elif action == 'show_help':
            self._show_help()
        elif action == 'show_status':
            self._show_status()
        elif action == 'quit':
            raise urwid.ExitMainLoop()
        elif action.startswith('switch_pane_'):
            pane_num = action.split('_')[-1]
            self._switch_to_pane_number(int(pane_num))
    
    def _create_new_pane(self):
        """Create a new terminal pane."""
        pane_count = len(self.session_manager.get_all_panes())
        new_pane_name = f"pane_{pane_count + 1}"
        
        # Create pane in session manager
        if self.session_manager.create_pane(new_pane_name, "bash"):
            # Create terminal widget
            terminal_widget = TerminalWidget(new_pane_name, self.session_manager)
            
            # Add to layout
            self.layout_manager.add_pane(new_pane_name, terminal_widget)
            
            # Set as active
            self.session_manager.set_active_pane(new_pane_name)
            self.layout_manager.set_active_pane(new_pane_name)
            
            # Update UI
            self._update_main_widget()
            self._update_status_bar()
    
    def _switch_to_next_pane(self):
        """Switch to the next pane."""
        panes = list(self.session_manager.get_all_panes().keys())
        if not panes:
            return
        
        current_pane = self.session_manager.get_active_pane()
        if current_pane in panes:
            current_index = panes.index(current_pane)
            next_index = (current_index + 1) % len(panes)
            next_pane = panes[next_index]
        else:
            next_pane = panes[0]
        
        self.session_manager.set_active_pane(next_pane)
        self.layout_manager.set_active_pane(next_pane)
        self._update_main_widget()
        self._update_status_bar()
    
    def _switch_to_prev_pane(self):
        """Switch to the previous pane."""
        panes = list(self.session_manager.get_all_panes().keys())
        if not panes:
            return
        
        current_pane = self.session_manager.get_active_pane()
        if current_pane in panes:
            current_index = panes.index(current_pane)
            prev_index = (current_index - 1) % len(panes)
            prev_pane = panes[prev_index]
        else:
            prev_pane = panes[0]
        
        self.session_manager.set_active_pane(prev_pane)
        self.layout_manager.set_active_pane(prev_pane)
        self._update_main_widget()
        self._update_status_bar()
    
    def _switch_to_pane_number(self, pane_num: int):
        """Switch to a specific pane by number."""
        panes = list(self.session_manager.get_all_panes().keys())
        if 1 <= pane_num <= len(panes):
            target_pane = panes[pane_num - 1]
            self.session_manager.set_active_pane(target_pane)
            self.layout_manager.set_active_pane(target_pane)
            self._update_main_widget()
            self._update_status_bar()
    
    def _close_active_pane(self):
        """Close the active pane."""
        active_pane = self.session_manager.get_active_pane()
        if active_pane and len(self.session_manager.get_all_panes()) > 1:
            self.session_manager.remove_pane(active_pane)
            self.layout_manager.remove_pane(active_pane)
            self._update_main_widget()
            self._update_status_bar()
    
    def _save_session(self):
        """Save the current session."""
        filename = f"session_{int(time.time())}.yaml"
        if self.session_manager.save_session(filename):
            self._update_status_bar(f"Session saved: {filename}")
        else:
            self._update_status_bar("Failed to save session")
    
    def _load_session(self):
        """Load a session."""
        # For now, just show a message
        self._update_status_bar("Load session: Not implemented yet")
    
    def _split_horizontal(self):
        """Split the active pane horizontally."""
        self.layout_manager.split_horizontal()
        self._update_main_widget()
        self._update_status_bar()
    
    def _split_vertical(self):
        """Split the active pane vertically."""
        self.layout_manager.split_vertical()
        self._update_main_widget()
        self._update_status_bar()
    
    def _increase_pane_size(self):
        """Increase the size of the active pane."""
        active_pane = self.layout_manager.get_active_pane()
        if active_pane:
            self.layout_manager.increase_pane_size()
            self._update_main_widget()
    
    def _decrease_pane_size(self):
        """Decrease the size of the active pane."""
        active_pane = self.layout_manager.get_active_pane()
        if active_pane:
            self.layout_manager.decrease_pane_size()
            self._update_main_widget()
    
    def _show_search(self):
        """Show the search interface."""
        self.current_mode = 'search'
        self.loop.widget = self.search_overlay
    
    def _show_help(self):
        """Show the help interface."""
        self.current_mode = 'help'
        self.loop.widget = self.help_overlay
    
    def _show_status(self):
        """Show status information."""
        status_info = self.session_manager.get_session_info()
        self._update_status_bar(status_info)
    
    def _update_main_widget(self):
        """Update the main widget."""
        self.main_widget.body = self.layout_manager.get_layout_widget()
    
    def _update_status_bar(self, message: str = None):
        """Update the status bar."""
        if message:
            self.status_bar.set_text(message)
        else:
            status = self.layout_manager.get_status_bar_text()
            self.status_bar.set_text(status)
    
    def run(self):
        """Run the main UI loop."""
        try:
            # Log session start
            if self.logger:
                self.logger.log_session_started("default", len(self.session_manager.get_all_panes()))
            
            # Update status bar
            self._update_status_bar()
            
            # Run the main loop
            self.loop.run()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        finally:
            # Cleanup
            self.session_manager.cleanup()
            
            # Log session end
            if self.logger:
                self.logger.log_session_ended("default", 0.0) 