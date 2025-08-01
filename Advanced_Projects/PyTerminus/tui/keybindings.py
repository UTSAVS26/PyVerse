"""
Keyboard shortcuts and keybindings for PyTerminus.
"""

from typing import Dict, Callable, Any
import urwid


class KeyBindings:
    """Manages keyboard shortcuts and keybindings."""
    
    def __init__(self):
        self.bindings: Dict[str, Callable] = {}
        self._setup_default_bindings()
    
    def _setup_default_bindings(self):
        """Setup default keybindings."""
        # Navigation
        self.bindings['ctrl n'] = 'new_pane'
        self.bindings['ctrl tab'] = 'next_pane'
        self.bindings['ctrl shift tab'] = 'prev_pane'
        self.bindings['ctrl 1'] = 'switch_pane_1'
        self.bindings['ctrl 2'] = 'switch_pane_2'
        self.bindings['ctrl 3'] = 'switch_pane_3'
        self.bindings['ctrl 4'] = 'switch_pane_4'
        self.bindings['ctrl 5'] = 'switch_pane_5'
        self.bindings['ctrl 6'] = 'switch_pane_6'
        self.bindings['ctrl 7'] = 'switch_pane_7'
        self.bindings['ctrl 8'] = 'switch_pane_8'
        self.bindings['ctrl 9'] = 'switch_pane_9'
        
        # Pane management
        self.bindings['ctrl x'] = 'close_pane'
        self.bindings['ctrl r'] = 'rename_pane'
        self.bindings['ctrl s'] = 'save_session'
        self.bindings['ctrl l'] = 'load_session'
        
        # Layout
        self.bindings['ctrl h'] = 'split_horizontal'
        self.bindings['ctrl v'] = 'split_vertical'
        self.bindings['ctrl +'] = 'increase_pane_size'
        self.bindings['ctrl -'] = 'decrease_pane_size'
        
        # Search and history
        self.bindings['ctrl /'] = 'search_history'
        self.bindings['ctrl f'] = 'search_output'
        self.bindings['ctrl g'] = 'clear_search'
        
        # Session management
        self.bindings['ctrl d'] = 'detach_session'
        self.bindings['ctrl z'] = 'suspend_session'
        self.bindings['ctrl c'] = 'copy_mode'
        
        # Help and info
        self.bindings['f1'] = 'show_help'
        self.bindings['f2'] = 'show_status'
        self.bindings['f3'] = 'show_logs'
        
        # Exit
        self.bindings['ctrl q'] = 'quit'
        self.bindings['ctrl \\'] = 'force_quit'
    
    def get_binding(self, key: str) -> str:
        """Get the action bound to a key."""
        return self.bindings.get(key, 'unknown')
    
    def set_binding(self, key: str, action: str):
        """Set a key binding."""
        self.bindings[key] = action
    
    def remove_binding(self, key: str):
        """Remove a key binding."""
        if key in self.bindings:
            del self.bindings[key]
    
    def get_all_bindings(self) -> Dict[str, str]:
        """Get all current bindings."""
        return self.bindings.copy()
    
    def get_help_text(self) -> str:
        """Get formatted help text for all bindings."""
        help_text = "PyTerminus Key Bindings:\n\n"
        
        # Group bindings by category
        categories = {
            'Navigation': [
                ('ctrl n', 'New pane'),
                ('ctrl tab', 'Next pane'),
                ('ctrl shift tab', 'Previous pane'),
                ('ctrl 1-9', 'Switch to pane 1-9')
            ],
            'Pane Management': [
                ('ctrl x', 'Close pane'),
                ('ctrl r', 'Rename pane'),
                ('ctrl s', 'Save session'),
                ('ctrl l', 'Load session')
            ],
            'Layout': [
                ('ctrl h', 'Split horizontal'),
                ('ctrl v', 'Split vertical'),
                ('ctrl +', 'Increase pane size'),
                ('ctrl -', 'Decrease pane size')
            ],
            'Search': [
                ('ctrl /', 'Search history'),
                ('ctrl f', 'Search output'),
                ('ctrl g', 'Clear search')
            ],
            'Session': [
                ('ctrl d', 'Detach session'),
                ('ctrl z', 'Suspend session'),
                ('ctrl c', 'Copy mode')
            ],
            'Help': [
                ('f1', 'Show help'),
                ('f2', 'Show status'),
                ('f3', 'Show logs')
            ],
            'Exit': [
                ('ctrl q', 'Quit'),
                ('ctrl \\', 'Force quit')
            ]
        }
        
        for category, bindings in categories.items():
            help_text += f"{category}:\n"
            for key, description in bindings:
                help_text += f"  {key:<15} {description}\n"
            help_text += "\n"
        
        return help_text
    
    def parse_key(self, key: str) -> str:
        """Parse a key event and return the corresponding action."""
        # Handle special keys
        if key == 'tab':
            return self.get_binding('ctrl tab')
        elif key == 'shift tab':
            return self.get_binding('ctrl shift tab')
        elif key.startswith('ctrl '):
            return self.get_binding(key)
        elif key.startswith('f') and key[1:].isdigit():
            return self.get_binding(key)
        else:
            return 'unknown'
    
    def is_navigation_key(self, key: str) -> bool:
        """Check if a key is used for navigation."""
        navigation_keys = [
            'ctrl tab', 'ctrl shift tab', 'ctrl 1', 'ctrl 2', 'ctrl 3',
            'ctrl 4', 'ctrl 5', 'ctrl 6', 'ctrl 7', 'ctrl 8', 'ctrl 9'
        ]
        return key in navigation_keys
    
    def is_pane_management_key(self, key: str) -> bool:
        """Check if a key is used for pane management."""
        management_keys = [
            'ctrl n', 'ctrl x', 'ctrl r', 'ctrl s', 'ctrl l'
        ]
        return key in management_keys
    
    def is_layout_key(self, key: str) -> bool:
        """Check if a key is used for layout management."""
        layout_keys = [
            'ctrl h', 'ctrl v', 'ctrl +', 'ctrl -'
        ]
        return key in layout_keys 