"""
Session manager for handling multiple terminal panes and session persistence.
"""

import yaml
import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from .terminal_pane import TerminalPane, PaneConfig


@dataclass
class SessionConfig:
    """Configuration for a session."""
    name: str
    panes: List[Dict[str, Any]]
    created_at: Optional[str] = None
    last_modified: Optional[str] = None


class SessionManager:
    """Manages multiple terminal panes and session persistence."""
    
    def __init__(self, default_shell: str = 'bash', log_dir: Path = None, logger=None):
        self.default_shell = default_shell
        self.log_dir = log_dir or Path('./logs')
        self.logger = logger
        
        self.panes: Dict[str, TerminalPane] = {}
        self.active_pane = None
        self.session_name = "default"
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def create_pane(self, name: str, command: str = None, cwd: str = None, 
                   shell: str = None) -> bool:
        """Create a new terminal pane."""
        if name in self.panes:
            print(f"Pane '{name}' already exists")
            return False
        
        # Use default shell if no command specified
        if not command:
            command = shell or self.default_shell
        
        # Create pane configuration
        config = PaneConfig(
            name=name,
            command=command,
            cwd=cwd,
            shell=shell or self.default_shell
        )
        
        # Create and start the pane
        pane = TerminalPane(config, output_callback=self._on_pane_output)
        
        if pane.start():
            self.panes[name] = pane
            if not self.active_pane:
                self.active_pane = name
            
            if self.logger:
                self.logger.log_pane_created(name, command, cwd)
            
            return True
        else:
            print(f"Failed to start pane '{name}'")
            return False
    
    def remove_pane(self, name: str) -> bool:
        """Remove a terminal pane."""
        if name not in self.panes:
            return False
        
        pane = self.panes[name]
        pane.stop()
        del self.panes[name]
        
        # Update active pane if needed
        if self.active_pane == name:
            self.active_pane = next(iter(self.panes.keys()), None)
        
        if self.logger:
            self.logger.log_pane_removed(name)
        
        return True
    
    def get_pane(self, name: str) -> Optional[TerminalPane]:
        """Get a terminal pane by name."""
        return self.panes.get(name)
    
    def get_active_pane(self) -> Optional[TerminalPane]:
        """Get the currently active pane."""
        if self.active_pane and self.active_pane in self.panes:
            return self.panes[self.active_pane]
        return None
    
    def set_active_pane(self, name: str) -> bool:
        """Set the active pane."""
        if name in self.panes:
            self.active_pane = name
            return True
        return False
    
    def write_to_active_pane(self, data: str) -> bool:
        """Write data to the active pane."""
        pane = self.get_active_pane()
        if pane:
            return pane.write_input(data)
        return False
    
    def get_all_panes(self) -> Dict[str, TerminalPane]:
        """Get all panes."""
        return self.panes.copy()
    
    def get_pane_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific pane."""
        pane = self.get_pane(name)
        if pane:
            return pane.get_status()
        return None
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all panes."""
        return {name: pane.get_status() for name, pane in self.panes.items()}
    
    def resize_pane(self, name: str, width: int, height: int) -> bool:
        """Resize a pane."""
        pane = self.get_pane(name)
        if pane:
            pane.resize(width, height)
            return True
        return False
    
    def save_session(self, filename: str = None) -> bool:
        """Save the current session to a file."""
        if not filename:
            filename = f"{self.session_name}.yaml"
        
        session_data = {
            'name': self.session_name,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'last_modified': time.strftime('%Y-%m-%d %H:%M:%S'),
            'panes': []
        }
        
        for name, pane in self.panes.items():
            pane_data = {
                'name': name,
                'command': pane.config.command,
                'cwd': pane.config.cwd,
                'shell': pane.config.shell
            }
            session_data['panes'].append(pane_data)
        
        try:
            filepath = Path(filename)
            with open(filepath, 'w') as f:
                yaml.dump(session_data, f, default_flow_style=False)
            
            if self.logger:
                self.logger.log_session_saved(filename, len(self.panes))
            
            return True
        except Exception as e:
            print(f"Error saving session: {e}")
            return False
    
    def load_session(self, filename: str) -> bool:
        """Load a session from a file."""
        try:
            filepath = Path(filename)
            if not filepath.exists():
                print(f"Session file '{filename}' not found")
                return False
            
            with open(filepath, 'r') as f:
                session_data = yaml.safe_load(f)
            
            # Stop existing panes
            for pane in self.panes.values():
                pane.stop()
            self.panes.clear()
            
            # Load session configuration
            self.session_name = session_data.get('name', 'default')
            
            # Create panes from session data
            for pane_data in session_data.get('panes', []):
                name = pane_data.get('name', f"pane_{len(self.panes)}")
                command = pane_data.get('command', self.default_shell)
                cwd = pane_data.get('cwd')
                shell = pane_data.get('shell', self.default_shell)
                
                self.create_pane(name, command, cwd, shell)
            
            if self.logger:
                self.logger.log_session_loaded(filename, len(self.panes))
            
            return True
            
        except Exception as e:
            print(f"Error loading session: {e}")
            return False
    
    def get_session_info(self) -> str:
        """Get information about the current session."""
        active_count = sum(1 for pane in self.panes.values() if pane.is_alive())
        total_count = len(self.panes)
        
        return f"{active_count}/{total_count} terminals ({self.session_name})"
    
    def _on_pane_output(self, pane_name: str, output: str):
        """Callback for pane output."""
        if self.logger:
            self.logger.log_pane_output(pane_name, output)
    
    def cleanup(self):
        """Cleanup all panes."""
        for pane in self.panes.values():
            pane.stop()
        self.panes.clear()
        self.active_pane = None 