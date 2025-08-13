"""
Smart logging system for PyTerminus.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class Logger:
    """Smart logger for PyTerminus sessions and outputs."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Session log file
        self.session_log = self.log_dir / "session.log"
        self.output_log = self.log_dir / "output.log"
        self.command_log = self.log_dir / "commands.log"
        
        # Initialize log files
        self._init_log_files()
    
    def _init_log_files(self):
        """Initialize log files with headers."""
        if not self.session_log.exists():
            with open(self.session_log, 'w') as f:
                f.write("# PyTerminus Session Log\n")
                f.write(f"# Started: {datetime.now().isoformat()}\n\n")
        
        if not self.output_log.exists():
            with open(self.output_log, 'w') as f:
                f.write("# PyTerminus Output Log\n")
                f.write(f"# Started: {datetime.now().isoformat()}\n\n")
        
        if not self.command_log.exists():
            with open(self.command_log, 'w') as f:
                f.write("# PyTerminus Command Log\n")
                f.write(f"# Started: {datetime.now().isoformat()}\n\n")
    
    def _write_log_entry(self, log_file: Path, entry_type: str, data: Dict[str, Any]):
        """Write a log entry to the specified file."""
        timestamp = datetime.now().isoformat()
        entry = {
            'timestamp': timestamp,
            'type': entry_type,
            'data': data
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def log_session_started(self, session_name: str, panes: int = 0):
        """Log session start."""
        self._write_log_entry(
            self.session_log,
            'session_started',
            {
                'session_name': session_name,
                'panes_count': panes
            }
        )
    
    def log_session_ended(self, session_name: str, duration: float):
        """Log session end."""
        self._write_log_entry(
            self.session_log,
            'session_ended',
            {
                'session_name': session_name,
                'duration_seconds': duration
            }
        )
    
    def log_session_saved(self, filename: str, panes_count: int):
        """Log session save."""
        self._write_log_entry(
            self.session_log,
            'session_saved',
            {
                'filename': filename,
                'panes_count': panes_count
            }
        )
    
    def log_session_loaded(self, filename: str, panes_count: int):
        """Log session load."""
        self._write_log_entry(
            self.session_log,
            'session_loaded',
            {
                'filename': filename,
                'panes_count': panes_count
            }
        )
    
    def log_pane_created(self, pane_name: str, command: str, cwd: Optional[str] = None):
        """Log pane creation."""
        self._write_log_entry(
            self.session_log,
            'pane_created',
            {
                'pane_name': pane_name,
                'command': command,
                'cwd': cwd
            }
        )
    
    def log_pane_removed(self, pane_name: str):
        """Log pane removal."""
        self._write_log_entry(
            self.session_log,
            'pane_removed',
            {
                'pane_name': pane_name
            }
        )
    
    def log_pane_output(self, pane_name: str, output: str):
        """Log pane output."""
        # Truncate very long outputs
        if len(output) > 1000:
            output = output[:1000] + "... [truncated]"
        
        self._write_log_entry(
            self.output_log,
            'pane_output',
            {
                'pane_name': pane_name,
                'output': output,
                'output_length': len(output)
            }
        )
    
    def log_command_executed(self, pane_name: str, command: str, success: bool = True):
        """Log command execution."""
        self._write_log_entry(
            self.command_log,
            'command_executed',
            {
                'pane_name': pane_name,
                'command': command,
                'success': success
            }
        )
    
    def log_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """Log errors."""
        data = {
            'error_type': error_type,
            'error_message': error_message
        }
        if context:
            data['context'] = context
        
        self._write_log_entry(
            self.session_log,
            'error',
            data
        )
    
    def get_recent_logs(self, log_type: str = 'session', limit: int = 50) -> list:
        """Get recent log entries."""
        log_file = {
            'session': self.session_log,
            'output': self.output_log,
            'command': self.command_log
        }.get(log_type, self.session_log)
        
        if not log_file.exists():
            return []
        
        logs = []
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    try:
                        logs.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        
        return logs[-limit:]
    
    def search_logs(self, query: str, log_type: str = 'session') -> list:
        """Search logs for specific content."""
        logs = self.get_recent_logs(log_type, limit=1000)
        results = []
        
        query_lower = query.lower()
        for log in logs:
            log_str = json.dumps(log).lower()
            if query_lower in log_str:
                results.append(log)
        
        return results
    
    def get_session_summary(self, session_name: str = None) -> Dict[str, Any]:
        """Get a summary of session activity."""
        logs = self.get_recent_logs('session', limit=1000)
        
        summary = {
            'total_sessions': 0,
            'total_panes_created': 0,
            'total_commands': 0,
            'errors': 0,
            'last_activity': None
        }
        
        for log in logs:
            if log.get('type') == 'session_started':
                summary['total_sessions'] += 1
            elif log.get('type') == 'pane_created':
                summary['total_panes_created'] += 1
            elif log.get('type') == 'error':
                summary['errors'] += 1
            
            summary['last_activity'] = log.get('timestamp')
        
        # Get command count from command log
        command_logs = self.get_recent_logs('command', limit=1000)
        summary['total_commands'] = len(command_logs)
        
        return summary 