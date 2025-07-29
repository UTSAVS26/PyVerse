import os
from datetime import datetime
import getpass

SYSTEM_PATHS = {
    'downloads': os.path.join(os.path.expanduser('~'), 'Downloads'),
    'desktop': os.path.join(os.path.expanduser('~'), 'Desktop'),
    'documents': os.path.join(os.path.expanduser('~'), 'Documents'),
    # Add more as needed
}

def get_system_path(name: str) -> str:
    """Resolve a friendly name to a system path."""
    return SYSTEM_PATHS.get(name.lower(), name)

def parse_size(size_str: str) -> int:
    """Parse a human-friendly file size string (e.g., '10MB', '2GB', '1TB') to bytes."""
    size_str = size_str.strip().upper()
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024**2,
        'GB': 1024**3,
        'TB': 1024**4
    }
    for unit in units:
        if size_str.endswith(unit):
            try:
                return int(float(size_str[:-len(unit)]) * units[unit])
            except ValueError:
                raise ValueError(f"Invalid size format: {size_str}")
    try:
        return int(size_str)
    except ValueError:
        raise ValueError(f"Invalid size format: {size_str}")

def log_command(command: str, approved: bool, logfile: str = 'smartcli.log'):
    """Log the command with timestamp, approval status, user, and cwd."""
    with open(logfile, 'a') as f:
        f.write(f"{datetime.now().isoformat()} | {getpass.getuser()} | {os.getcwd()} | {'APPROVED' if approved else 'BLOCKED'} | {command}\n") 