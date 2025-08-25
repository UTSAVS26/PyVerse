"""
HoneypotAI - Advanced Adaptive Cybersecurity Intelligence Platform
Honeypot Module: Core honeypot services and management
"""

from .honeypot_manager import HoneypotManager
from .ssh_server import SSHServer
from .http_server import HTTPServer
from .ftp_server import FTPServer
from .base_server import BaseServer

__version__ = "1.0.0"
__author__ = "HoneypotAI Team"

__all__ = [
    "HoneypotManager",
    "SSHServer", 
    "HTTPServer",
    "FTPServer",
    "BaseServer"
]
