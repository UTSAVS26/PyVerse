"""
ClipCrypt: Encrypted Clipboard Manager

A secure, searchable, and local-only clipboard history manager
that encrypts all clipboard entries using AES-GCM encryption.
"""

__version__ = "1.0.0"
__author__ = "Shivansh Katiyar"
__description__ = "Secure encrypted clipboard manager"

from .core import ClipCrypt
from .encryption import EncryptionManager
from .storage import StorageManager
from .clipboard import ClipboardMonitor

__all__ = [
    "ClipCrypt",
    "EncryptionManager", 
    "StorageManager",
    "ClipboardMonitor"
] 