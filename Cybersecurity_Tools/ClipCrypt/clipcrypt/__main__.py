"""
Main entry point for ClipCrypt.

This module allows running ClipCrypt as a module:
python -m clipcrypt
"""

from .cli import cli

if __name__ == '__main__':
    cli() 