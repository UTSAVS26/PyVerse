"""
User interface components for the Accent Strength Estimator.
"""

# Try to import all interfaces, but handle missing dependencies gracefully
try:
    from .cli_interface import CLIInterface
    from .gui_interface import GUIInterface
    from .web_interface import WebInterface
    __all__ = ['CLIInterface', 'GUIInterface', 'WebInterface']
except ImportError:
    # If external dependencies are not available, only export what we can
    __all__ = []
