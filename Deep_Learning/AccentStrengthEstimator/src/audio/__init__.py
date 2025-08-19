"""
Audio processing components for the Accent Strength Estimator.
"""

# Import only the modules that don't require external dependencies
from .reference_generator import ReferenceGenerator

# Try to import modules that require external dependencies
try:
    from .processor import AudioProcessor
    from .recorder import AudioRecorder
    __all__ = ['AudioRecorder', 'AudioProcessor', 'ReferenceGenerator']
except ImportError:
    # If external dependencies are not available, only export what we can
    __all__ = ['ReferenceGenerator']
