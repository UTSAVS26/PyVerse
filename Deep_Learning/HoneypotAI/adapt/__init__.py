"""
HoneypotAI - Advanced Adaptive Cybersecurity Intelligence Platform
Adaptive Response Module: Dynamic threat response and mitigation
"""

from .adaptive_response import AdaptiveResponse
from .firewall_manager import FirewallManager
from .response_strategies import ResponseStrategy, ResponseStrategyManager

__version__ = "1.0.0"
__author__ = "HoneypotAI Team"

__all__ = [
    "AdaptiveResponse",
    "FirewallManager", 
    "ResponseStrategy",
    "ResponseStrategyManager"
]
