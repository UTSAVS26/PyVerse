"""
Firewall Manager for HoneypotAI
Mock firewall management for adaptive response system
"""

import threading
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger(__name__)

class FirewallManager:
    """Mock firewall manager for demonstration purposes"""
    
    def __init__(self):
        self.logger = structlog.get_logger("adapt.firewall_manager")
        
        # Mock firewall rules
        self.blocked_ips = {}  # IP -> rule_info
        self.throttled_ips = {}  # IP -> throttle_info
        self.rate_limits = {}  # IP -> rate_limit_info
        
        # Statistics
        self.stats = {
            "rules_created": 0,
            "rules_removed": 0,
            "blocks_active": 0,
            "throttles_active": 0
        }
    
    def block_ip(self, ip: str, duration: int = 3600, reason: str = "threat_detected") -> bool:
        """Block an IP address"""
        try:
            block_until = datetime.now() + timedelta(seconds=duration)
            
            self.blocked_ips[ip] = {
                "block_until": block_until,
                "reason": reason,
                "created": datetime.now(),
                "duration": duration
            }
            
            self.stats["rules_created"] += 1
            self.stats["blocks_active"] = len(self.blocked_ips)
            
            self.logger.info(f"Blocked IP {ip} until {block_until} (reason: {reason})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error blocking IP {ip}: {e}")
            return False
    
    def unblock_ip(self, ip: str) -> bool:
        """Unblock an IP address"""
        try:
            if ip in self.blocked_ips:
                del self.blocked_ips[ip]
                self.stats["rules_removed"] += 1
                self.stats["blocks_active"] = len(self.blocked_ips)
                
                self.logger.info(f"Unblocked IP {ip}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error unblocking IP {ip}: {e}")
            return False
    
    def throttle_ip(self, ip: str, rate_limit: int = 10, duration: int = 300) -> bool:
        """Throttle an IP address"""
        try:
            throttle_until = datetime.now() + timedelta(seconds=duration)
            
            self.throttled_ips[ip] = {
                "throttle_until": throttle_until,
                "rate_limit": rate_limit,
                "created": datetime.now(),
                "duration": duration
            }
            
            self.stats["rules_created"] += 1
            self.stats["throttles_active"] = len(self.throttled_ips)
            
            self.logger.info(f"Throttled IP {ip} (rate: {rate_limit}/min) until {throttle_until}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error throttling IP {ip}: {e}")
            return False
    
    def unthrottle_ip(self, ip: str) -> bool:
        """Remove throttling for an IP address"""
        try:
            if ip in self.throttled_ips:
                del self.throttled_ips[ip]
                self.stats["rules_removed"] += 1
                self.stats["throttles_active"] = len(self.throttled_ips)
                
                self.logger.info(f"Unthrottled IP {ip}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error unthrottling IP {ip}: {e}")
            return False
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if an IP is currently blocked"""
        if ip not in self.blocked_ips:
            return False
        
        # Check if block has expired
        block_info = self.blocked_ips[ip]
        if time.time() > block_info["block_until"]:
            # Auto-remove expired block
            del self.blocked_ips[ip]
            self.stats["blocks_active"] = len(self.blocked_ips)
            return False
        
        return True
    
    def is_ip_throttled(self, ip: str) -> bool:
        """Check if an IP is currently throttled"""
        if ip not in self.throttled_ips:
            return False
        
        # Check if throttle has expired
        throttle_info = self.throttled_ips[ip]
        if time.time() > throttle_info["throttle_until"]:
            # Auto-remove expired throttle
            del self.throttled_ips[ip]
            self.stats["throttles_active"] = len(self.throttled_ips)
            return False
        
        return True
    
    def get_blocked_ips(self) -> Dict[str, Any]:
        """Get all currently blocked IPs"""
        # Clean up expired blocks
        expired_blocks = [
            ip for ip, info in self.blocked_ips.items()
            if time.time() > info["block_until"]
        ]
        for ip in expired_blocks:
            del self.blocked_ips[ip]
        
        self.stats["blocks_active"] = len(self.blocked_ips)
        return self.blocked_ips
    
    def get_throttled_ips(self) -> Dict[str, Any]:
        """Get all currently throttled IPs"""
        # Clean up expired throttles
        expired_throttles = [
            ip for ip, info in self.throttled_ips.items()
            if time.time() > info["throttle_until"]
        ]
        for ip in expired_throttles:
            del self.throttled_ips[ip]
        
        self.stats["throttles_active"] = len(self.throttled_ips)
        return self.throttled_ips
    
    def get_stats(self) -> Dict[str, Any]:
        """Get firewall statistics"""
        return self.stats.copy()
    
    def clear_all_rules(self):
        """Clear all firewall rules"""
        self.blocked_ips.clear()
        self.throttled_ips.clear()
        self.stats["blocks_active"] = 0
        self.stats["throttles_active"] = 0
        
        self.logger.info("Cleared all firewall rules")
