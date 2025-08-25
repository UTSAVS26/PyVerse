"""
Adaptive Response System for HoneypotAI
Handles dynamic threat response and mitigation strategies
"""

import threading
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger(__name__)

class AdaptiveResponse:
    """Adaptive response system for threat mitigation"""
    
    def __init__(self):
        self.logger = structlog.get_logger("adapt.adaptive_response")
        
        # Configuration
        self.blocking_strategy = "dynamic"  # static, dynamic, adaptive
        self.throttling_enabled = True
        self.decoy_responses = True
        self.auto_blocking = True
        
        # Response thresholds
        self.block_threshold = 3  # Number of threats before blocking
        self.block_duration = 3600  # Block duration in seconds
        self.throttle_threshold = 2  # Number of threats before throttling
        
        # State tracking
        self.blocked_ips = {}  # IP -> block_until_timestamp
        self.throttled_ips = {}  # IP -> throttle_until_timestamp
        self.threat_history = {}  # IP -> list of threats
        self.response_stats = {
            "total_threats": 0,
            "blocks_issued": 0,
            "throttles_issued": 0,
            "decoys_sent": 0,
            "start_time": None
        }
        
        # Threading
        self.running = False
        self.response_thread = None
        self.cleanup_thread = None
        
        # Response strategies
        self.response_strategies = {
            "brute_force": self._handle_brute_force,
            "sql_injection": self._handle_sql_injection,
            "xss": self._handle_xss,
            "path_traversal": self._handle_path_traversal,
            "command_injection": self._handle_command_injection,
            "scanning": self._handle_scanning,
            "dos": self._handle_dos,
            "anomaly": self._handle_anomaly,
            "classified_attack": self._handle_classified_attack
        }
    
    def set_blocking_strategy(self, strategy: str):
        """Set the blocking strategy"""
        if strategy in ["static", "dynamic", "adaptive"]:
            self.blocking_strategy = strategy
            self.logger.info(f"Set blocking strategy to {strategy}")
        else:
            self.logger.error(f"Invalid blocking strategy: {strategy}")
    
    def set_throttling_enabled(self, enabled: bool):
        """Enable or disable throttling"""
        self.throttling_enabled = enabled
        self.logger.info(f"Throttling {'enabled' if enabled else 'disabled'}")
    
    def set_decoy_responses(self, enabled: bool):
        """Enable or disable decoy responses"""
        self.decoy_responses = enabled
        self.logger.info(f"Decoy responses {'enabled' if enabled else 'disabled'}")
    
    def set_auto_blocking(self, enabled: bool):
        """Enable or disable automatic blocking"""
        self.auto_blocking = enabled
        self.logger.info(f"Auto blocking {'enabled' if enabled else 'disabled'}")
    
    def start_adaptive_defense(self) -> bool:
        """Start the adaptive defense system"""
        try:
            self.running = True
            self.response_stats["start_time"] = datetime.now()
            
            # Start response thread
            self.response_thread = threading.Thread(target=self._response_loop, daemon=True)
            self.response_thread.start()
            
            # Start cleanup thread
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
            
            self.logger.info("Started adaptive defense system")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting adaptive defense: {e}")
            return False
    
    def stop_adaptive_defense(self):
        """Stop the adaptive defense system"""
        self.running = False
        
        if self.response_thread:
            self.response_thread.join(timeout=5)
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        
        self.logger.info("Stopped adaptive defense system")
    
    def handle_threat(self, threat: Dict[str, Any]):
        """Handle a detected threat"""
        try:
            source_ip = threat.get("source_ip", "")
            threat_type = threat.get("threat_type", "unknown")
            confidence = threat.get("confidence", 0.0)
            
            # Update threat history
            if source_ip not in self.threat_history:
                self.threat_history[source_ip] = []
            
            self.threat_history[source_ip].append({
                "timestamp": datetime.now(),
                "threat_type": threat_type,
                "confidence": confidence,
                "details": threat.get("details", {})
            })
            
            # Update statistics
            self.response_stats["total_threats"] += 1
            
            # Check if IP should be blocked
            if self.auto_blocking and self._should_block_ip(source_ip):
                self._block_ip(source_ip, threat_type)
            
            # Check if IP should be throttled
            elif self.throttling_enabled and self._should_throttle_ip(source_ip):
                self._throttle_ip(source_ip, threat_type)
            
            # Apply specific response strategy
            if threat_type in self.response_strategies:
                self.response_strategies[threat_type](threat)
            else:
                self._handle_generic_threat(threat)
            
            self.logger.info(f"Handled threat: {threat_type} from {source_ip} (confidence: {confidence})")
            
        except Exception as e:
            self.logger.error(f"Error handling threat: {e}")
    
    def _should_block_ip(self, source_ip: str) -> bool:
        """Determine if an IP should be blocked"""
        if source_ip in self.blocked_ips:
            return False  # Already blocked
        
        threat_count = len(self.threat_history.get(source_ip, []))
        return threat_count >= self.block_threshold
    
    def _should_throttle_ip(self, source_ip: str) -> bool:
        """Determine if an IP should be throttled"""
        if source_ip in self.throttled_ips or source_ip in self.blocked_ips:
            return False  # Already throttled or blocked
        
        threat_count = len(self.threat_history.get(source_ip, []))
        return threat_count >= self.throttle_threshold
    
    def _block_ip(self, source_ip: str, reason: str) -> bool:
        """Block an IP address"""
        try:
            block_until = datetime.now() + timedelta(seconds=self.block_duration)
            self.blocked_ips[source_ip] = {
                "block_until": block_until,
                "reason": reason,
                "timestamp": datetime.now()
            }
            
            self.response_stats["blocks_issued"] += 1
            self.logger.warning(f"Blocked IP {source_ip} until {block_until} (reason: {reason})")
            
            # In a real implementation, you would update firewall rules here
            # self.firewall_manager.block_ip(source_ip, self.block_duration)
            return True
            
        except Exception as e:
            self.logger.error(f"Error blocking IP {source_ip}: {e}")
            return False
    
    def _throttle_ip(self, source_ip: str, reason: str) -> bool:
        """Throttle an IP address"""
        try:
            throttle_until = datetime.now() + timedelta(seconds=300)  # 5 minutes
            self.throttled_ips[source_ip] = {
                "throttle_until": throttle_until,
                "reason": reason,
                "timestamp": datetime.now()
            }
            
            self.response_stats["throttles_issued"] += 1
            self.logger.warning(f"Throttled IP {source_ip} until {throttle_until} (reason: {reason})")
            
            # In a real implementation, you would update rate limiting here
            return True
            
        except Exception as e:
            self.logger.error(f"Error throttling IP {source_ip}: {e}")
            return False
    
    def _handle_brute_force(self, threat: Dict[str, Any]):
        """Handle brute force attacks"""
        source_ip = threat.get("source_ip", "")
        
        # Immediate blocking for brute force
        if self.auto_blocking:
            self._block_ip(source_ip, "brute_force")
        
        # Send decoy response
        if self.decoy_responses:
            self._send_decoy_response(source_ip, "brute_force")
    
    def _handle_sql_injection(self, threat: Dict[str, Any]):
        """Handle SQL injection attacks"""
        source_ip = threat.get("source_ip", "")
        
        # Block after multiple attempts
        if self.auto_blocking and len(self.threat_history.get(source_ip, [])) >= 2:
            self._block_ip(source_ip, "sql_injection")
        
        # Send decoy response
        if self.decoy_responses:
            self._send_decoy_response(source_ip, "sql_injection")
    
    def _handle_xss(self, threat: Dict[str, Any]):
        """Handle XSS attacks"""
        source_ip = threat.get("source_ip", "")
        
        # Throttle XSS attempts
        if self.throttling_enabled:
            self._throttle_ip(source_ip, "xss")
        
        # Send decoy response
        if self.decoy_responses:
            self._send_decoy_response(source_ip, "xss")
    
    def _handle_path_traversal(self, threat: Dict[str, Any]):
        """Handle path traversal attacks"""
        source_ip = threat.get("source_ip", "")
        
        # Immediate blocking for path traversal
        if self.auto_blocking:
            self._block_ip(source_ip, "path_traversal")
        
        # Send decoy response
        if self.decoy_responses:
            self._send_decoy_response(source_ip, "path_traversal")
    
    def _handle_command_injection(self, threat: Dict[str, Any]):
        """Handle command injection attacks"""
        source_ip = threat.get("source_ip", "")
        
        # Immediate blocking for command injection
        if self.auto_blocking:
            self._block_ip(source_ip, "command_injection")
        
        # Send decoy response
        if self.decoy_responses:
            self._send_decoy_response(source_ip, "command_injection")
    
    def _handle_scanning(self, threat: Dict[str, Any]):
        """Handle scanning activities"""
        source_ip = threat.get("source_ip", "")
        
        # Throttle scanning
        if self.throttling_enabled:
            self._throttle_ip(source_ip, "scanning")
        
        # Send decoy response
        if self.decoy_responses:
            self._send_decoy_response(source_ip, "scanning")
    
    def _handle_dos(self, threat: Dict[str, Any]):
        """Handle DoS attacks"""
        source_ip = threat.get("source_ip", "")
        
        # Immediate blocking for DoS
        if self.auto_blocking:
            self._block_ip(source_ip, "dos")
    
    def _handle_anomaly(self, threat: Dict[str, Any]):
        """Handle anomaly detections"""
        source_ip = threat.get("source_ip", "")
        confidence = threat.get("confidence", 0.0)
        
        # Block high-confidence anomalies
        if self.auto_blocking and confidence > 0.8:
            self._block_ip(source_ip, "anomaly")
        elif self.throttling_enabled and confidence > 0.6:
            self._throttle_ip(source_ip, "anomaly")
    
    def _handle_classified_attack(self, threat: Dict[str, Any]):
        """Handle classified attacks"""
        source_ip = threat.get("source_ip", "")
        attack_type = threat.get("details", {}).get("attack_type", "unknown")
        
        # Apply specific handling based on attack type
        if attack_type in self.response_strategies:
            self.response_strategies[attack_type](threat)
        else:
            self._handle_generic_threat(threat)
    
    def _handle_generic_threat(self, threat: Dict[str, Any]):
        """Handle generic threats"""
        source_ip = threat.get("source_ip", "")
        
        # Default response: throttle
        if self.throttling_enabled:
            self._throttle_ip(source_ip, "generic_threat")
    
    def _send_decoy_response(self, source_ip: str, threat_type: str):
        """Send decoy response to attacker"""
        try:
            # In a real implementation, you would send misleading responses
            # For now, we just log the action
            self.response_stats["decoys_sent"] += 1
            self.logger.info(f"Sent decoy response to {source_ip} for {threat_type}")
            
        except Exception as e:
            self.logger.error(f"Error sending decoy response: {e}")
    
    def _response_loop(self):
        """Main response processing loop"""
        while self.running:
            try:
                # Process any pending responses
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in response loop: {e}")
                time.sleep(5)
    
    def _cleanup_loop(self):
        """Cleanup expired blocks and throttles"""
        while self.running:
            try:
                current_time = datetime.now()
                
                # Cleanup expired blocks
                expired_blocks = [
                    ip for ip, data in self.blocked_ips.items()
                    if data["block_until"] < current_time
                ]
                for ip in expired_blocks:
                    del self.blocked_ips[ip]
                    self.logger.info(f"Unblocked IP {ip}")
                
                # Cleanup expired throttles
                expired_throttles = [
                    ip for ip, data in self.throttled_ips.items()
                    if data["throttle_until"] < current_time
                ]
                for ip in expired_throttles:
                    del self.throttled_ips[ip]
                    self.logger.info(f"Unthrottled IP {ip}")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                time.sleep(60)
    
    def get_status(self) -> Dict[str, Any]:
        """Get adaptive response status"""
        stats = self.response_stats.copy()
        
        # Calculate uptime
        if stats["start_time"]:
            stats["uptime"] = (datetime.now() - stats["start_time"]).total_seconds()
        
        # Add current state
        stats["blocked_ips_count"] = len(self.blocked_ips)
        stats["throttled_ips_count"] = len(self.throttled_ips)
        stats["threat_history_count"] = len(self.threat_history)
        
        # Add configuration
        stats["configuration"] = {
            "blocking_strategy": self.blocking_strategy,
            "throttling_enabled": self.throttling_enabled,
            "decoy_responses": self.decoy_responses,
            "auto_blocking": self.auto_blocking,
            "block_threshold": self.block_threshold,
            "throttle_threshold": self.throttle_threshold
        }
        
        return stats
    
    def get_blocked_ips(self) -> Dict[str, Any]:
        """Get list of currently blocked IPs"""
        return self.blocked_ips
    
    def get_throttled_ips(self) -> Dict[str, Any]:
        """Get list of currently throttled IPs"""
        return self.throttled_ips
    
    def unblock_ip(self, source_ip: str) -> bool:
        """Manually unblock an IP"""
        if source_ip in self.blocked_ips:
            del self.blocked_ips[source_ip]
            self.logger.info(f"Manually unblocked IP {source_ip}")
            return True
        return False
    
    def unthrottle_ip(self, source_ip: str) -> bool:
        """Manually unthrottle an IP"""
        if source_ip in self.throttled_ips:
            del self.throttled_ips[source_ip]
            self.logger.info(f"Manually unthrottled IP {source_ip}")
            return True
        return False
    
    def clear_threat_history(self, source_ip: Optional[str] = None):
        """Clear threat history for an IP or all IPs"""
        if source_ip:
            if source_ip in self.threat_history:
                del self.threat_history[source_ip]
                self.logger.info(f"Cleared threat history for {source_ip}")
        else:
            self.threat_history.clear()
            self.logger.info("Cleared all threat history")
