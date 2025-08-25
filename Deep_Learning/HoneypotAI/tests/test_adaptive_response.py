"""
Tests for HoneypotAI Adaptive Response Module
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapt import AdaptiveResponse, FirewallManager, ResponseStrategyManager
from adapt.response_strategies import (
    ImmediateBlockStrategy, GradualEscalationStrategy, 
    DecoyResponseStrategy, AdaptiveStrategy, PassiveMonitoringStrategy
)

class TestAdaptiveResponse:
    """Test adaptive response functionality"""
    
    def test_adaptive_response_initialization(self):
        """Test adaptive response initialization"""
        response = AdaptiveResponse()
        
        assert response.blocking_strategy == "dynamic"
        assert response.throttling_enabled is True
        assert response.decoy_responses is True
        assert response.auto_blocking is True
        assert response.block_threshold == 3
        assert response.block_duration == 3600
        assert response.throttle_threshold == 2
        assert len(response.blocked_ips) == 0
        assert len(response.throttled_ips) == 0
        assert len(response.threat_history) == 0
        assert not response.running
    
    def test_set_blocking_strategy(self):
        """Test setting blocking strategy"""
        response = AdaptiveResponse()
        
        response.set_blocking_strategy("static")
        assert response.blocking_strategy == "static"
        
        response.set_blocking_strategy("adaptive")
        assert response.blocking_strategy == "adaptive"
        
        # Test invalid strategy
        response.set_blocking_strategy("invalid")
        assert response.blocking_strategy == "adaptive"  # Should not change
    
    def test_set_throttling_enabled(self):
        """Test setting throttling enabled"""
        response = AdaptiveResponse()
        
        response.set_throttling_enabled(False)
        assert response.throttling_enabled is False
        
        response.set_throttling_enabled(True)
        assert response.throttling_enabled is True
    
    def test_set_decoy_responses(self):
        """Test setting decoy responses"""
        response = AdaptiveResponse()
        
        response.set_decoy_responses(False)
        assert response.decoy_responses is False
        
        response.set_decoy_responses(True)
        assert response.decoy_responses is True
    
    def test_set_auto_blocking(self):
        """Test setting auto blocking"""
        response = AdaptiveResponse()
        
        response.set_auto_blocking(False)
        assert response.auto_blocking is False
        
        response.set_auto_blocking(True)
        assert response.auto_blocking is True
    
    def test_should_block_ip(self):
        """Test IP blocking logic"""
        response = AdaptiveResponse()
        
        # Initially should not block
        assert not response._should_block_ip("192.168.1.1")
        
        # Add threats to history
        response.threat_history["192.168.1.1"] = [Mock(), Mock(), Mock(), Mock()]
        
        # Should block after 3 threats
        assert response._should_block_ip("192.168.1.1")
        
        # Already blocked IP should not be blocked again
        response.blocked_ips["192.168.1.1"] = {"block_until": time.time() + 3600}
        assert not response._should_block_ip("192.168.1.1")
    
    def test_should_throttle_ip(self):
        """Test IP throttling logic"""
        response = AdaptiveResponse()
        
        # Initially should not throttle
        assert not response._should_throttle_ip("192.168.1.1")
        
        # Add threats to history
        response.threat_history["192.168.1.1"] = [Mock(), Mock()]
        
        # Should throttle after 2 threats
        assert response._should_throttle_ip("192.168.1.1")
        
        # Already throttled IP should not be throttled again
        response.throttled_ips["192.168.1.1"] = {"throttle_until": time.time() + 300}
        assert not response._should_throttle_ip("192.168.1.1")
        
        # Blocked IP should not be throttled
        response.blocked_ips["192.168.1.1"] = {"block_until": time.time() + 3600}
        assert not response._should_throttle_ip("192.168.1.1")
    
    def test_block_ip(self):
        """Test IP blocking"""
        response = AdaptiveResponse()
        
        success = response._block_ip("192.168.1.1", "test_reason")
        assert success
        assert "192.168.1.1" in response.blocked_ips
        assert response.response_stats["blocks_issued"] == 1
    
    def test_throttle_ip(self):
        """Test IP throttling"""
        response = AdaptiveResponse()
        
        success = response._throttle_ip("192.168.1.1", "test_reason")
        assert success
        assert "192.168.1.1" in response.throttled_ips
        assert response.response_stats["throttles_issued"] == 1
    
    def test_handle_threat(self):
        """Test threat handling"""
        response = AdaptiveResponse()
        
        threat = {
            "source_ip": "192.168.1.1",
            "threat_type": "brute_force",
            "confidence": 0.9
        }
        
        response.handle_threat(threat)
        
        assert "192.168.1.1" in response.threat_history
        assert response.response_stats["total_threats"] == 1
        assert len(response.threat_history["192.168.1.1"]) == 1
    
    def test_handle_brute_force(self):
        """Test brute force handling"""
        response = AdaptiveResponse()
        
        threat = {
            "source_ip": "192.168.1.1",
            "threat_type": "brute_force",
            "confidence": 0.9
        }
        
        response._handle_brute_force(threat)
        
        # Should be blocked immediately
        assert "192.168.1.1" in response.blocked_ips
        assert response.response_stats["decoys_sent"] == 1
    
    def test_handle_sql_injection(self):
        """Test SQL injection handling"""
        response = AdaptiveResponse()
        
        threat = {
            "source_ip": "192.168.1.1",
            "threat_type": "sql_injection",
            "confidence": 0.9
        }
        
        # Add multiple threats to trigger blocking
        response.threat_history["192.168.1.1"] = [Mock(), Mock()]
        
        response._handle_sql_injection(threat)
        
        # Should be blocked after multiple attempts
        assert "192.168.1.1" in response.blocked_ips
        assert response.response_stats["decoys_sent"] == 1
    
    def test_handle_xss(self):
        """Test XSS handling"""
        response = AdaptiveResponse()
        
        threat = {
            "source_ip": "192.168.1.1",
            "threat_type": "xss",
            "confidence": 0.9
        }
        
        response._handle_xss(threat)
        
        # Should be throttled
        assert "192.168.1.1" in response.throttled_ips
        assert response.response_stats["decoys_sent"] == 1
    
    def test_handle_path_traversal(self):
        """Test path traversal handling"""
        response = AdaptiveResponse()
        
        threat = {
            "source_ip": "192.168.1.1",
            "threat_type": "path_traversal",
            "confidence": 0.9
        }
        
        response._handle_path_traversal(threat)
        
        # Should be blocked immediately
        assert "192.168.1.1" in response.blocked_ips
        assert response.response_stats["decoys_sent"] == 1
    
    def test_handle_command_injection(self):
        """Test command injection handling"""
        response = AdaptiveResponse()
        
        threat = {
            "source_ip": "192.168.1.1",
            "threat_type": "command_injection",
            "confidence": 0.9
        }
        
        response._handle_command_injection(threat)
        
        # Should be blocked immediately
        assert "192.168.1.1" in response.blocked_ips
        assert response.response_stats["decoys_sent"] == 1
    
    def test_handle_scanning(self):
        """Test scanning handling"""
        response = AdaptiveResponse()
        
        threat = {
            "source_ip": "192.168.1.1",
            "threat_type": "scanning",
            "confidence": 0.9
        }
        
        response._handle_scanning(threat)
        
        # Should be throttled
        assert "192.168.1.1" in response.throttled_ips
        assert response.response_stats["decoys_sent"] == 1
    
    def test_handle_dos(self):
        """Test DoS handling"""
        response = AdaptiveResponse()
        
        threat = {
            "source_ip": "192.168.1.1",
            "threat_type": "dos",
            "confidence": 0.9
        }
        
        response._handle_dos(threat)
        
        # Should be blocked immediately
        assert "192.168.1.1" in response.blocked_ips
    
    def test_handle_anomaly(self):
        """Test anomaly handling"""
        response = AdaptiveResponse()
        
        # High confidence anomaly
        threat = {
            "source_ip": "192.168.1.1",
            "threat_type": "anomaly",
            "confidence": 0.9
        }
        
        response._handle_anomaly(threat)
        
        # Should be blocked
        assert "192.168.1.1" in response.blocked_ips
        
        # Medium confidence anomaly
        threat2 = {
            "source_ip": "192.168.1.2",
            "threat_type": "anomaly",
            "confidence": 0.7
        }
        
        response._handle_anomaly(threat2)
        
        # Should be throttled
        assert "192.168.1.2" in response.throttled_ips
    
    def test_get_status(self):
        """Test getting status"""
        response = AdaptiveResponse()
        
        # Add some mock data
        response.blocked_ips["192.168.1.1"] = {"block_until": time.time() + 3600}
        response.throttled_ips["192.168.1.2"] = {"throttle_until": time.time() + 300}
        response.threat_history["192.168.1.1"] = [Mock()]
        
        status = response.get_status()
        
        assert "total_threats" in status
        assert "blocks_issued" in status
        assert "throttles_issued" in status
        assert "decoys_sent" in status
        assert "blocked_ips_count" in status
        assert "throttled_ips_count" in status
        assert "threat_history_count" in status
        assert "configuration" in status
    
    def test_get_blocked_ips(self):
        """Test getting blocked IPs"""
        response = AdaptiveResponse()
        
        response.blocked_ips["192.168.1.1"] = {"block_until": time.time() + 3600}
        response.blocked_ips["192.168.1.2"] = {"block_until": time.time() + 3600}
        
        blocked = response.get_blocked_ips()
        assert len(blocked) == 2
        assert "192.168.1.1" in blocked
        assert "192.168.1.2" in blocked
    
    def test_get_throttled_ips(self):
        """Test getting throttled IPs"""
        response = AdaptiveResponse()
        
        response.throttled_ips["192.168.1.1"] = {"throttle_until": time.time() + 300}
        response.throttled_ips["192.168.1.2"] = {"throttle_until": time.time() + 300}
        
        throttled = response.get_throttled_ips()
        assert len(throttled) == 2
        assert "192.168.1.1" in throttled
        assert "192.168.1.2" in throttled
    
    def test_unblock_ip(self):
        """Test unblocking IP"""
        response = AdaptiveResponse()
        
        response.blocked_ips["192.168.1.1"] = {"block_until": time.time() + 3600}
        
        success = response.unblock_ip("192.168.1.1")
        assert success
        assert "192.168.1.1" not in response.blocked_ips
        
        # Test unblocking non-blocked IP
        success = response.unblock_ip("192.168.1.2")
        assert not success
    
    def test_unthrottle_ip(self):
        """Test unthrottling IP"""
        response = AdaptiveResponse()
        
        response.throttled_ips["192.168.1.1"] = {"throttle_until": time.time() + 300}
        
        success = response.unthrottle_ip("192.168.1.1")
        assert success
        assert "192.168.1.1" not in response.throttled_ips
        
        # Test unthrottling non-throttled IP
        success = response.unthrottle_ip("192.168.1.2")
        assert not success
    
    def test_clear_threat_history(self):
        """Test clearing threat history"""
        response = AdaptiveResponse()
        
        response.threat_history["192.168.1.1"] = [Mock()]
        response.threat_history["192.168.1.2"] = [Mock()]
        
        # Clear specific IP
        response.clear_threat_history("192.168.1.1")
        assert "192.168.1.1" not in response.threat_history
        assert "192.168.1.2" in response.threat_history
        
        # Clear all
        response.clear_threat_history()
        assert len(response.threat_history) == 0

class TestFirewallManager:
    """Test firewall manager functionality"""
    
    def test_firewall_manager_initialization(self):
        """Test firewall manager initialization"""
        manager = FirewallManager()
        
        assert len(manager.blocked_ips) == 0
        assert len(manager.throttled_ips) == 0
        assert len(manager.rate_limits) == 0
        assert manager.stats["rules_created"] == 0
        assert manager.stats["rules_removed"] == 0
        assert manager.stats["blocks_active"] == 0
        assert manager.stats["throttles_active"] == 0
    
    def test_block_ip(self):
        """Test blocking IP"""
        manager = FirewallManager()
        
        success = manager.block_ip("192.168.1.1", 3600, "test_reason")
        assert success
        assert "192.168.1.1" in manager.blocked_ips
        assert manager.stats["rules_created"] == 1
        assert manager.stats["blocks_active"] == 1
    
    def test_unblock_ip(self):
        """Test unblocking IP"""
        manager = FirewallManager()
        
        manager.blocked_ips["192.168.1.1"] = {"block_until": time.time() + 3600}
        
        success = manager.unblock_ip("192.168.1.1")
        assert success
        assert "192.168.1.1" not in manager.blocked_ips
        assert manager.stats["rules_removed"] == 1
        
        # Test unblocking non-blocked IP
        success = manager.unblock_ip("192.168.1.2")
        assert not success
    
    def test_throttle_ip(self):
        """Test throttling IP"""
        manager = FirewallManager()
        
        success = manager.throttle_ip("192.168.1.1", 10, 300)
        assert success
        assert "192.168.1.1" in manager.throttled_ips
        assert manager.stats["rules_created"] == 1
        assert manager.stats["throttles_active"] == 1
    
    def test_unthrottle_ip(self):
        """Test unthrottling IP"""
        manager = FirewallManager()
        
        manager.throttled_ips["192.168.1.1"] = {"throttle_until": time.time() + 300}
        
        success = manager.unthrottle_ip("192.168.1.1")
        assert success
        assert "192.168.1.1" not in manager.throttled_ips
        assert manager.stats["rules_removed"] == 1
        
        # Test unthrottling non-throttled IP
        success = manager.unthrottle_ip("192.168.1.2")
        assert not success
    
    def test_is_ip_blocked(self):
        """Test checking if IP is blocked"""
        manager = FirewallManager()
        
        # Initially not blocked
        assert not manager.is_ip_blocked("192.168.1.1")
        
        # Block IP
        manager.blocked_ips["192.168.1.1"] = {"block_until": time.time() + 3600}
        assert manager.is_ip_blocked("192.168.1.1")
        
        # Expired block
        manager.blocked_ips["192.168.1.2"] = {"block_until": time.time() - 100}
        assert not manager.is_ip_blocked("192.168.1.2")
    
    def test_is_ip_throttled(self):
        """Test checking if IP is throttled"""
        manager = FirewallManager()
        
        # Initially not throttled
        assert not manager.is_ip_throttled("192.168.1.1")
        
        # Throttle IP
        manager.throttled_ips["192.168.1.1"] = {"throttle_until": time.time() + 300}
        assert manager.is_ip_throttled("192.168.1.1")
        
        # Expired throttle
        manager.throttled_ips["192.168.1.2"] = {"throttle_until": time.time() - 100}
        assert not manager.is_ip_throttled("192.168.1.2")
    
    def test_get_blocked_ips(self):
        """Test getting blocked IPs"""
        manager = FirewallManager()
        
        manager.blocked_ips["192.168.1.1"] = {"block_until": time.time() + 3600}
        manager.blocked_ips["192.168.1.2"] = {"block_until": time.time() - 100}  # Expired
        
        blocked = manager.get_blocked_ips()
        assert len(blocked) == 1  # Only non-expired
        assert "192.168.1.1" in blocked
        assert "192.168.1.2" not in blocked
    
    def test_get_throttled_ips(self):
        """Test getting throttled IPs"""
        manager = FirewallManager()
        
        manager.throttled_ips["192.168.1.1"] = {"throttle_until": time.time() + 300}
        manager.throttled_ips["192.168.1.2"] = {"throttle_until": time.time() - 100}  # Expired
        
        throttled = manager.get_throttled_ips()
        assert len(throttled) == 1  # Only non-expired
        assert "192.168.1.1" in throttled
        assert "192.168.1.2" not in throttled
    
    def test_get_stats(self):
        """Test getting statistics"""
        manager = FirewallManager()
        
        # Add some rules
        manager.block_ip("192.168.1.1")
        manager.throttle_ip("192.168.1.2")
        
        stats = manager.get_stats()
        assert stats["rules_created"] == 2
        assert stats["blocks_active"] == 1
        assert stats["throttles_active"] == 1
    
    def test_clear_all_rules(self):
        """Test clearing all rules"""
        manager = FirewallManager()
        
        # Add some rules
        manager.block_ip("192.168.1.1")
        manager.throttle_ip("192.168.1.2")
        
        manager.clear_all_rules()
        
        assert len(manager.blocked_ips) == 0
        assert len(manager.throttled_ips) == 0
        assert manager.stats["blocks_active"] == 0
        assert manager.stats["throttles_active"] == 0

class TestResponseStrategies:
    """Test response strategies"""
    
    def test_immediate_block_strategy(self):
        """Test immediate block strategy"""
        strategy = ImmediateBlockStrategy()
        
        threat = {
            "source_ip": "192.168.1.1",
            "threat_type": "brute_force"
        }
        
        result = strategy.execute(threat)
        
        assert result["action"] == "block"
        assert result["ip"] == "192.168.1.1"
        assert result["duration"] == 3600
        assert "immediate_block_brute_force" in result["reason"]
    
    def test_gradual_escalation_strategy(self):
        """Test gradual escalation strategy"""
        strategy = GradualEscalationStrategy()
        
        # Low threat count
        threat = {
            "source_ip": "192.168.1.1",
            "threat_type": "xss",
            "threat_count": 1
        }
        
        result = strategy.execute(threat)
        assert result["action"] == "throttle"
        assert result["duration"] == 300
        
        # Medium threat count
        threat["threat_count"] = 3
        result = strategy.execute(threat)
        assert result["action"] == "throttle"
        assert result["duration"] == 1800
        
        # High threat count
        threat["threat_count"] = 6
        result = strategy.execute(threat)
        assert result["action"] == "block"
        assert result["duration"] == 3600
    
    def test_decoy_response_strategy(self):
        """Test decoy response strategy"""
        strategy = DecoyResponseStrategy()
        
        threat = {
            "source_ip": "192.168.1.1",
            "threat_type": "sql_injection"
        }
        
        result = strategy.execute(threat)
        
        assert result["action"] == "decoy"
        assert result["ip"] == "192.168.1.1"
        assert "Database connection error" in result["response"]
    
    def test_adaptive_strategy(self):
        """Test adaptive strategy"""
        strategy = AdaptiveStrategy()
        
        # Low confidence, few threats
        threat = {
            "source_ip": "192.168.1.1",
            "threat_type": "anomaly",
            "confidence": 0.5
        }
        
        result = strategy.execute(threat)
        assert result["action"] == "throttle"
        assert result["duration"] == 300
        
        # High confidence, many threats
        threat["confidence"] = 0.95
        strategy.threat_history["192.168.1.1"] = [
            {"confidence": 0.9},
            {"confidence": 0.8},
            {"confidence": 0.7}
        ]
        
        result = strategy.execute(threat)
        assert result["action"] == "block"
        assert result["duration"] == 3600  # 1 hour for 4 threats with avg confidence 0.84
    
    def test_passive_monitoring_strategy(self):
        """Test passive monitoring strategy"""
        strategy = PassiveMonitoringStrategy()
        
        threat = {
            "source_ip": "192.168.1.1",
            "threat_type": "scanning"
        }
        
        result = strategy.execute(threat)
        
        assert result["action"] == "monitor"
        assert result["ip"] == "192.168.1.1"
        assert "passive_monitoring_scanning" in result["reason"]

class TestResponseStrategyManager:
    """Test response strategy manager"""
    
    def test_strategy_manager_initialization(self):
        """Test strategy manager initialization"""
        manager = ResponseStrategyManager()
        
        assert "immediate_block" in manager.strategies
        assert "gradual_escalation" in manager.strategies
        assert "decoy_response" in manager.strategies
        assert "adaptive" in manager.strategies
        assert "passive_monitoring" in manager.strategies
    
    def test_get_strategy(self):
        """Test getting strategy"""
        manager = ResponseStrategyManager()
        
        strategy = manager.get_strategy("immediate_block")
        assert isinstance(strategy, ImmediateBlockStrategy)
        
        strategy = manager.get_strategy("invalid")
        assert strategy is None
    
    def test_execute_strategy(self):
        """Test executing strategy"""
        manager = ResponseStrategyManager()
        
        threat = {
            "source_ip": "192.168.1.1",
            "threat_type": "brute_force"
        }
        
        result = manager.execute_strategy("immediate_block", threat)
        assert result is not None
        assert result["action"] == "block"
        
        result = manager.execute_strategy("invalid", threat)
        assert result is None
    
    def test_get_all_strategies(self):
        """Test getting all strategies"""
        manager = ResponseStrategyManager()
        
        strategies = manager.get_all_strategies()
        assert len(strategies) == 5
        assert "immediate_block" in strategies
        assert "gradual_escalation" in strategies
        assert "decoy_response" in strategies
        assert "adaptive" in strategies
        assert "passive_monitoring" in strategies
    
    def test_get_strategy_info(self):
        """Test getting strategy information"""
        manager = ResponseStrategyManager()
        
        info = manager.get_strategy_info()
        assert len(info) == 5
        
        immediate_info = info["immediate_block"]
        assert "name" in immediate_info
        assert "description" in immediate_info
        assert "type" in immediate_info

if __name__ == "__main__":
    pytest.main([__file__])
