"""
Response Strategies for HoneypotAI
Defines different response strategies for various threat types
"""

from typing import Dict, Any, Optional
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)

class ResponseStrategy:
    """Base class for response strategies"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = structlog.get_logger(f"adapt.strategy.{name}")
    
    def execute(self, threat: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the response strategy"""
        raise NotImplementedError("Subclasses must implement execute()")
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__
        }

class ImmediateBlockStrategy(ResponseStrategy):
    """Immediate blocking strategy for high-priority threats"""
    
    def __init__(self):
        super().__init__(
            "immediate_block",
            "Immediately block IP addresses for high-priority threats"
        )
    
    def execute(self, threat: Dict[str, Any]) -> Dict[str, Any]:
        """Execute immediate blocking"""
        source_ip = threat.get("source_ip", "")
        threat_type = threat.get("threat_type", "unknown")
        
        self.logger.warning(f"Executing immediate block for {source_ip} (threat: {threat_type})")
        
        return {
            "action": "block",
            "ip": source_ip,
            "duration": 3600,  # 1 hour
            "reason": f"immediate_block_{threat_type}",
            "timestamp": datetime.now().isoformat()
        }

class GradualEscalationStrategy(ResponseStrategy):
    """Gradual escalation strategy for repeated threats"""
    
    def __init__(self):
        super().__init__(
            "gradual_escalation",
            "Gradually escalate response based on threat frequency"
        )
    
    def execute(self, threat: Dict[str, Any]) -> Dict[str, Any]:
        """Execute gradual escalation"""
        source_ip = threat.get("source_ip", "")
        threat_type = threat.get("threat_type", "unknown")
        threat_count = threat.get("threat_count", 1)
        
        if threat_count <= 2:
            action = "throttle"
            duration = 300  # 5 minutes
        elif threat_count <= 5:
            action = "throttle"
            duration = 1800  # 30 minutes
        else:
            action = "block"
            duration = 3600  # 1 hour
        
        self.logger.info(f"Executing gradual escalation for {source_ip} (count: {threat_count}, action: {action})")
        
        return {
            "action": action,
            "ip": source_ip,
            "duration": duration,
            "reason": f"gradual_escalation_{threat_type}",
            "timestamp": datetime.now().isoformat()
        }

class DecoyResponseStrategy(ResponseStrategy):
    """Decoy response strategy to mislead attackers"""
    
    def __init__(self):
        super().__init__(
            "decoy_response",
            "Send misleading responses to confuse attackers"
        )
    
    def execute(self, threat: Dict[str, Any]) -> Dict[str, Any]:
        """Execute decoy response"""
        source_ip = threat.get("source_ip", "")
        threat_type = threat.get("threat_type", "unknown")
        
        # Generate decoy response based on threat type
        decoy_response = self._generate_decoy_response(threat_type)
        
        self.logger.info(f"Executing decoy response for {source_ip} (threat: {threat_type})")
        
        return {
            "action": "decoy",
            "ip": source_ip,
            "response": decoy_response,
            "reason": f"decoy_response_{threat_type}",
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_decoy_response(self, threat_type: str) -> str:
        """Generate appropriate decoy response"""
        decoy_responses = {
            "brute_force": "Access denied. Too many failed attempts.",
            "sql_injection": "Database connection error. Please try again later.",
            "xss": "Invalid input detected. Security violation.",
            "path_traversal": "File not found. Access denied.",
            "command_injection": "Command execution failed. Permission denied.",
            "scanning": "Service temporarily unavailable.",
            "dos": "Rate limit exceeded. Please slow down.",
            "anomaly": "Unusual activity detected. Access restricted.",
            "default": "Service error. Please contact administrator."
        }
        
        return decoy_responses.get(threat_type, decoy_responses["default"])

class AdaptiveStrategy(ResponseStrategy):
    """Adaptive strategy that learns from threat patterns"""
    
    def __init__(self):
        super().__init__(
            "adaptive",
            "Adaptive response based on threat patterns and ML insights"
        )
        self.threat_history = {}
    
    def execute(self, threat: Dict[str, Any]) -> Dict[str, Any]:
        """Execute adaptive response"""
        source_ip = threat.get("source_ip", "")
        threat_type = threat.get("threat_type", "unknown")
        confidence = threat.get("confidence", 0.0)
        
        # Update threat history
        if source_ip not in self.threat_history:
            self.threat_history[source_ip] = []
        
        self.threat_history[source_ip].append({
            "threat_type": threat_type,
            "confidence": confidence,
            "timestamp": datetime.now()
        })
        
        # Analyze threat pattern
        threat_count = len(self.threat_history[source_ip])
        avg_confidence = sum(t["confidence"] for t in self.threat_history[source_ip]) / threat_count
        
        # Determine response based on pattern
        if avg_confidence > 0.9 and threat_count > 2:
            action = "block"
            duration = 7200  # 2 hours
        elif avg_confidence > 0.7 or threat_count > 5:
            action = "block"
            duration = 3600  # 1 hour
        elif threat_count > 3:
            action = "throttle"
            duration = 1800  # 30 minutes
        else:
            action = "throttle"
            duration = 300  # 5 minutes
        
        self.logger.info(f"Executing adaptive response for {source_ip} (confidence: {avg_confidence:.2f}, count: {threat_count}, action: {action})")
        
        return {
            "action": action,
            "ip": source_ip,
            "duration": duration,
            "reason": f"adaptive_{threat_type}",
            "confidence": avg_confidence,
            "threat_count": threat_count,
            "timestamp": datetime.now().isoformat()
        }

class PassiveMonitoringStrategy(ResponseStrategy):
    """Passive monitoring strategy for low-priority threats"""
    
    def __init__(self):
        super().__init__(
            "passive_monitoring",
            "Passive monitoring without active response"
        )
    
    def execute(self, threat: Dict[str, Any]) -> Dict[str, Any]:
        """Execute passive monitoring"""
        source_ip = threat.get("source_ip", "")
        threat_type = threat.get("threat_type", "unknown")
        
        self.logger.info(f"Executing passive monitoring for {source_ip} (threat: {threat_type})")
        
        return {
            "action": "monitor",
            "ip": source_ip,
            "reason": f"passive_monitoring_{threat_type}",
            "timestamp": datetime.now().isoformat()
        }

class ResponseStrategyManager:
    """Manager for response strategies"""
    
    def __init__(self):
        self.strategies = {
            "immediate_block": ImmediateBlockStrategy(),
            "gradual_escalation": GradualEscalationStrategy(),
            "decoy_response": DecoyResponseStrategy(),
            "adaptive": AdaptiveStrategy(),
            "passive_monitoring": PassiveMonitoringStrategy()
        }
        
        self.logger = structlog.get_logger("adapt.strategy_manager")
    
    def get_strategy(self, strategy_name: str) -> Optional[ResponseStrategy]:
        """Get a strategy by name"""
        return self.strategies.get(strategy_name)
    
    def execute_strategy(self, strategy_name: str, threat: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a strategy by name"""
        strategy = self.get_strategy(strategy_name)
        if strategy:
            return strategy.execute(threat)
        else:
            self.logger.error(f"Unknown strategy: {strategy_name}")
            return None
    
    def get_all_strategies(self) -> Dict[str, ResponseStrategy]:
        """Get all available strategies"""
        return self.strategies.copy()
    
    def get_strategy_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all strategies"""
        return {
            name: strategy.get_strategy_info()
            for name, strategy in self.strategies.items()
        }
