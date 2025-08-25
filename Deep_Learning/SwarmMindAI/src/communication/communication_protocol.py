"""
Communication protocol for SwarmMindAI multi-agent coordination.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
import json


@dataclass
class Message:
    """Represents a message between agents."""
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    message_type: str
    content: Dict[str, Any]
    timestamp: float
    priority: int = 1
    expires_at: Optional[float] = None
    position: Optional[Tuple[float, float]] = None


class CommunicationProtocol:
    """
    Manages communication between agents in the swarm.
    
    Features:
    - Local and global message routing
    - Message prioritization and expiration
    - Pheromone-based communication
    - Bandwidth management
    - Message encryption and validation
    """
    
    def __init__(self, 
                 max_message_history: int = 1000,
                 max_bandwidth: int = 1000,
                 pheromone_decay_rate: float = 0.95):
        """
        Initialize the communication protocol.
        
        Args:
            max_message_history: Maximum number of messages to keep in history
            max_bandwidth: Maximum messages per time step
            pheromone_decay_rate: Rate at which pheromones decay
        """
        self.max_message_history = max_message_history
        self.max_bandwidth = max_bandwidth
        self.pheromone_decay_rate = pheromone_decay_rate
        
        # Message management
        self.message_queue = []
        self.message_history = []
        self.broadcast_messages = []
        
        # Pheromone system
        self.pheromones = {}  # position -> pheromone_strength
        self.pheromone_types = ["resource", "danger", "coordination", "exploration"]
        
        # Communication statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.bandwidth_usage = 0
        self.communication_errors = 0
        
        # Agent communication graph
        self.agent_connections = {}
        self.communication_ranges = {}
        
        # Message type handlers
        self.message_handlers = {
            "resource_discovery": self._handle_resource_discovery,
            "danger_alert": self._handle_danger_alert,
            "coordination_request": self._handle_coordination_request,
            "formation_update": self._handle_formation_update,
            "task_assignment": self._handle_task_assignment,
            "performance_report": self._handle_performance_report
        }
    
    def send_message(self, 
                    sender_id: str,
                    recipient_id: Optional[str],
                    message_type: str,
                    content: Dict[str, Any],
                    priority: int = 1,
                    expires_at: Optional[float] = None,
                    position: Optional[Tuple[float, float]] = None) -> bool:
        """
        Send a message to another agent or broadcast.
        
        Args:
            sender_id: ID of the sending agent
            recipient_id: ID of the receiving agent (None for broadcast)
            message_type: Type of message
            content: Message content
            priority: Message priority (higher = more important)
            expires_at: Expiration timestamp
            position: Position where message was sent
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        # Check bandwidth
        if self.bandwidth_usage >= self.max_bandwidth:
            return False
        
        # Create message
        message = Message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=message_type,
            content=content,
            timestamp=time.time(),
            priority=priority,
            expires_at=expires_at,
            position=position
        )
        
        # Add to appropriate queue
        if recipient_id:
            self.message_queue.append(message)
        else:
            self.broadcast_messages.append(message)
        
        # Update statistics
        self.messages_sent += 1
        self.bandwidth_usage += 1
        
        # Add to pheromone system if position is provided
        if position and message_type in self.pheromone_types:
            self._add_pheromone(position, message_type, priority)
        
        return True
    
    def receive_messages(self, agent_id: str, 
                        agent_position: Tuple[float, float],
                        communication_range: float) -> List[Message]:
        """
        Receive messages for a specific agent.
        
        Args:
            agent_id: ID of the receiving agent
            agent_position: Current position of the agent
            communication_range: Communication range of the agent
            
        Returns:
            List of received messages
        """
        received_messages = []
        
        # Check direct messages
        messages_to_remove = []
        for message in self.message_queue:
            if message.recipient_id == agent_id:
                # Check if message has expired
                if message.expires_at and time.time() > message.expires_at:
                    messages_to_remove.append(message)
                    continue
                
                received_messages.append(message)
                messages_to_remove.append(message)
        
        # Remove processed messages
        for message in messages_to_remove:
            self.message_queue.remove(message)
        
        # Check broadcast messages
        broadcast_to_remove = []
        for message in self.broadcast_messages:
            # Check if message has expired
            if message.expires_at and time.time() > message.expires_at:
                broadcast_to_remove.append(message)
                continue
            
            # Check if agent is within range
            if message.position:
                distance = np.linalg.norm(np.array(agent_position) - np.array(message.position))
                if distance <= communication_range:
                    received_messages.append(message)
            
            # Remove expired broadcast messages
            if message.expires_at and time.time() > message.expires_at:
                broadcast_to_remove.append(message)
        
        # Remove expired broadcast messages
        for message in broadcast_to_remove:
            self.broadcast_messages.remove(message)
        
        # Update statistics
        self.messages_received += len(received_messages)
        
        # Add to message history
        for message in received_messages:
            self._add_to_history(message)
        
        return received_messages
    
    def _add_pheromone(self, position: Tuple[float, float], 
                       pheromone_type: str, strength: float):
        """Add a pheromone to the environment."""
        key = (position, pheromone_type)
        
        if key in self.pheromones:
            # Increase existing pheromone strength
            self.pheromones[key] = min(1.0, self.pheromones[key] + strength)
        else:
            # Create new pheromone
            self.pheromones[key] = strength
    
    def get_pheromones(self, position: Tuple[float, float], 
                       radius: float, pheromone_type: Optional[str] = None) -> List[Tuple[Tuple[float, float], str, float]]:
        """
        Get pheromones within a certain radius of a position.
        
        Args:
            position: Center position
            radius: Search radius
            pheromone_type: Type of pheromone to search for (None for all types)
            
        Returns:
            List of (position, type, strength) tuples
        """
        nearby_pheromones = []
        
        for (pheromone_pos, p_type), strength in self.pheromones.items():
            # Check type filter
            if pheromone_type and p_type != pheromone_type:
                continue
            
            # Check distance
            distance = np.linalg.norm(np.array(position) - np.array(pheromone_pos))
            if distance <= radius:
                nearby_pheromones.append((pheromone_pos, p_type, strength))
        
        return nearby_pheromones
    
    def update_pheromones(self):
        """Update pheromone strengths (decay over time)."""
        pheromones_to_remove = []
        
        for key, strength in self.pheromones.items():
            # Decay pheromone strength
            self.pheromones[key] = strength * self.pheromone_decay_rate
            
            # Remove very weak pheromones
            if self.pheromones[key] < 0.01:
                pheromones_to_remove.append(key)
        
        # Remove weak pheromones
        for key in pheromones_to_remove:
            del self.pheromones[key]
    
    def _add_to_history(self, message: Message):
        """Add message to history."""
        self.message_history.append(message)
        
        # Keep only recent messages
        if len(self.message_history) > self.max_message_history:
            self.message_history.pop(0)
    
    def _handle_resource_discovery(self, message: Message) -> Dict[str, Any]:
        """Handle resource discovery messages."""
        content = message.content
        return {
            "type": "resource_discovery",
            "resources": content.get("resources", []),
            "position": content.get("position"),
            "timestamp": message.timestamp
        }
    
    def _handle_danger_alert(self, message: Message) -> Dict[str, Any]:
        """Handle danger alert messages."""
        content = message.content
        return {
            "type": "danger_alert",
            "danger_type": content.get("danger_type"),
            "position": content.get("position"),
            "severity": content.get("severity", 1),
            "timestamp": message.timestamp
        }
    
    def _handle_coordination_request(self, message: Message) -> Dict[str, Any]:
        """Handle coordination request messages."""
        content = message.content
        return {
            "type": "coordination_request",
            "request_type": content.get("request_type"),
            "parameters": content.get("parameters", {}),
            "priority": message.priority,
            "timestamp": message.timestamp
        }
    
    def _handle_formation_update(self, message: Message) -> Dict[str, Any]:
        """Handle formation update messages."""
        content = message.content
        return {
            "type": "formation_update",
            "formation_type": content.get("formation_type"),
            "target_positions": content.get("target_positions", []),
            "timestamp": message.timestamp
        }
    
    def _handle_task_assignment(self, message: Message) -> Dict[str, Any]:
        """Handle task assignment messages."""
        content = message.content
        return {
            "type": "task_assignment",
            "task_id": content.get("task_id"),
            "task_type": content.get("task_type"),
            "parameters": content.get("parameters", {}),
            "priority": message.priority,
            "timestamp": message.timestamp
        }
    
    def _handle_performance_report(self, message: Message) -> Dict[str, Any]:
        """Handle performance report messages."""
        content = message.content
        return {
            "type": "performance_report",
            "metrics": content.get("metrics", {}),
            "status": content.get("status"),
            "timestamp": message.timestamp
        }
    
    def process_message(self, message: Message) -> Dict[str, Any]:
        """Process a message using appropriate handler."""
        message_type = message.message_type
        
        if message_type in self.message_handlers:
            return self.message_handlers[message_type](message)
        else:
            # Default handler for unknown message types
            return {
                "type": "unknown",
                "original_type": message_type,
                "content": message.content,
                "timestamp": message.timestamp
            }
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "bandwidth_usage": self.bandwidth_usage,
            "communication_errors": self.communication_errors,
            "active_pheromones": len(self.pheromones),
            "message_queue_size": len(self.message_queue),
            "broadcast_queue_size": len(self.broadcast_messages),
            "history_size": len(self.message_history)
        }
    
    def reset_bandwidth(self):
        """Reset bandwidth usage counter."""
        self.bandwidth_usage = 0
    
    def clear_expired_messages(self):
        """Clear all expired messages."""
        current_time = time.time()
        
        # Clear expired direct messages
        self.message_queue = [
            msg for msg in self.message_queue
            if not msg.expires_at or current_time <= msg.expires_at
        ]
        
        # Clear expired broadcast messages
        self.broadcast_messages = [
            msg for msg in self.broadcast_messages
            if not msg.expires_at or current_time <= msg.expires_at
        ]
    
    def get_message_summary(self, time_window: float = 60.0) -> Dict[str, int]:
        """Get summary of messages in a time window."""
        current_time = time.time()
        start_time = current_time - time_window
        
        message_counts = {}
        
        for message in self.message_history:
            if message.timestamp >= start_time:
                msg_type = message.message_type
                message_counts[msg_type] = message_counts.get(msg_type, 0) + 1
        
        return message_counts
    
    def optimize_communication(self):
        """Optimize communication based on current usage."""
        # Clear old messages if memory usage is high
        if len(self.message_history) > self.max_message_history * 0.8:
            # Remove oldest 20% of messages
            remove_count = int(len(self.message_history) * 0.2)
            self.message_history = self.message_history[remove_count:]
        
        # Update pheromones
        self.update_pheromones()
        
        # Clear expired messages
        self.clear_expired_messages()
    
    def reset(self):
        """Reset the communication protocol."""
        self.message_queue.clear()
        self.message_history.clear()
        self.broadcast_messages.clear()
        self.pheromones.clear()
        self.agent_connections.clear()
        self.communication_ranges.clear()
        
        self.messages_sent = 0
        self.messages_received = 0
        self.bandwidth_usage = 0
        self.communication_errors = 0
