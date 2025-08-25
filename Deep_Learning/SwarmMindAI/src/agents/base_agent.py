"""
Base agent class for SwarmMindAI.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import uuid
import math


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the swarm.
    
    Features:
    - Basic movement and sensing capabilities
    - Energy management
    - Communication interfaces
    - State management
    """
    
    def __init__(self, agent_id: str, position: Tuple[float, float], 
                 agent_type: str, world, capabilities: Dict[str, Any] = None):
        """
        Initialize a base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            position: Initial (x, y) position
            agent_type: Type of agent (explorer, collector, coordinator)
            world: Reference to the world object
            capabilities: Dictionary of agent capabilities
        """
        self.agent_id = agent_id
        self.position = np.array(position, dtype=np.float32)
        self.agent_type = agent_type
        self.world = world
        
        # Default capabilities
        self.capabilities = capabilities or {
            "mobility": True,
            "sensor_capability": True,
            "collection_capability": False,
            "communication": True
        }
        
        # Physical properties
        self.radius = 15.0
        self.max_speed = 5.0
        self.energy = 100.0
        self.max_energy = 100.0
        self.energy_consumption_rate = 0.1
        
        # Movement state
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.target_position = None
        self.movement_mode = "idle"  # idle, moving, following, avoiding
        
        # Sensing and perception
        self.sensor_range = 50.0
        self.vision_range = 100.0
        self.collision_detection_range = 25.0
        
        # Memory and learning
        self.memory = []
        self.experience_buffer = []
        self.learning_rate = 0.01
        
        # Communication
        self.message_queue = []
        self.broadcast_range = 80.0
        self.local_messages = []
        
        # Task management
        self.current_task = None
        self.task_history = []
        self.task_success_rate = 0.0
        
        # Performance metrics
        self.steps_alive = 0
        self.resources_collected = 0
        self.collisions_avoided = 0
        self.tasks_completed = 0
        
        # Initialize sensors
        self._initialize_sensors()
    
    def _initialize_sensors(self):
        """Initialize agent sensors."""
        self.sensors = {
            "proximity": [],
            "vision": [],
            "collision": [],
            "resource": [],
            "obstacle": []
        }
    
    @abstractmethod
    def decide_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide on the next action based on observations.
        
        Args:
            observations: Current observations from the environment
            
        Returns:
            Dictionary describing the chosen action
        """
        pass
    
    def move(self, direction: Tuple[float, float], speed: Optional[float] = None):
        """
        Move the agent in a specified direction.
        
        Args:
            direction: (dx, dy) direction vector
            speed: Movement speed (uses max_speed if None)
        """
        if speed is None:
            speed = self.max_speed
        
        # Normalize direction vector
        direction = np.array(direction, dtype=np.float32)
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        
        # Calculate new velocity
        self.velocity = direction * speed
        
        # Calculate new position
        new_position = self.position + self.velocity
        
        # Check for collisions
        if not self.world.check_collision(new_position[0], new_position[1], self.radius):
            self.position = new_position
            self.movement_mode = "moving"
        else:
            # Try to avoid obstacle
            self._avoid_obstacle()
    
    def _avoid_obstacle(self):
        """Implement obstacle avoidance behavior."""
        # Simple obstacle avoidance: move perpendicular to obstacle direction
        nearby_obstacles = self.world.get_nearby_obstacles(
            self.position[0], self.position[1], self.collision_detection_range
        )
        
        if nearby_obstacles:
            # Find closest obstacle
            closest_obstacle = min(nearby_obstacles, 
                                 key=lambda o: np.linalg.norm(self.position - np.array([o.x, o.y])))
            
            # Calculate avoidance direction
            obstacle_pos = np.array([closest_obstacle.x, closest_obstacle.y])
            to_obstacle = obstacle_pos - self.position
            avoidance_direction = np.array([-to_obstacle[1], to_obstacle[0]])  # Perpendicular
            
            # Normalize and move
            if np.linalg.norm(avoidance_direction) > 0:
                avoidance_direction = avoidance_direction / np.linalg.norm(avoidance_direction)
                self.move(avoidance_direction, self.max_speed * 0.5)
                self.movement_mode = "avoiding"
                self.collisions_avoided += 1
    
    def sense_environment(self) -> Dict[str, Any]:
        """
        Sense the environment around the agent.
        
        Returns:
            Dictionary containing sensory information
        """
        observations = {
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "energy": self.energy,
            "nearby_resources": [],
            "nearby_obstacles": [],
            "nearby_agents": [],
            "messages": self.local_messages.copy()
        }
        
        # Sense nearby resources
        nearby_resources = self.world.get_nearby_resources(
            self.position[0], self.position[1], self.sensor_range
        )
        observations["nearby_resources"] = [
            {
                "type": r.resource_type,
                "position": (r.x, r.y),
                "value": r.value,
                "distance": np.linalg.norm(self.position - np.array([r.x, r.y]))
            }
            for r in nearby_resources
        ]
        
        # Sense nearby obstacles
        nearby_obstacles = self.world.get_nearby_obstacles(
            self.position[0], self.position[1], self.sensor_range
        )
        observations["nearby_obstacles"] = [
            {
                "position": (o.x, o.y),
                "radius": o.radius,
                "type": o.obstacle_type,
                "distance": np.linalg.norm(self.position - np.array([o.x, o.y]))
            }
            for o in nearby_obstacles
        ]
        
        # Sense nearby agents (simplified - would need agent references)
        observations["nearby_agents"] = []
        
        return observations
    
    def collect_resource(self, resource_position: Tuple[float, float]) -> bool:
        """
        Attempt to collect a resource at the specified position.
        
        Args:
            resource_position: (x, y) position of the resource
            
        Returns:
            True if resource was collected, False otherwise
        """
        if not self.capabilities.get("collection_capability", False):
            return False
        
        distance = np.linalg.norm(self.position - np.array(resource_position))
        if distance <= self.radius:
            # Try to collect the resource
            collected_resource = self.world.collect_resource(
                resource_position[0], resource_position[1], self.radius
            )
            
            if collected_resource:
                self.resources_collected += 1
                self.energy = min(self.max_energy, self.energy + 10.0)  # Energy boost
                return True
        
        return False
    
    def send_message(self, message: Dict[str, Any], target_agent: Optional[str] = None):
        """
        Send a message to another agent or broadcast.
        
        Args:
            message: Message content
            target_agent: Target agent ID (None for broadcast)
        """
        message_data = {
            "sender": self.agent_id,
            "target": target_agent,
            "content": message,
            "timestamp": self.steps_alive,
            "position": self.position.copy()
        }
        
        if target_agent:
            # Direct message
            self.message_queue.append(message_data)
        else:
            # Broadcast message
            self.message_queue.append(message_data)
    
    def receive_message(self, message: Dict[str, Any]):
        """Receive a message from another agent."""
        # Check if message is within broadcast range
        sender_position = message.get("position", [0, 0])
        distance = np.linalg.norm(self.position - np.array(sender_position))
        
        if distance <= self.broadcast_range:
            self.local_messages.append(message)
    
    def update_energy(self):
        """Update agent energy levels."""
        # Consume energy based on movement and actions
        energy_consumption = self.energy_consumption_rate
        
        if self.movement_mode == "moving":
            energy_consumption += 0.2
        elif self.movement_mode == "avoiding":
            energy_consumption += 0.3
        
        self.energy = max(0.0, self.energy - energy_consumption)
    
    def is_active(self) -> bool:
        """Check if the agent is still active."""
        return self.energy > 0.0
    
    def get_agent_state(self) -> Dict[str, Any]:
        """Get the current state of the agent."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "energy": self.energy,
            "max_energy": self.max_energy,
            "movement_mode": self.movement_mode,
            "current_task": self.current_task,
            "capabilities": self.capabilities.copy(),
            "performance": {
                "steps_alive": self.steps_alive,
                "resources_collected": self.resources_collected,
                "collisions_avoided": self.collisions_avoided,
                "tasks_completed": self.tasks_completed,
                "task_success_rate": self.task_success_rate
            }
        }
    
    def step(self) -> Dict[str, Any]:
        """
        Execute one agent step.
        
        Returns:
            Dictionary containing agent actions and state
        """
        if not self.is_active():
            return {"agent_id": self.agent_id, "actions": [], "status": "inactive"}
        
        self.steps_alive += 1
        
        # Sense environment
        observations = self.sense_environment()
        
        # Decide on action
        action = self.decide_action(observations)
        
        # Execute action
        actions = self._execute_action(action)
        
        # Update energy
        self.update_energy()
        
        # Clear old messages
        if len(self.local_messages) > 10:
            self.local_messages = self.local_messages[-10:]
        
        return {
            "agent_id": self.agent_id,
            "actions": actions,
            "status": "active",
            "position": self.position.tolist(),
            "energy": self.energy
        }
    
    def _execute_action(self, action: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute the chosen action.
        
        Args:
            action: Action to execute
            
        Returns:
            List of action results
        """
        actions = []
        
        if "move" in action:
            direction = action["move"]["direction"]
            speed = action["move"].get("speed")
            self.move(direction, speed)
            actions.append({"action": "move", "direction": direction, "speed": speed})
        
        if "collect" in action:
            resource_pos = action["collect"]["position"]
            success = self.collect_resource(resource_pos)
            actions.append({
                "action": "collect", 
                "position": resource_pos, 
                "success": success,
                "resource_collected": success
            })
        
        if "communicate" in action:
            message = action["communicate"]["message"]
            target = action["communicate"].get("target")
            self.send_message(message, target)
            actions.append({
                "action": "communicate", 
                "message": message, 
                "target": target
            })
        
        return actions
    
    def reset(self):
        """Reset the agent to initial state."""
        self.energy = self.max_energy
        self.steps_alive = 0
        self.resources_collected = 0
        self.collisions_avoided = 0
        self.tasks_completed = 0
        self.task_success_rate = 0.0
        self.movement_mode = "idle"
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.message_queue.clear()
        self.local_messages.clear()
        self.memory.clear()
        self.experience_buffer.clear()
    
    def close(self):
        """Clean up agent resources."""
        self.message_queue.clear()
        self.local_messages.clear()
        self.memory.clear()
        self.experience_buffer.clear()
