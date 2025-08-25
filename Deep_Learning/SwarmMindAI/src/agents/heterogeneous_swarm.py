"""
Heterogeneous swarm management for SwarmMindAI.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random
import uuid
from .base_agent import BaseAgent
from .agent_types import ExplorerAgent, CollectorAgent, CoordinatorAgent


class HeterogeneousSwarm:
    """
    Manages a heterogeneous swarm of different agent types.
    
    Features:
    - Dynamic agent creation and management
    - Inter-agent communication and coordination
    - Swarm formation and behavior patterns
    - Performance monitoring and optimization
    """
    
    def __init__(self, world, num_agents: int = 20, 
                 agent_types: List[str] = None, seed: Optional[int] = None):
        """
        Initialize the heterogeneous swarm.
        
        Args:
            world: World object for the simulation
            num_agents: Total number of agents in the swarm
            agent_types: List of agent types to create
            seed: Random seed for reproducibility
        """
        self.world = world
        self.num_agents = num_agents
        self.agent_types = agent_types or ["explorer", "collector", "coordinator"]
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Agent management
        self.agents: List[BaseAgent] = []
        self.agent_type_counts = {}
        self.agent_positions = {}
        
        # Communication and coordination
        self.global_message_queue = []
        self.broadcast_messages = []
        self.agent_communication_graph = {}
        
        # Swarm behavior
        self.formation_mode = "dispersed"  # dispersed, clustered, circular, line
        self.coordination_level = "medium"  # low, medium, high
        self.adaptation_rate = 0.1
        
        # Performance tracking
        self.swarm_metrics = {}
        self.performance_history = []
        self.optimization_interval = 100
        
        # Initialize the swarm
        self._initialize_swarm()
        self._setup_communication_network()
    
    def _initialize_swarm(self):
        """Initialize the swarm with agents of different types."""
        # Calculate agent type distribution
        total_types = len(self.agent_types)
        base_count = self.num_agents // total_types
        remainder = self.num_agents % total_types
        
        # Distribute agents by type
        for i, agent_type in enumerate(self.agent_types):
            count = base_count + (1 if i < remainder else 0)
            self.agent_type_counts[agent_type] = count
        
        # Create agents
        agent_id_counter = 0
        for agent_type, count in self.agent_type_counts.items():
            for _ in range(count):
                agent_id = f"{agent_type}_{agent_id_counter}"
                position = self._generate_agent_position()
                
                # Create agent based on type
                if agent_type == "explorer":
                    agent = ExplorerAgent(agent_id, position, self.world)
                elif agent_type == "collector":
                    agent = CollectorAgent(agent_id, position, self.world)
                elif agent_type == "coordinator":
                    agent = CoordinatorAgent(agent_id, position, self.world)
                else:
                    # Default to base agent
                    agent = BaseAgent(agent_id, position, agent_type, self.world)
                
                self.agents.append(agent)
                self.agent_positions[agent_id] = position
                agent_id_counter += 1
        
        # Ensure minimum number of each type
        self._ensure_minimum_agent_types()
    
    def _generate_agent_position(self) -> Tuple[float, float]:
        """Generate a valid position for a new agent."""
        max_attempts = 100
        for _ in range(max_attempts):
            x = random.uniform(50, self.world.width - 50)
            y = random.uniform(50, self.world.height - 50)
            
            # Check if position is valid (no collisions)
            if not self.world.check_collision(x, y, 20.0):
                return (x, y)
        
        # Fallback to center if no valid position found
        return (self.world.width // 2, self.world.height // 2)
    
    def _ensure_minimum_agent_types(self):
        """Ensure minimum number of each agent type."""
        min_counts = {
            "explorer": 2,
            "collector": 3,
            "coordinator": 1
        }
        
        for agent_type, min_count in min_counts.items():
            current_count = self.agent_type_counts.get(agent_type, 0)
            if current_count < min_count:
                # Add more agents of this type
                for _ in range(min_count - current_count):
                    agent_id = f"{agent_type}_{len(self.agents)}"
                    position = self._generate_agent_position()
                    
                    if agent_type == "explorer":
                        agent = ExplorerAgent(agent_id, position, self.world)
                    elif agent_type == "collector":
                        agent = CollectorAgent(agent_id, position, self.world)
                    elif agent_type == "coordinator":
                        agent = CoordinatorAgent(agent_id, position, self.world)
                    
                    self.agents.append(agent)
                    self.agent_positions[agent_id] = position
                    self.agent_type_counts[agent_type] = self.agent_type_counts.get(agent_type, 0) + 1
    
    def _setup_communication_network(self):
        """Setup the communication network between agents."""
        # Initialize communication graph
        for agent in self.agents:
            self.agent_communication_graph[agent.agent_id] = []
        
        # Create communication links based on agent types
        for agent in self.agents:
            if agent.agent_type == "coordinator":
                # Coordinators can communicate with all agents
                for other_agent in self.agents:
                    if other_agent.agent_id != agent.agent_id:
                        self.agent_communication_graph[agent.agent_id].append(other_agent.agent_id)
            elif agent.agent_type == "explorer":
                # Explorers communicate with coordinators and nearby agents
                for other_agent in self.agents:
                    if other_agent.agent_type == "coordinator":
                        self.agent_communication_graph[agent.agent_id].append(other_agent.agent_id)
                    elif self._agents_are_nearby(agent, other_agent):
                        self.agent_communication_graph[agent.agent_id].append(other_agent.agent_id)
            elif agent.agent_type == "collector":
                # Collectors communicate with coordinators and nearby collectors
                for other_agent in self.agents:
                    if other_agent.agent_type == "coordinator":
                        self.agent_communication_graph[agent.agent_id].append(other_agent.agent_id)
                    elif (other_agent.agent_type == "collector" and 
                          self._agents_are_nearby(agent, other_agent)):
                        self.agent_communication_graph[agent.agent_id].append(other_agent.agent_id)
    
    def _agents_are_nearby(self, agent1: BaseAgent, agent2: BaseAgent, 
                           distance_threshold: float = 100.0) -> bool:
        """Check if two agents are nearby each other."""
        distance = np.linalg.norm(agent1.position - agent2.position)
        return distance <= distance_threshold
    
    def step(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Execute one swarm step.
        
        Returns:
            Dictionary mapping agent IDs to their actions
        """
        # Process global messages
        self._process_global_messages()
        
        # Execute agent steps
        agent_actions = {}
        for agent in self.agents:
            if agent.is_active():
                actions = agent.step()
                agent_actions[agent.agent_id] = actions.get("actions", [])
                
                # Update agent position
                self.agent_positions[agent.agent_id] = agent.position.tolist()
        
        # Process inter-agent communication
        self._process_agent_communication()
        
        # Update swarm formation if needed
        if self.steps_alive % 50 == 0:
            self._update_swarm_formation()
        
        # Optimize swarm performance
        if self.steps_alive % self.optimization_interval == 0:
            self._optimize_swarm_performance()
        
        # Update swarm metrics
        self._update_swarm_metrics()
        
        return agent_actions
    
    def _process_global_messages(self):
        """Process global messages and distribute to relevant agents."""
        for message in self.global_message_queue:
            # Distribute message to relevant agents
            for agent in self.agents:
                if self._should_receive_message(agent, message):
                    agent.receive_message(message)
        
        # Clear processed messages
        self.global_message_queue.clear()
    
    def _should_receive_message(self, agent: BaseAgent, message: Dict[str, Any]) -> bool:
        """Check if an agent should receive a specific message."""
        message_type = message.get("content", {}).get("type", "")
        
        if message_type == "task_assignment":
            # Task assignments go to relevant agents
            return agent.agent_type in ["collector", "explorer"]
        elif message_type == "resource_discovery":
            # Resource discoveries go to collectors and coordinators
            return agent.agent_type in ["collector", "coordinator"]
        elif message_type == "performance_optimization":
            # Performance optimizations go to all agents
            return True
        else:
            # Default: send to all agents
            return True
    
    def _process_agent_communication(self):
        """Process communication between agents."""
        # Collect messages from all agents
        for agent in self.agents:
            if agent.is_active() and agent.message_queue:
                for message in agent.message_queue:
                    # Process message based on target
                    if message["target"]:
                        # Direct message
                        target_agent = self._find_agent_by_id(message["target"])
                        if target_agent:
                            target_agent.receive_message(message)
                    else:
                        # Broadcast message
                        self._broadcast_message(message, agent)
                
                # Clear processed messages
                agent.message_queue.clear()
    
    def _find_agent_by_id(self, agent_id: str) -> Optional[BaseAgent]:
        """Find an agent by its ID."""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None
    
    def _broadcast_message(self, message: Dict[str, Any], sender: BaseAgent):
        """Broadcast a message to nearby agents."""
        for agent in self.agents:
            if agent.agent_id != sender.agent_id:
                # Check if agent is within broadcast range
                distance = np.linalg.norm(sender.position - agent.position)
                if distance <= sender.broadcast_range:
                    agent.receive_message(message)
    
    def _update_swarm_formation(self):
        """Update the swarm formation based on current mode."""
        if self.formation_mode == "dispersed":
            self._maintain_dispersed_formation()
        elif self.formation_mode == "clustered":
            self._maintain_clustered_formation()
        elif self.formation_mode == "circular":
            self._maintain_circular_formation()
        elif self.formation_mode == "line":
            self._maintain_line_formation()
    
    def _maintain_dispersed_formation(self):
        """Maintain a dispersed formation for exploration."""
        # No specific formation maintenance needed
        pass
    
    def _maintain_clustered_formation(self):
        """Maintain a clustered formation for resource collection."""
        # Move agents towards resource-rich areas
        pass
    
    def _maintain_circular_formation(self):
        """Maintain a circular formation for defense or coordination."""
        # Calculate swarm center
        if not self.agents:
            return
        
        center = np.mean([agent.position for agent in self.agents if agent.is_active()], axis=0)
        radius = 150.0
        
        # Position agents in a circle around the center
        for i, agent in enumerate(self.agents):
            if agent.is_active():
                angle = (2 * np.pi * i) / len(self.agents)
                target_x = center[0] + radius * np.cos(angle)
                target_y = center[1] + radius * np.sin(angle)
                
                # Keep within world bounds
                target_x = max(50, min(self.world.width - 50, target_x))
                target_y = max(50, min(self.world.height - 50, target_y))
                
                # Move agent towards target position
                direction = np.array([target_x, target_y]) - agent.position
                if np.linalg.norm(direction) > 10:  # Only move if far from target
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        agent.move(direction, agent.max_speed * 0.3)
    
    def _maintain_line_formation(self):
        """Maintain a line formation for systematic coverage."""
        # Position agents in a line formation
        pass
    
    def _optimize_swarm_performance(self):
        """Optimize swarm performance based on current metrics."""
        # Analyze current performance
        current_metrics = self._calculate_swarm_metrics()
        
        # Identify areas for improvement
        if current_metrics["coordination_score"] < 0.6:
            # Improve coordination
            self._improve_coordination()
        
        if current_metrics["resource_efficiency"] < 0.5:
            # Improve resource utilization
            self._improve_resource_utilization()
        
        if current_metrics["energy_efficiency"] < 0.7:
            # Improve energy efficiency
            self._improve_energy_efficiency()
    
    def _improve_coordination(self):
        """Improve swarm coordination."""
        # Increase communication frequency
        self.coordination_level = "high"
        
        # Adjust formation for better coordination
        if self.formation_mode == "dispersed":
            self.formation_mode = "clustered"
    
    def _improve_resource_utilization(self):
        """Improve resource utilization."""
        # Increase number of collector agents if needed
        collector_count = self.agent_type_counts.get("collector", 0)
        if collector_count < 5:
            self._add_agent("collector")
    
    def _improve_energy_efficiency(self):
        """Improve energy efficiency."""
        # Reduce movement speeds for energy conservation
        for agent in self.agents:
            if hasattr(agent, 'max_speed'):
                agent.max_speed *= 0.9
    
    def _add_agent(self, agent_type: str):
        """Add a new agent of the specified type."""
        agent_id = f"{agent_type}_{len(self.agents)}"
        position = self._generate_agent_position()
        
        if agent_type == "explorer":
            agent = ExplorerAgent(agent_id, position, self.world)
        elif agent_type == "collector":
            agent = CollectorAgent(agent_id, position, self.world)
        elif agent_type == "coordinator":
            agent = CoordinatorAgent(agent_id, position, self.world)
        else:
            return
        
        self.agents.append(agent)
        self.agent_positions[agent_id] = position
        self.agent_type_counts[agent_type] = self.agent_type_counts.get(agent_type, 0) + 1
        
        # Update communication network
        self._setup_communication_network()
    
    def _update_swarm_metrics(self):
        """Update swarm performance metrics."""
        if not self.agents:
            return
        
        active_agents = [a for a in self.agents if a.is_active()]
        
        # Calculate basic metrics
        total_energy = sum(agent.energy for agent in active_agents)
        avg_energy = total_energy / len(active_agents) if active_agents else 0
        
        # Calculate coordination score
        coordination_score = self._calculate_coordination_score()
        
        # Calculate resource efficiency
        resource_efficiency = self._calculate_resource_efficiency()
        
        # Calculate energy efficiency
        energy_efficiency = avg_energy / 100.0  # Normalize by max energy
        
        # Update metrics
        self.swarm_metrics = {
            "total_agents": len(self.agents),
            "active_agents": len(active_agents),
            "agent_type_distribution": self.agent_type_counts.copy(),
            "average_energy": avg_energy,
            "coordination_score": coordination_score,
            "resource_efficiency": resource_efficiency,
            "energy_efficiency": energy_efficiency,
            "formation_mode": self.formation_mode,
            "coordination_level": self.coordination_level
        }
        
        # Store in history
        self.performance_history.append(self.swarm_metrics.copy())
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def _calculate_coordination_score(self) -> float:
        """Calculate how well the swarm is coordinated."""
        if not self.agents:
            return 0.0
        
        # Calculate average distance between agents
        total_distance = 0
        count = 0
        
        for i, agent1 in enumerate(self.agents):
            if not agent1.is_active():
                continue
            for j, agent2 in enumerate(self.agents[i+1:], i+1):
                if agent2.is_active():
                    distance = np.linalg.norm(agent1.position - agent2.position)
                    total_distance += distance
                    count += 1
        
        if count == 0:
            return 0.0
        
        avg_distance = total_distance / count
        
        # Normalize distance (closer agents = better coordination)
        max_possible_distance = np.sqrt(self.world.width**2 + self.world.height**2)
        coordination_score = 1.0 - (avg_distance / max_possible_distance)
        
        return max(0.0, min(1.0, coordination_score))
    
    def _calculate_resource_efficiency(self) -> float:
        """Calculate resource collection efficiency."""
        if not self.agents:
            return 0.0
        
        # Count total resources collected
        total_resources = 0
        for agent in self.agents:
            if hasattr(agent, 'resources_collected'):
                total_resources += agent.resources_collected
        
        # Normalize by number of agents and time
        efficiency = total_resources / (len(self.agents) * max(self.steps_alive, 1))
        
        return min(1.0, efficiency)
    
    def get_swarm_state(self) -> Dict[str, Any]:
        """Get the current state of the swarm."""
        return {
            "num_agents": len(self.agents),
            "agent_types": self.agent_type_counts.copy(),
            "active_agents": len([a for a in self.agents if a.is_active()]),
            "formation_mode": self.formation_mode,
            "coordination_level": self.coordination_level,
            "metrics": self.swarm_metrics.copy(),
            "agent_positions": self.agent_positions.copy()
        }
    
    def reset(self):
        """Reset the swarm to initial state."""
        # Reset all agents
        for agent in self.agents:
            agent.reset()
        
        # Reset swarm state
        self.global_message_queue.clear()
        self.broadcast_messages.clear()
        self.performance_history.clear()
        self.swarm_metrics.clear()
        
        # Reset formation and coordination
        self.formation_mode = "dispersed"
        self.coordination_level = "medium"
        
        # Reposition agents
        for agent in self.agents:
            new_position = self._generate_agent_position()
            agent.position = np.array(new_position, dtype=np.float32)
            self.agent_positions[agent.agent_id] = new_position
    
    def close(self):
        """Clean up swarm resources."""
        for agent in self.agents:
            agent.close()
        
        self.agents.clear()
        self.agent_positions.clear()
        self.agent_communication_graph.clear()
        self.global_message_queue.clear()
        self.broadcast_messages.clear()
        self.performance_history.clear()
        self.swarm_metrics.clear()
    
    @property
    def steps_alive(self) -> int:
        """Get the number of steps the swarm has been alive."""
        if not self.agents:
            return 0
        return max(agent.steps_alive for agent in self.agents)
