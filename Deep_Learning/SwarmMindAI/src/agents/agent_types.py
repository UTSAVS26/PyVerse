"""
Specific agent type implementations for SwarmMindAI.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random
import math
from .base_agent import BaseAgent


class ExplorerAgent(BaseAgent):
    """
    Explorer agent specialized in discovering new areas and resources.
    
    Features:
    - Efficient area coverage algorithms
    - Resource discovery and mapping
    - Obstacle avoidance and pathfinding
    - Communication of discoveries to other agents
    """
    
    def __init__(self, agent_id: str, position: Tuple[float, float], world):
        """Initialize an explorer agent."""
        capabilities = {
            "mobility": True,
            "sensor_capability": True,
            "collection_capability": False,
            "communication": True,
            "exploration": True
        }
        
        super().__init__(agent_id, position, "explorer", world, capabilities)
        
        # Explorer-specific properties
        self.exploration_mode = "systematic"  # systematic, random, adaptive
        self.coverage_grid = set()
        self.discovered_resources = []
        self.exploration_targets = []
        self.coverage_radius = 30.0
        
        # Enhanced sensing for exploration
        self.sensor_range = 80.0
        self.vision_range = 150.0
        
        # Movement patterns
        self.movement_pattern = "spiral"  # spiral, grid, random
        self.spiral_angle = 0.0
        self.grid_step = 40.0
        
        # Initialize exploration targets
        self._initialize_exploration_targets()
    
    def _initialize_exploration_targets(self):
        """Initialize exploration targets based on world size."""
        world_width, world_height = self.world.width, self.world.height
        
        # Create grid-based exploration targets
        grid_size = 100
        for x in range(50, world_width - 50, grid_size):
            for y in range(50, world_height - 50, grid_size):
                self.exploration_targets.append((x, y))
        
        # Shuffle targets for more natural exploration
        random.shuffle(self.exploration_targets)
    
    def decide_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Decide on the next action based on observations."""
        action = {}
        
        # Check for nearby resources to report
        if observations["nearby_resources"]:
            self._update_discovered_resources(observations["nearby_resources"])
            action["communicate"] = {
                "message": {
                    "type": "resource_discovery",
                    "resources": observations["nearby_resources"],
                    "position": observations["position"]
                }
            }
        
        # Check for obstacles and avoid them
        if observations["nearby_obstacles"]:
            action["move"] = self._calculate_avoidance_movement(observations["nearby_obstacles"])
        else:
            # Continue exploration
            action["move"] = self._calculate_exploration_movement(observations)
        
        return action
    
    def _calculate_exploration_movement(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate movement for exploration."""
        if self.movement_pattern == "spiral":
            return self._spiral_movement()
        elif self.movement_pattern == "grid":
            return self._grid_movement()
        else:
            return self._random_movement()
    
    def _spiral_movement(self) -> Dict[str, Any]:
        """Generate spiral movement pattern."""
        # Spiral outward from current position
        radius = 50.0 + (self.steps_alive * 0.1)
        angle = self.spiral_angle
        
        target_x = self.position[0] + radius * math.cos(angle)
        target_y = self.position[1] + radius * math.sin(angle)
        
        # Keep within world bounds
        target_x = max(50, min(self.world.width - 50, target_x))
        target_y = max(50, min(self.world.height - 50, target_y))
        
        direction = np.array([target_x - self.position[0], target_y - self.position[1]])
        
        # Update spiral angle
        self.spiral_angle += 0.1
        
        return {
            "direction": direction.tolist(),
            "speed": self.max_speed * 0.8
        }
    
    def _grid_movement(self) -> Dict[str, Any]:
        """Generate grid-based movement pattern."""
        if not self.exploration_targets:
            self._initialize_exploration_targets()
        
        # Move towards next target
        if self.exploration_targets:
            target = self.exploration_targets[0]
            direction = np.array(target) - self.position
            
            # If close to target, move to next
            if np.linalg.norm(direction) < self.grid_step:
                self.exploration_targets.pop(0)
                if self.exploration_targets:
                    target = self.exploration_targets[0]
                    direction = np.array(target) - self.position
            
            return {
                "direction": direction.tolist(),
                "speed": self.max_speed * 0.9
            }
        
        return self._random_movement()
    
    def _random_movement(self) -> Dict[str, Any]:
        """Generate random movement pattern."""
        # Random direction with slight bias towards unexplored areas
        angle = random.uniform(0, 2 * math.pi)
        direction = np.array([math.cos(angle), math.sin(angle)])
        
        return {
            "direction": direction.tolist(),
            "speed": self.max_speed * 0.7
        }
    
    def _calculate_avoidance_movement(self, obstacles: List[Dict]) -> Dict[str, Any]:
        """Calculate movement to avoid obstacles."""
        if not obstacles:
            return self._random_movement()
        
        # Find closest obstacle
        closest_obstacle = min(obstacles, key=lambda o: o["distance"])
        
        # Calculate avoidance direction (perpendicular to obstacle direction)
        obstacle_pos = np.array(closest_obstacle["position"])
        to_obstacle = obstacle_pos - self.position
        avoidance_direction = np.array([-to_obstacle[1], to_obstacle[0]])
        
        # Normalize and add some randomness
        if np.linalg.norm(avoidance_direction) > 0:
            avoidance_direction = avoidance_direction / np.linalg.norm(avoidance_direction)
            # Add small random component
            random_component = np.random.normal(0, 0.3, 2)
            avoidance_direction = avoidance_direction + random_component
            avoidance_direction = avoidance_direction / np.linalg.norm(avoidance_direction)
        
        return {
            "direction": avoidance_direction.tolist(),
            "speed": self.max_speed * 0.6
        }
    
    def _update_discovered_resources(self, resources: List[Dict]):
        """Update the list of discovered resources."""
        for resource in resources:
            resource_info = {
                "type": resource["type"],
                "position": resource["position"],
                "value": resource["value"],
                "discovered_at": self.steps_alive
            }
            
            # Check if resource is already discovered
            if not any(r["position"] == resource["position"] for r in self.discovered_resources):
                self.discovered_resources.append(resource_info)
    
    def get_exploration_metrics(self) -> Dict[str, Any]:
        """Get exploration-specific metrics."""
        return {
            "discovered_resources": len(self.discovered_resources),
            "exploration_targets_remaining": len(self.exploration_targets),
            "coverage_area": len(self.coverage_grid),
            "exploration_efficiency": self.steps_alive / max(len(self.discovered_resources), 1)
        }


class CollectorAgent(BaseAgent):
    """
    Collector agent specialized in gathering resources efficiently.
    
    Features:
    - Resource collection optimization
    - Path planning to resources
    - Energy management for collection tasks
    - Coordination with other collectors
    """
    
    def __init__(self, agent_id: str, position: Tuple[float, float], world):
        """Initialize a collector agent."""
        capabilities = {
            "mobility": True,
            "sensor_capability": True,
            "collection_capability": True,
            "communication": True,
            "collection": True
        }
        
        super().__init__(agent_id, position, "collector", world, capabilities)
        
        # Collector-specific properties
        self.collection_target = None
        self.collection_mode = "efficient"  # efficient, greedy, coordinated
        self.resource_preferences = ["energy", "mineral", "food", "water"]
        self.collection_efficiency = 1.0
        self.carrying_capacity = 5
        self.current_cargo = []
        
        # Enhanced collection capabilities
        self.collection_range = 25.0
        self.collection_speed = 1.5
        
        # Path planning
        self.path_to_target = []
        self.pathfinding_algorithm = "simple"  # simple, a_star, potential_field
        
        # Initialize collection strategies
        self._initialize_collection_strategies()
    
    def _initialize_collection_strategies(self):
        """Initialize collection strategies and preferences."""
        # Resource value weights
        self.resource_weights = {
            "energy": 1.5,
            "mineral": 1.2,
            "food": 1.0,
            "water": 0.8
        }
        
        # Collection efficiency modifiers
        self.efficiency_modifiers = {
            "energy": 1.0,
            "mineral": 0.9,
            "food": 1.1,
            "water": 1.0
        }
    
    def decide_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Decide on the next action based on observations."""
        action = {}
        
        # Check if we can collect nearby resources
        if observations["nearby_resources"] and len(self.current_cargo) < self.carrying_capacity:
            best_resource = self._select_best_resource(observations["nearby_resources"])
            if best_resource:
                action["collect"] = {
                    "position": best_resource["position"]
                }
                return action
        
        # Move towards collection target or explore for resources
        if self.collection_target:
            action["move"] = self._move_towards_target()
        else:
            action["move"] = self._explore_for_resources(observations)
        
        # Communicate resource findings
        if observations["nearby_resources"]:
            action["communicate"] = {
                "message": {
                    "type": "resource_location",
                    "resources": observations["nearby_resources"],
                    "position": observations["position"]
                }
            }
        
        return action
    
    def _select_best_resource(self, resources: List[Dict]) -> Optional[Dict]:
        """Select the best resource to collect based on preferences and efficiency."""
        if not resources:
            return None
        
        # Score resources based on value, distance, and preferences
        scored_resources = []
        for resource in resources:
            score = self._calculate_resource_score(resource)
            scored_resources.append((score, resource))
        
        # Return highest scoring resource
        scored_resources.sort(key=lambda x: x[0], reverse=True)
        return scored_resources[0][1] if scored_resources else None
    
    def _calculate_resource_score(self, resource: Dict) -> float:
        """Calculate score for a resource based on multiple factors."""
        # Base value
        score = resource["value"]
        
        # Distance penalty
        distance = resource["distance"]
        distance_penalty = distance / 100.0
        score -= distance_penalty
        
        # Resource type preference
        resource_type = resource["type"]
        if resource_type in self.resource_weights:
            score *= self.resource_weights[resource_type]
        
        # Collection efficiency
        if resource_type in self.efficiency_modifiers:
            score *= self.efficiency_modifiers[resource_type]
        
        # Cargo capacity consideration
        if len(self.current_cargo) >= self.carrying_capacity:
            score *= 0.5  # Reduce score if cargo is full
        
        return score
    
    def _move_towards_target(self) -> Dict[str, Any]:
        """Move towards the current collection target."""
        if not self.collection_target:
            return self._random_movement()
        
        target_pos = np.array(self.collection_target["position"])
        direction = target_pos - self.position
        distance = np.linalg.norm(direction)
        
        # If close to target, stop
        if distance < self.collection_range:
            return {"direction": [0, 0], "speed": 0}
        
        # Normalize direction and move
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        
        return {
            "direction": direction.tolist(),
            "speed": self.max_speed * 0.9
        }
    
    def _explore_for_resources(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Explore the environment to find resources."""
        # Simple exploration: move in a direction where we haven't seen resources recently
        if observations["nearby_resources"]:
            # Move towards closest resource
            closest = min(observations["nearby_resources"], key=lambda r: r["distance"])
            direction = np.array(closest["position"]) - self.position
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                return {
                    "direction": direction.tolist(),
                    "speed": self.max_speed * 0.8
                }
        
        # Random exploration
        return self._random_movement()
    
    def _random_movement(self) -> Dict[str, Any]:
        """Generate random movement for exploration."""
        angle = random.uniform(0, 2 * math.pi)
        direction = np.array([math.cos(angle), math.sin(angle)])
        
        return {
            "direction": direction.tolist(),
            "speed": self.max_speed * 0.6
        }
    
    def set_collection_target(self, resource_info: Dict):
        """Set a resource as the collection target."""
        self.collection_target = resource_info
        self.path_to_target = self._calculate_path_to_target(resource_info["position"])
    
    def _calculate_path_to_target(self, target_position: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Calculate path to target (simplified pathfinding)."""
        # Simple direct path for now
        return [target_position]
    
    def get_collection_metrics(self) -> Dict[str, Any]:
        """Get collection-specific metrics."""
        return {
            "resources_collected": self.resources_collected,
            "current_cargo": len(self.current_cargo),
            "carrying_capacity": self.carrying_capacity,
            "collection_efficiency": self.collection_efficiency,
            "collection_target": self.collection_target is not None
        }


class CoordinatorAgent(BaseAgent):
    """
    Coordinator agent specialized in managing swarm coordination and task allocation.
    
    Features:
    - Task assignment and management
    - Swarm coordination and communication
    - Performance monitoring and optimization
    - Strategic decision making
    """
    
    def __init__(self, agent_id: str, position: Tuple[float, float], world):
        """Initialize a coordinator agent."""
        capabilities = {
            "mobility": True,
            "sensor_capability": True,
            "collection_capability": False,
            "communication": True,
            "coordination": True
        }
        
        super().__init__(agent_id, position, "coordinator", world, capabilities)
        
        # Coordinator-specific properties
        self.coordination_mode = "centralized"  # centralized, distributed, hybrid
        self.managed_agents = []
        self.task_assignments = {}
        self.performance_metrics = {}
        self.coordination_strategies = []
        
        # Enhanced communication
        self.communication_range = 200.0
        self.broadcast_frequency = 10
        self.message_history = []
        
        # Strategic planning
        self.strategic_goals = []
        self.adaptation_threshold = 0.7
        self.optimization_interval = 50
        
        # Initialize coordination systems
        self._initialize_coordination_systems()
    
    def _initialize_coordination_systems(self):
        """Initialize coordination and management systems."""
        self.coordination_strategies = [
            "task_optimization",
            "resource_allocation",
            "swarm_formation",
            "performance_monitoring"
        ]
        
        self.strategic_goals = [
            "maximize_resource_collection",
            "minimize_energy_consumption",
            "optimize_task_completion",
            "maintain_swarm_cohesion"
        ]
    
    def decide_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Decide on the next action based on observations."""
        action = {}
        
        # Analyze current situation
        situation_analysis = self._analyze_situation(observations)
        
        # Generate coordination actions
        coordination_actions = self._generate_coordination_actions(situation_analysis)
        
        # Execute coordination
        for coord_action in coordination_actions:
            if coord_action["type"] == "task_assignment":
                action["communicate"] = {
                    "message": {
                        "type": "task_assignment",
                        "assignments": coord_action["assignments"],
                        "priority": coord_action["priority"]
                    }
                }
            elif coord_action["type"] == "formation_control":
                action["move"] = self._calculate_formation_movement(coord_action["formation"])
            elif coord_action["type"] == "performance_optimization":
                action["communicate"] = {
                    "message": {
                        "type": "performance_optimization",
                        "recommendations": coord_action["recommendations"]
                    }
                }
        
        # Move to optimal coordination position
        if not action.get("move"):
            action["move"] = self._calculate_coordination_movement(observations)
        
        return action
    
    def _analyze_situation(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the current situation for coordination decisions."""
        analysis = {
            "resource_distribution": self._analyze_resource_distribution(observations),
            "agent_positions": self._analyze_agent_positions(observations),
            "task_progress": self._analyze_task_progress(),
            "swarm_cohesion": self._calculate_swarm_cohesion(),
            "performance_metrics": self._get_performance_metrics()
        }
        
        return analysis
    
    def _analyze_resource_distribution(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource distribution in the environment."""
        nearby_resources = observations.get("nearby_resources", [])
        
        if not nearby_resources:
            return {"status": "no_resources", "density": 0.0}
        
        # Calculate resource density
        total_value = sum(r["value"] for r in nearby_resources)
        avg_distance = np.mean([r["distance"] for r in nearby_resources])
        
        return {
            "status": "resources_available",
            "count": len(nearby_resources),
            "total_value": total_value,
            "average_distance": avg_distance,
            "density": len(nearby_resources) / max(avg_distance, 1.0)
        }
    
    def _analyze_agent_positions(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze agent positions and distribution."""
        # This would need access to other agent positions
        # For now, return simplified analysis
        return {
            "swarm_center": self.position.tolist(),
            "agent_density": 1.0,
            "formation_quality": 0.8
        }
    
    def _analyze_task_progress(self) -> Dict[str, Any]:
        """Analyze progress of assigned tasks."""
        return {
            "active_tasks": len(self.task_assignments),
            "completion_rate": 0.7,  # Would calculate from actual data
            "efficiency": 0.8
        }
    
    def _calculate_swarm_cohesion(self) -> float:
        """Calculate how well the swarm is coordinated."""
        # Simplified cohesion calculation
        return 0.8
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            "coordination_efficiency": 0.85,
            "task_completion_rate": 0.75,
            "resource_utilization": 0.8,
            "energy_efficiency": 0.9
        }
    
    def _generate_coordination_actions(self, situation_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate coordination actions based on situation analysis."""
        actions = []
        
        # Task assignment optimization
        if situation_analysis["task_progress"]["efficiency"] < self.adaptation_threshold:
            actions.append({
                "type": "task_assignment",
                "assignments": self._optimize_task_assignments(),
                "priority": "high"
            })
        
        # Formation control
        if situation_analysis["swarm_cohesion"] < 0.7:
            actions.append({
                "type": "formation_control",
                "formation": "circular"
            })
        
        # Performance optimization
        if self.steps_alive % self.optimization_interval == 0:
            actions.append({
                "type": "performance_optimization",
                "recommendations": self._generate_optimization_recommendations()
            })
        
        return actions
    
    def _optimize_task_assignments(self) -> Dict[str, Any]:
        """Optimize task assignments for better efficiency."""
        # Simplified task optimization
        return {
            "strategy": "load_balancing",
            "assignments": [],
            "expected_improvement": 0.15
        }
    
    def _calculate_formation_movement(self, formation_type: str) -> Dict[str, Any]:
        """Calculate movement for formation control."""
        if formation_type == "circular":
            # Move to center of swarm
            center_position = np.array([self.world.width / 2, self.world.height / 2])
            direction = center_position - self.position
            
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            
            return {
                "direction": direction.tolist(),
                "speed": self.max_speed * 0.5
            }
        
        return {"direction": [0, 0], "speed": 0}
    
    def _calculate_coordination_movement(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate movement for optimal coordination position."""
        # Move towards area with highest resource density
        resource_analysis = observations.get("nearby_resources", [])
        
        if resource_analysis:
            # Calculate center of resources
            resource_positions = [np.array(r["position"]) for r in resource_analysis]
            resource_center = np.mean(resource_positions, axis=0)
            
            direction = resource_center - self.position
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            
            return {
                "direction": direction.tolist(),
                "speed": self.max_speed * 0.7
            }
        
        # Default movement pattern
        return self._random_movement()
    
    def _random_movement(self) -> Dict[str, Any]:
        """Generate random movement for exploration."""
        angle = random.uniform(0, 2 * math.pi)
        direction = np.array([math.cos(angle), math.sin(angle)])
        
        return {
            "direction": direction.tolist(),
            "speed": self.max_speed * 0.4
        }
    
    def _generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate performance optimization recommendations."""
        return [
            {
                "type": "energy_optimization",
                "description": "Reduce movement speed to conserve energy",
                "expected_impact": 0.1
            },
            {
                "type": "task_distribution",
                "description": "Redistribute tasks for better load balancing",
                "expected_impact": 0.15
            }
        ]
    
    def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get coordination-specific metrics."""
        return {
            "managed_agents": len(self.managed_agents),
            "active_assignments": len(self.task_assignments),
            "coordination_efficiency": self._get_performance_metrics()["coordination_efficiency"],
            "swarm_cohesion": self._calculate_swarm_cohesion(),
            "strategic_goals": len(self.strategic_goals)
        }
