"""
World representation for SwarmMindAI simulation environment.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import random


@dataclass
class Obstacle:
    """Represents an obstacle in the world."""
    x: float
    y: float
    radius: float
    obstacle_type: str = "static"


@dataclass
class Resource:
    """Represents a resource that agents can collect."""
    x: float
    y: float
    resource_type: str
    value: float
    quantity: int
    collected: bool = False


class World:
    """
    Represents the 2D world where the swarm operates.
    
    Features:
    - Dynamic obstacle generation
    - Resource distribution
    - Spatial partitioning for efficient queries
    - Collision detection
    """
    
    def __init__(self, width: int, height: int, seed: Optional[int] = None):
        """
        Initialize the world.
        
        Args:
            width: World width in pixels
            height: World height in pixels
            seed: Random seed for reproducibility
        """
        self.width = width
        self.height = height
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # World elements
        self.obstacles: List[Obstacle] = []
        self.resources: List[Resource] = []
        self.dynamic_elements: List[Dict] = []
        
        # Spatial partitioning for performance
        self.grid_size = 50
        self.spatial_grid: Dict[Tuple[int, int], List] = {}
        
        # World state
        self.time_step = 0
        self.weather_conditions = "clear"
        
        self._initialize_world()
    
    def _initialize_world(self):
        """Initialize the world with obstacles and resources."""
        # Generate random obstacles
        num_obstacles = random.randint(10, 20)
        for _ in range(num_obstacles):
            x = random.uniform(50, self.width - 50)
            y = random.uniform(50, self.height - 50)
            radius = random.uniform(20, 60)
            obstacle = Obstacle(x, y, radius)
            self.obstacles.append(obstacle)
        
        # Generate resources
        num_resources = random.randint(15, 30)
        resource_types = ["food", "mineral", "energy", "water"]
        for _ in range(num_resources):
            x = random.uniform(50, self.width - 50)
            y = random.uniform(50, self.height - 50)
            resource_type = random.choice(resource_types)
            value = random.uniform(1.0, 10.0)
            quantity = random.randint(1, 5)
            resource = Resource(x, y, resource_type, value, quantity)
            self.resources.append(resource)
        
        self._update_spatial_grid()
    
    def _update_spatial_grid(self):
        """Update the spatial partitioning grid."""
        self.spatial_grid.clear()
        
        # Add obstacles to grid
        for obstacle in self.obstacles:
            grid_x = int(obstacle.x // self.grid_size)
            grid_y = int(obstacle.y // self.grid_size)
            if (grid_x, grid_y) not in self.spatial_grid:
                self.spatial_grid[(grid_x, grid_y)] = []
            self.spatial_grid[(grid_x, grid_y)].append(("obstacle", obstacle))
        
        # Add resources to grid
        for resource in self.resources:
            if not resource.collected:
                grid_x = int(resource.x // self.grid_size)
                grid_y = int(resource.y // self.grid_size)
                if (grid_x, grid_y) not in self.spatial_grid:
                    self.spatial_grid[(grid_x, grid_y)] = []
                self.spatial_grid[(grid_x, grid_y)].append(("resource", resource))
    
    def check_collision(self, x: float, y: float, radius: float) -> bool:
        """
        Check if a position collides with any obstacle.
        
        Args:
            x: X coordinate
            y: Y coordinate
            radius: Radius of the object to check
            
        Returns:
            True if collision detected, False otherwise
        """
        for obstacle in self.obstacles:
            distance = np.sqrt((x - obstacle.x)**2 + (y - obstacle.y)**2)
            if distance < (radius + obstacle.radius):
                return True
        return False
    
    def get_nearby_resources(self, x: float, y: float, radius: float) -> List[Resource]:
        """
        Get resources within a certain radius of a position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            radius: Search radius
            
        Returns:
            List of nearby resources
        """
        nearby = []
        for resource in self.resources:
            if not resource.collected:
                distance = np.sqrt((x - resource.x)**2 + (y - resource.y)**2)
                if distance <= radius:
                    nearby.append(resource)
        return nearby
    
    def get_nearby_obstacles(self, x: float, y: float, radius: float) -> List[Obstacle]:
        """
        Get obstacles within a certain radius of a position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            radius: Search radius
            
        Returns:
            List of nearby obstacles
        """
        nearby = []
        for obstacle in self.obstacles:
            distance = np.sqrt((x - obstacle.x)**2 + (y - obstacle.y)**2)
            if distance <= radius:
                nearby.append(obstacle)
        return nearby
    
    def collect_resource(self, x: float, y: float, radius: float) -> Optional[Resource]:
        """
        Attempt to collect a resource at a position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            radius: Collection radius
            
        Returns:
            Collected resource or None if no resource found
        """
        for resource in self.resources:
            if not resource.collected:
                distance = np.sqrt((x - resource.x)**2 + (y - resource.y)**2)
                if distance <= radius:
                    resource.collected = True
                    self._update_spatial_grid()
                    return resource
        return None
    
    def add_dynamic_obstacle(self, x: float, y: float, radius: float, 
                            velocity: Tuple[float, float], lifetime: int):
        """
        Add a dynamic obstacle to the world.
        
        Args:
            x: Initial X coordinate
            y: Initial Y coordinate
            radius: Obstacle radius
            velocity: (dx, dy) velocity vector
            lifetime: Number of time steps the obstacle will exist
        """
        obstacle = Obstacle(x, y, radius, "dynamic")
        self.dynamic_elements.append({
            "obstacle": obstacle,
            "velocity": velocity,
            "lifetime": lifetime,
            "age": 0
        })
    
    def update(self):
        """Update the world state for the next time step."""
        self.time_step += 1
        
        # Update dynamic elements
        elements_to_remove = []
        for element in self.dynamic_elements:
            element["age"] += 1
            if element["age"] >= element["lifetime"]:
                elements_to_remove.append(element)
            else:
                # Move dynamic obstacle
                obstacle = element["obstacle"]
                dx, dy = element["velocity"]
                obstacle.x += dx
                obstacle.y += dy
                
                # Bounce off boundaries
                if obstacle.x <= obstacle.radius or obstacle.x >= self.width - obstacle.radius:
                    element["velocity"] = (-dx, dy)
                if obstacle.y <= obstacle.radius or obstacle.y >= self.height - obstacle.radius:
                    element["velocity"] = (dx, -dy)
        
        # Remove expired elements
        for element in elements_to_remove:
            self.dynamic_elements.remove(element)
        
        # Randomly add new dynamic obstacles
        if random.random() < 0.01:  # 1% chance per time step
            x = random.uniform(50, self.width - 50)
            y = random.uniform(50, self.height - 50)
            radius = random.uniform(15, 30)
            velocity = (random.uniform(-2, 2), random.uniform(-2, 2))
            lifetime = random.randint(50, 200)
            self.add_dynamic_obstacle(x, y, radius, velocity, lifetime)
        
        # Update spatial grid
        self._update_spatial_grid()
    
    def get_world_state(self) -> Dict:
        """Get the current state of the world."""
        return {
            "width": self.width,
            "height": self.height,
            "time_step": self.time_step,
            "num_obstacles": len(self.obstacles),
            "num_resources": len([r for r in self.resources if not r.collected]),
            "num_dynamic_elements": len(self.dynamic_elements),
            "weather_conditions": self.weather_conditions
        }
    
    def reset(self):
        """Reset the world to initial state."""
        self.obstacles.clear()
        self.resources.clear()
        self.dynamic_elements.clear()
        self.spatial_grid.clear()
        self.time_step = 0
        self._initialize_world()
