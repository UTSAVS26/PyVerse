import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class CellType(Enum):
    EMPTY = 0
    PLANT = 1
    PREY = 2
    PREDATOR = 3

@dataclass
class Position:
    x: int
    y: int
    
    def distance_to(self, other: 'Position') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __eq__(self, other):
        if not isinstance(other, Position):
            return False
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))

class Environment:
    def __init__(self, width: int = 50, height: int = 50, 
                 plant_growth_rate: float = 0.1,
                 initial_plants: int = 100,
                 initial_prey: int = 50,
                 initial_predators: int = 20):
        self.width = width
        self.height = height
        self.plant_growth_rate = plant_growth_rate
        self.grid = np.zeros((height, width), dtype=int)
        self.agents = {}  # position -> agent
        self.agent_positions = {}  # agent_id -> position
        self.next_agent_id = 0
        self.step_count = 0
        
        # Initialize environment
        self._initialize_environment(initial_plants, initial_prey, initial_predators)
    
    def _initialize_environment(self, initial_plants: int, initial_prey: int, initial_predators: int):
        """Initialize the environment with initial agents and plants."""
        # Add initial plants
        for _ in range(initial_plants):
            pos = self._get_random_empty_position()
            if pos:
                self.grid[pos.y, pos.x] = CellType.PLANT.value
        
        # Add initial prey
        for _ in range(initial_prey):
            pos = self._get_random_empty_position()
            if pos:
                from agents import Prey
                prey = Prey(self._get_next_agent_id(), pos, energy=50)
                self._add_agent(prey, pos)
        
        # Add initial predators
        for _ in range(initial_predators):
            pos = self._get_random_empty_position()
            if pos:
                from agents import Predator
                predator = Predator(self._get_next_agent_id(), pos, energy=50)
                self._add_agent(predator, pos)
    
    def _get_next_agent_id(self) -> int:
        """Get the next available agent ID."""
        agent_id = self.next_agent_id
        self.next_agent_id += 1
        return agent_id
    
    def _get_random_empty_position(self) -> Optional[Position]:
        """Get a random empty position in the grid."""
        empty_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == CellType.EMPTY.value:
                    empty_positions.append(Position(x, y))
        
        if empty_positions:
            return random.choice(empty_positions)
        return None
    
    def _add_agent(self, agent, position: Position):
        """Add an agent to the environment."""
        self.agents[position] = agent
        self.agent_positions[agent.id] = position
        self.grid[position.y, position.x] = agent.cell_type.value
    
    def _remove_agent(self, agent_id: int):
        """Remove an agent from the environment."""
        if agent_id in self.agent_positions:
            pos = self.agent_positions[agent_id]
            if pos in self.agents:
                del self.agents[pos]
            del self.agent_positions[agent_id]
            self.grid[pos.y, pos.x] = CellType.EMPTY.value
    
    def _move_agent(self, agent_id: int, new_position: Position):
        """Move an agent to a new position."""
        if agent_id in self.agent_positions:
            old_pos = self.agent_positions[agent_id]
            agent = self.agents[old_pos]
            
            # Update grid
            self.grid[old_pos.y, old_pos.x] = CellType.EMPTY.value
            self.grid[new_position.y, new_position.x] = agent.cell_type.value
            
            # Update agent tracking
            del self.agents[old_pos]
            self.agents[new_position] = agent
            self.agent_positions[agent_id] = new_position
    
    def get_agent_at(self, position: Position):
        """Get agent at a specific position."""
        return self.agents.get(position)
    
    def get_agents_in_radius(self, center: Position, radius: int, agent_type=None):
        """Get all agents within a certain radius of a position."""
        agents_in_radius = []
        for pos, agent in self.agents.items():
            if center.distance_to(pos) <= radius:
                if agent_type is None or isinstance(agent, agent_type):
                    agents_in_radius.append((pos, agent))
        return agents_in_radius
    
    def get_plants_in_radius(self, center: Position, radius: int):
        """Get all plant positions within a certain radius."""
        plants = []
        for y in range(max(0, center.y - radius), min(self.height, center.y + radius + 1)):
            for x in range(max(0, center.x - radius), min(self.width, center.x + radius + 1)):
                if self.grid[y, x] == CellType.PLANT.value:
                    plant_pos = Position(x, y)
                    if center.distance_to(plant_pos) <= radius:
                        plants.append(plant_pos)
        return plants
    
    def is_valid_position(self, position: Position) -> bool:
        """Check if a position is within the grid bounds."""
        return 0 <= position.x < self.width and 0 <= position.y < self.height
    
    def is_empty(self, position: Position) -> bool:
        """Check if a position is empty."""
        if not self.is_valid_position(position):
            return False
        return self.grid[position.y, position.x] == CellType.EMPTY.value
    
    def step(self):
        """Execute one simulation step."""
        self.step_count += 1
        
        # Grow plants
        self._grow_plants()
        
        # Get all agents for processing
        agents_to_process = list(self.agents.items())
        
        # Process each agent
        for position, agent in agents_to_process:
            if agent.id not in self.agent_positions:  # Agent was removed
                continue
            
            # Agent takes action
            new_position = agent.act(self)
            
            # Handle movement and interactions
            if new_position and new_position != position:
                if self.is_empty(new_position):
                    self._move_agent(agent.id, new_position)
                else:
                    # Handle collision
                    self._handle_collision(agent, position, new_position)
            
            # Check if agent should die
            if agent.energy <= 0:
                self._remove_agent(agent.id)
            else:
                # Reduce energy
                agent.energy -= agent.energy_decay_rate
    
    def _grow_plants(self):
        """Grow new plants based on growth rate."""
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == CellType.EMPTY.value:
                    if random.random() < self.plant_growth_rate:
                        self.grid[y, x] = CellType.PLANT.value
    
    def _handle_collision(self, agent, current_pos: Position, target_pos: Position):
        """Handle collision between agent and target position."""
        target_agent = self.get_agent_at(target_pos)
        
        if target_agent:
            # Agent interaction
            if hasattr(agent, 'interact_with') and hasattr(target_agent, 'interact_with'):
                agent.interact_with(target_agent, self)
                target_agent.interact_with(agent, self)
        elif self.grid[target_pos.y, target_pos.x] == CellType.PLANT.value:
            # Plant interaction
            if hasattr(agent, 'eat_plant'):
                agent.eat_plant(self, target_pos)
    
    def get_statistics(self) -> Dict:
        """Get current ecosystem statistics."""
        stats = {
            'step': self.step_count,
            'plants': np.sum(self.grid == CellType.PLANT.value),
            'prey': 0,
            'predators': 0,
            'total_agents': len(self.agents)
        }
        
        for agent in self.agents.values():
            if hasattr(agent, 'cell_type'):
                if agent.cell_type == CellType.PREY:
                    stats['prey'] += 1
                elif agent.cell_type == CellType.PREDATOR:
                    stats['predators'] += 1
        
        return stats
    
    def get_grid_state(self) -> np.ndarray:
        """Get a copy of the current grid state."""
        return self.grid.copy()
