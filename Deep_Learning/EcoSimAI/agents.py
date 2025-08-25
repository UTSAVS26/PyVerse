import random
import math
from typing import Optional, List, Tuple
from environment import Position, CellType

class Agent:
    """Base class for all agents in the ecosystem."""
    
    def __init__(self, agent_id: int, position: Position, energy: float = 50):
        self.id = agent_id
        self.position = position
        self.energy = energy
        self.energy_decay_rate = 1.0
        self.vision_range = 5
        self.cell_type = None
        self.age = 0
        self.max_energy = 100
        
    def act(self, environment) -> Optional[Position]:
        """Base action method - should be overridden by subclasses."""
        self.age += 1
        # Apply energy decay
        self.energy -= self.energy_decay_rate
        if self.energy < 0:
            self.energy = 0
        return self.position
    
    def get_state(self, environment) -> dict:
        """Get the current state of the agent for RL."""
        return {
            'energy': self.energy / self.max_energy,
            'age': self.age,
            'position': (self.position.x, self.position.y)
        }

class Plant(Agent):
    """Plant agent that grows and spreads."""
    
    def __init__(self, agent_id: int, position: Position, energy: float = 30):
        super().__init__(agent_id, position, energy)
        self.cell_type = CellType.PLANT
        self.energy_decay_rate = 0.1
        self.growth_rate = 0.2
        self.spread_rate = 0.02
        self.max_energy = 50
        
    def act(self, environment) -> Optional[Position]:
        """Plants grow and potentially spread."""
        super().act(environment)
        
        # Grow (increase energy)
        if self.energy < self.max_energy:
            self.energy += self.growth_rate
            
        # Try to spread to nearby empty cells
        if random.random() < self.spread_rate:
            nearby_empty = self._get_nearby_empty_positions(environment)
            if nearby_empty:
                new_pos = random.choice(nearby_empty)
                # Create a new plant
                new_plant = Plant(environment._get_next_agent_id(), new_pos, energy=10)
                environment._add_agent(new_plant, new_pos)
        
        return self.position
    
    def _get_nearby_empty_positions(self, environment) -> List[Position]:
        """Get empty positions within 2 cells of the plant."""
        empty_positions = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                new_pos = Position(self.position.x + dx, self.position.y + dy)
                if environment.is_valid_position(new_pos) and environment.is_empty(new_pos):
                    empty_positions.append(new_pos)
        return empty_positions

class Prey(Agent):
    """Prey agent that eats plants and avoids predators."""
    
    def __init__(self, agent_id: int, position: Position, energy: float = 50):
        super().__init__(agent_id, position, energy)
        self.cell_type = CellType.PREY
        self.energy_decay_rate = 1.5
        self.vision_range = 8
        self.max_energy = 80
        self.reproduction_threshold = 70
        self.reproduction_cost = 30
        
    def act(self, environment) -> Optional[Position]:
        """Prey behavior: find food, avoid predators, reproduce."""
        super().act(environment)
        
        # Check for reproduction
        if self.energy >= self.reproduction_threshold:
            self._try_reproduce(environment)
        
        # Find best action
        new_position = self._find_best_move(environment)
        
        # Try to eat if we're on a plant
        if environment.grid[self.position.y, self.position.x] == CellType.PLANT.value:
            self.eat_plant(environment, self.position)
        
        return new_position
    
    def _find_best_move(self, environment) -> Position:
        """Find the best move based on food and predator locations."""
        # Get nearby predators
        nearby_predators = environment.get_agents_in_radius(
            self.position, self.vision_range, Predator
        )
        
        # Get nearby plants
        nearby_plants = environment.get_plants_in_radius(self.position, self.vision_range)
        
        # If predators nearby, move away
        if nearby_predators:
            return self._move_away_from_predators(environment, nearby_predators)
        
        # If plants nearby and hungry, move towards plants
        if nearby_plants and self.energy < 40:
            return self._move_towards_plants(environment, nearby_plants)
        
        # Random movement
        return self._random_move(environment)
    
    def _move_away_from_predators(self, environment, predators) -> Position:
        """Move away from nearby predators."""
        # Calculate average predator position
        avg_x = sum(pos.x for pos, _ in predators) / len(predators)
        avg_y = sum(pos.y for pos, _ in predators) / len(predators)
        
        # Calculate direction away from predators
        dx = self.position.x - avg_x
        dy = self.position.y - avg_y
        
        # Normalize direction
        distance = math.sqrt(dx*dx + dy*dy)
        if distance > 0:
            dx = dx / distance
            dy = dy / distance
        
        # Try to move in that direction
        new_x = int(self.position.x + dx)
        new_y = int(self.position.y + dy)
        new_pos = Position(new_x, new_y)
        
        if environment.is_valid_position(new_pos) and environment.is_empty(new_pos):
            return new_pos
        
        return self.position
    
    def _move_towards_plants(self, environment, plants) -> Position:
        """Move towards the nearest plant."""
        if not plants:
            return self.position
        
        # Find nearest plant
        nearest_plant = min(plants, key=lambda p: self.position.distance_to(p))
        
        # Calculate direction to plant
        dx = nearest_plant.x - self.position.x
        dy = nearest_plant.y - self.position.y
        
        # Normalize direction
        distance = math.sqrt(dx*dx + dy*dy)
        if distance > 0:
            dx = dx / distance
            dy = dy / distance
        
        # Try to move towards plant
        new_x = int(self.position.x + dx)
        new_y = int(self.position.y + dy)
        new_pos = Position(new_x, new_y)
        
        if environment.is_valid_position(new_pos) and environment.is_empty(new_pos):
            return new_pos
        
        return self.position
    
    def _random_move(self, environment) -> Position:
        """Move randomly to an adjacent cell."""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), 
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            new_pos = Position(self.position.x + dx, self.position.y + dy)
            if environment.is_valid_position(new_pos) and environment.is_empty(new_pos):
                return new_pos
        
        return self.position
    
    def eat_plant(self, environment, plant_position: Position):
        """Eat a plant and gain energy."""
        if environment.grid[plant_position.y, plant_position.x] == CellType.PLANT.value:
            environment.grid[plant_position.y, plant_position.x] = CellType.EMPTY.value
            self.energy = min(self.max_energy, self.energy + 20)
    
    def _try_reproduce(self, environment):
        """Try to reproduce if conditions are met."""
        if self.energy >= self.reproduction_threshold:
            # Find empty position nearby
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    new_pos = Position(self.position.x + dx, self.position.y + dy)
                    if environment.is_valid_position(new_pos) and environment.is_empty(new_pos):
                        # Create offspring
                        offspring = Prey(environment._get_next_agent_id(), new_pos, energy=30)
                        environment._add_agent(offspring, new_pos)
                        
                        # Reduce parent energy
                        self.energy -= self.reproduction_cost
                        return
    
    def interact_with(self, other_agent, environment):
        """Handle interaction with another agent."""
        if isinstance(other_agent, Predator):
            # Prey gets eaten
            environment._remove_agent(self.id)

class Predator(Agent):
    """Predator agent that hunts prey."""
    
    def __init__(self, agent_id: int, position: Position, energy: float = 50):
        super().__init__(agent_id, position, energy)
        self.cell_type = CellType.PREDATOR
        self.energy_decay_rate = 2.0
        self.vision_range = 10
        self.max_energy = 100
        self.reproduction_threshold = 80
        self.reproduction_cost = 40
        self.hunt_success_rate = 0.7
        
    def act(self, environment) -> Optional[Position]:
        """Predator behavior: hunt prey, reproduce."""
        super().act(environment)
        
        # Check for reproduction
        if self.energy >= self.reproduction_threshold:
            self._try_reproduce(environment)
        
        # Find best action
        new_position = self._find_best_move(environment)
        
        return new_position
    
    def _find_best_move(self, environment) -> Position:
        """Find the best move based on prey locations."""
        # Get nearby prey
        nearby_prey = environment.get_agents_in_radius(
            self.position, self.vision_range, Prey
        )
        
        # If prey nearby, move towards them
        if nearby_prey:
            return self._move_towards_prey(environment, nearby_prey)
        
        # Random movement
        return self._random_move(environment)
    
    def _move_towards_prey(self, environment, prey_list) -> Position:
        """Move towards the nearest prey."""
        if not prey_list:
            return self.position
        
        # Find nearest prey
        nearest_prey_pos, nearest_prey = min(prey_list, key=lambda x: self.position.distance_to(x[0]))
        
        # Calculate direction to prey
        dx = nearest_prey_pos.x - self.position.x
        dy = nearest_prey_pos.y - self.position.y
        
        # Normalize direction
        distance = math.sqrt(dx*dx + dy*dy)
        if distance > 0:
            dx = dx / distance
            dy = dy / distance
        
        # Try to move towards prey
        new_x = int(self.position.x + dx)
        new_y = int(self.position.y + dy)
        new_pos = Position(new_x, new_y)
        
        if environment.is_valid_position(new_pos) and environment.is_empty(new_pos):
            return new_pos
        
        return self.position
    
    def _random_move(self, environment) -> Position:
        """Move randomly to an adjacent cell."""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), 
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            new_pos = Position(self.position.x + dx, self.position.y + dy)
            if environment.is_valid_position(new_pos) and environment.is_empty(new_pos):
                return new_pos
        
        return self.position
    
    def _try_reproduce(self, environment):
        """Try to reproduce if conditions are met."""
        if self.energy >= self.reproduction_threshold:
            # Find empty position nearby
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    new_pos = Position(self.position.x + dx, self.position.y + dy)
                    if environment.is_valid_position(new_pos) and environment.is_empty(new_pos):
                        # Create offspring
                        offspring = Predator(environment._get_next_agent_id(), new_pos, energy=40)
                        environment._add_agent(offspring, new_pos)
                        
                        # Reduce parent energy
                        self.energy -= self.reproduction_cost
                        return
    
    def interact_with(self, other_agent, environment):
        """Handle interaction with another agent."""
        if isinstance(other_agent, Prey):
            # Hunt the prey
            if random.random() < self.hunt_success_rate:
                # Successful hunt
                environment._remove_agent(other_agent.id)
                self.energy = min(self.max_energy, self.energy + 30)
            else:
                # Failed hunt
                self.energy -= 5
