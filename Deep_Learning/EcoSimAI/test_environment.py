import pytest
import numpy as np
from environment import Environment, Position, CellType
from agents import Prey, Predator, Plant

class TestEnvironment:
    """Test cases for the Environment class."""
    
    def test_environment_initialization(self):
        """Test environment initialization with default parameters."""
        env = Environment()
        
        assert env.width == 50
        assert env.height == 50
        assert env.plant_growth_rate == 0.1
        assert env.step_count == 0
        # Note: next_agent_id will be > 0 because environment initializes with agents
        assert env.next_agent_id > 0
        assert env.grid.shape == (50, 50)
        assert isinstance(env.agents, dict)
        assert isinstance(env.agent_positions, dict)
    
    def test_environment_custom_initialization(self):
        """Test environment initialization with custom parameters."""
        env = Environment(width=30, height=40, plant_growth_rate=0.2)
        
        assert env.width == 30
        assert env.height == 40
        assert env.plant_growth_rate == 0.2
        assert env.grid.shape == (40, 30)
    
    def test_position_creation_and_operations(self):
        """Test Position class functionality."""
        pos1 = Position(5, 10)
        pos2 = Position(8, 12)
        
        assert pos1.x == 5
        assert pos1.y == 10
        # Distance should be sqrt((8-5)² + (12-10)²) = sqrt(9 + 4) = sqrt(13) ≈ 3.61
        assert pos1.distance_to(pos2) == pytest.approx(3.605551275463989, rel=1e-6)
        assert pos1 != pos2
        assert hash(pos1) != hash(pos2)
    
    def test_get_random_empty_position(self):
        """Test getting random empty positions."""
        env = Environment(width=10, height=10)
        
        # Clear the environment first
        env.agents.clear()
        env.agent_positions.clear()
        env.grid.fill(CellType.EMPTY.value)
        
        # Fill most of the grid
        for y in range(8):
            for x in range(8):
                env.grid[y, x] = CellType.PLANT.value
        
        # Should still find empty positions
        empty_pos = env._get_random_empty_position()
        assert empty_pos is not None
        assert env.is_empty(empty_pos)
    
    def test_position_validation(self):
        """Test position validation methods."""
        env = Environment(width=10, height=10)
        
        # Valid positions
        assert env.is_valid_position(Position(0, 0))
        assert env.is_valid_position(Position(9, 9))
        
        # Invalid positions
        assert not env.is_valid_position(Position(-1, 0))
        assert not env.is_valid_position(Position(0, -1))
        assert not env.is_valid_position(Position(10, 0))
        assert not env.is_valid_position(Position(0, 10))
    
    def test_empty_position_check(self):
        """Test checking if positions are empty."""
        env = Environment(width=10, height=10)
        
        # Clear the environment first
        env.agents.clear()
        env.agent_positions.clear()
        env.grid.fill(CellType.EMPTY.value)
        
        # Initially all positions should be empty
        assert env.is_empty(Position(5, 5))
        
        # Add a plant
        env.grid[5, 5] = CellType.PLANT.value
        assert not env.is_empty(Position(5, 5))
    
    def test_agent_management(self):
        """Test adding and removing agents."""
        env = Environment(width=10, height=10)
        
        # Add agent
        pos = Position(5, 5)
        prey = Prey(0, pos, energy=50)
        env._add_agent(prey, pos)
        
        assert pos in env.agents
        assert 0 in env.agent_positions
        assert env.agents[pos] == prey
        assert env.agent_positions[0] == pos
        assert env.grid[pos.y, pos.x] == CellType.PREY.value
        
        # Remove agent
        env._remove_agent(0)
        assert pos not in env.agents
        assert 0 not in env.agent_positions
        assert env.grid[pos.y, pos.x] == CellType.EMPTY.value
    
    def test_agent_movement(self):
        """Test agent movement functionality."""
        env = Environment(width=10, height=10)
        
        # Add agent
        old_pos = Position(5, 5)
        new_pos = Position(6, 6)
        prey = Prey(0, old_pos, energy=50)
        env._add_agent(prey, old_pos)
        
        # Move agent
        env._move_agent(0, new_pos)
        
        assert old_pos not in env.agents
        assert new_pos in env.agents
        assert env.agent_positions[0] == new_pos
        assert env.grid[old_pos.y, old_pos.x] == CellType.EMPTY.value
        assert env.grid[new_pos.y, new_pos.x] == CellType.PREY.value
    
    def test_get_agent_at(self):
        """Test getting agent at specific position."""
        env = Environment(width=10, height=10)
        
        pos = Position(5, 5)
        prey = Prey(0, pos, energy=50)
        env._add_agent(prey, pos)
        
        assert env.get_agent_at(pos) == prey
        assert env.get_agent_at(Position(6, 6)) is None
    
    def test_get_agents_in_radius(self):
        """Test getting agents within a radius."""
        env = Environment(width=10, height=10)
        
        # Add agents at different positions
        prey1 = Prey(0, Position(5, 5), energy=50)
        prey2 = Prey(1, Position(6, 6), energy=50)
        predator = Predator(2, Position(8, 8), energy=50)
        
        env._add_agent(prey1, Position(5, 5))
        env._add_agent(prey2, Position(6, 6))
        env._add_agent(predator, Position(8, 8))
        
        # Get prey within radius of (5, 5)
        nearby_prey = env.get_agents_in_radius(Position(5, 5), 2, Prey)
        assert len(nearby_prey) == 2  # Both prey should be within radius
        
        # Get predators within radius
        nearby_predators = env.get_agents_in_radius(Position(5, 5), 2, Predator)
        assert len(nearby_predators) == 0  # Predator is too far
    
    def test_get_plants_in_radius(self):
        """Test getting plants within a radius."""
        env = Environment(width=10, height=10)
        
        # Clear the environment first
        env.agents.clear()
        env.agent_positions.clear()
        env.grid.fill(CellType.EMPTY.value)
        
        # Add plants
        env.grid[5, 5] = CellType.PLANT.value
        env.grid[6, 6] = CellType.PLANT.value
        env.grid[8, 8] = CellType.PLANT.value
        
        # Get plants within radius of (5, 5)
        nearby_plants = env.get_plants_in_radius(Position(5, 5), 2)
        # Should find plants at (5,5) and (6,6) within radius 2
        assert len(nearby_plants) >= 2  # At least two plants within radius
    
    def test_plant_growth(self):
        """Test plant growth mechanism."""
        env = Environment(width=10, height=10, plant_growth_rate=0.5)
        
        initial_plants = np.sum(env.grid == CellType.PLANT.value)
        
        # Run multiple steps to allow plant growth
        for _ in range(10):
            env._grow_plants()
        
        final_plants = np.sum(env.grid == CellType.PLANT.value)
        assert final_plants >= initial_plants  # Should have grown some plants
    
    def test_environment_step(self):
        """Test environment step execution."""
        env = Environment(width=10, height=10)
        
        initial_step = env.step_count
        
        # Add some agents
        prey = Prey(0, Position(5, 5), energy=50)
        predator = Predator(1, Position(6, 6), energy=50)
        env._add_agent(prey, Position(5, 5))
        env._add_agent(predator, Position(6, 6))
        
        # Run a step
        env.step()
        
        assert env.step_count == initial_step + 1
        # Agents should have aged and lost energy
        assert prey.age > 0
        assert predator.age > 0
    
    def test_collision_handling(self):
        """Test collision handling between agents."""
        env = Environment(width=10, height=10)
        
        # Place prey and predator adjacent
        prey = Prey(0, Position(5, 5), energy=50)
        predator = Predator(1, Position(6, 6), energy=50)
        env._add_agent(prey, Position(5, 5))
        env._add_agent(predator, Position(6, 6))
        
        # Simulate collision by having predator move to prey's position
        env._handle_collision(predator, Position(6, 6), Position(5, 5))
        
        # Prey should be removed due to interaction
        assert 0 not in env.agent_positions
    
    def test_plant_eating(self):
        """Test plant eating mechanism."""
        env = Environment(width=10, height=10)
        
        # Add a plant
        plant_pos = Position(5, 5)
        env.grid[plant_pos.y, plant_pos.x] = CellType.PLANT.value
        
        # Add prey
        prey = Prey(0, Position(4, 5), energy=50)
        env._add_agent(prey, Position(4, 5))
        
        # Simulate prey eating plant
        prey.eat_plant(env, plant_pos)
        
        # Plant should be removed
        assert env.grid[plant_pos.y, plant_pos.x] == CellType.EMPTY.value
        # Prey should gain energy
        assert prey.energy > 50
    
    def test_statistics_tracking(self):
        """Test statistics collection."""
        env = Environment(width=10, height=10)
        
        # Clear the environment first
        env.agents.clear()
        env.agent_positions.clear()
        env.grid.fill(CellType.EMPTY.value)
        
        # Add agents
        prey = Prey(0, Position(5, 5), energy=50)
        predator = Predator(1, Position(6, 6), energy=50)
        env._add_agent(prey, Position(5, 5))
        env._add_agent(predator, Position(6, 6))
        
        # Add plants
        env.grid[3, 3] = CellType.PLANT.value
        env.grid[4, 4] = CellType.PLANT.value
        
        stats = env.get_statistics()
        
        assert stats['step'] == 0
        assert stats['plants'] == 2
        assert stats['prey'] == 1
        assert stats['predators'] == 1
        assert stats['total_agents'] == 2
    
    def test_grid_state_copy(self):
        """Test grid state copying."""
        env = Environment(width=10, height=10)
        
        # Modify grid
        env.grid[5, 5] = CellType.PLANT.value
        
        # Get copy
        grid_copy = env.get_grid_state()
        
        # Modify original
        env.grid[5, 5] = CellType.EMPTY.value
        
        # Copy should be unchanged
        assert grid_copy[5, 5] == CellType.PLANT.value
        assert env.grid[5, 5] == CellType.EMPTY.value
    
    def test_agent_id_generation(self):
        """Test agent ID generation."""
        env = Environment()
        
        # Get current next_agent_id
        current_id = env.next_agent_id
        
        id1 = env._get_next_agent_id()
        id2 = env._get_next_agent_id()
        id3 = env._get_next_agent_id()
        
        assert id1 == current_id
        assert id2 == current_id + 1
        assert id3 == current_id + 2
        assert env.next_agent_id == current_id + 3

if __name__ == "__main__":
    pytest.main([__file__])
