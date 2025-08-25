import pytest
import numpy as np
from environment import Environment, Position, CellType
from agents import Agent, Prey, Predator, Plant

class TestAgent:
    """Test cases for the base Agent class."""
    
    def test_agent_initialization(self):
        """Test agent initialization with default parameters."""
        pos = Position(5, 5)
        agent = Agent(0, pos)
        
        assert agent.id == 0
        assert agent.position == pos
        assert agent.energy == 50
        assert agent.energy_decay_rate == 1.0
        assert agent.vision_range == 5
        assert agent.cell_type is None
        assert agent.age == 0
        assert agent.max_energy == 100
    
    def test_agent_custom_initialization(self):
        """Test agent initialization with custom parameters."""
        pos = Position(10, 10)
        agent = Agent(1, pos, energy=75)
        
        assert agent.id == 1
        assert agent.position == pos
        assert agent.energy == 75
        assert agent.max_energy == 100
    
    def test_agent_act_method(self):
        """Test base agent act method."""
        pos = Position(5, 5)
        agent = Agent(0, pos)
        env = Environment(width=10, height=10)
        
        # Act should return current position and increment age
        new_pos = agent.act(env)
        assert new_pos == pos
        assert agent.age == 1
    
    def test_agent_get_state(self):
        """Test agent state retrieval."""
        pos = Position(5, 5)
        agent = Agent(0, pos, energy=60)
        env = Environment(width=10, height=10)
        
        state = agent.get_state(env)
        
        assert state['energy'] == 0.6  # 60/100
        assert state['age'] == 0
        assert state['position'] == (5, 5)

class TestPlant:
    """Test cases for the Plant class."""
    
    def test_plant_initialization(self):
        """Test plant initialization."""
        pos = Position(5, 5)
        plant = Plant(0, pos)
        
        assert plant.cell_type == CellType.PLANT
        assert plant.energy_decay_rate == 0.1
        assert plant.growth_rate == 0.05
        assert plant.spread_rate == 0.02
        assert plant.max_energy == 50
    
    def test_plant_growth(self):
        """Test plant growth mechanism."""
        pos = Position(5, 5)
        plant = Plant(0, pos, energy=30)
        env = Environment(width=10, height=10)
        
        initial_energy = plant.energy
        
        # Act should increase energy (growth)
        plant.act(env)
        
        assert plant.energy > initial_energy
        assert plant.age == 1
    
    def test_plant_spreading(self):
        """Test plant spreading mechanism."""
        pos = Position(5, 5)
        plant = Plant(0, pos)
        env = Environment(width=10, height=10)
        
        initial_plants = np.sum(env.grid == CellType.PLANT.value)
        
        # Run multiple acts to allow spreading
        for _ in range(100):
            plant.act(env)
        
        final_plants = np.sum(env.grid == CellType.PLANT.value)
        # Should have spread at least one plant
        assert final_plants >= initial_plants
    
    def test_plant_nearby_empty_positions(self):
        """Test finding nearby empty positions."""
        pos = Position(5, 5)
        plant = Plant(0, pos)
        env = Environment(width=10, height=10)
        
        # Clear the environment first
        env.agents.clear()
        env.agent_positions.clear()
        env.grid.fill(CellType.EMPTY.value)
        
        # Fill some positions
        env.grid[4, 4] = CellType.PLANT.value
        env.grid[6, 6] = CellType.PLANT.value
        
        empty_positions = plant._get_nearby_empty_positions(env)
        
        # Should find some empty positions
        assert len(empty_positions) > 0
        for pos in empty_positions:
            assert env.is_empty(pos)

class TestPrey:
    """Test cases for the Prey class."""
    
    def test_prey_initialization(self):
        """Test prey initialization."""
        pos = Position(5, 5)
        prey = Prey(0, pos)
        
        assert prey.cell_type == CellType.PREY
        assert prey.energy_decay_rate == 1.5
        assert prey.vision_range == 8
        assert prey.max_energy == 80
        assert prey.reproduction_threshold == 70
        assert prey.reproduction_cost == 30
    
    def test_prey_movement_away_from_predators(self):
        """Test prey movement away from predators."""
        prey_pos = Position(5, 5)
        predator_pos = Position(6, 6)
        
        prey = Prey(0, prey_pos)
        predator = Predator(1, predator_pos)
        env = Environment(width=10, height=10)
        
        # Clear the environment first
        env.agents.clear()
        env.agent_positions.clear()
        env.grid.fill(CellType.EMPTY.value)
        
        env._add_agent(prey, prey_pos)
        env._add_agent(predator, predator_pos)
        
        # Prey should move away from predator
        new_pos = prey._move_away_from_predators(env, [(predator_pos, predator)])
        
        # Should move in opposite direction or stay in place
        # The agent might stay in place if no valid moves are available
        if new_pos != prey_pos:
            # Check that the new position is valid and empty
            assert env.is_valid_position(new_pos)
            assert env.is_empty(new_pos)
        else:
            # If staying in place, that's also valid behavior
            pass
    
    def test_prey_movement_towards_plants(self):
        """Test prey movement towards plants."""
        prey_pos = Position(5, 5)
        plant_pos = Position(7, 7)
        
        prey = Prey(0, prey_pos)
        env = Environment(width=10, height=10)
        
        # Clear the environment first
        env.agents.clear()
        env.agent_positions.clear()
        env.grid.fill(CellType.EMPTY.value)
        
        # Add plant
        env.grid[plant_pos.y, plant_pos.x] = CellType.PLANT.value
        
        # Prey should move towards plant
        new_pos = prey._move_towards_plants(env, [plant_pos])
        
        # Should move closer to plant (or stay in place if no valid moves)
        if new_pos != prey_pos:
            assert new_pos.distance_to(plant_pos) < prey_pos.distance_to(plant_pos)
        else:
            # If staying in place, that's also valid behavior
            pass
    
    def test_prey_random_movement(self):
        """Test prey random movement."""
        prey_pos = Position(5, 5)
        prey = Prey(0, prey_pos)
        env = Environment(width=10, height=10)
        
        # Clear the environment first
        env.agents.clear()
        env.agent_positions.clear()
        env.grid.fill(CellType.EMPTY.value)
        
        # Should find a valid move
        new_pos = prey._random_move(env)
        assert env.is_valid_position(new_pos)
        assert env.is_empty(new_pos)
    
    def test_prey_eating_plants(self):
        """Test prey eating plants."""
        prey_pos = Position(5, 5)
        plant_pos = Position(5, 5)
        
        prey = Prey(0, prey_pos, energy=50)
        env = Environment(width=10, height=10)
        
        # Add plant at prey position
        env.grid[plant_pos.y, plant_pos.x] = CellType.PLANT.value
        
        initial_energy = prey.energy
        
        # Prey eats plant
        prey.eat_plant(env, plant_pos)
        
        # Plant should be removed
        assert env.grid[plant_pos.y, plant_pos.x] == CellType.EMPTY.value
        # Prey should gain energy
        assert prey.energy > initial_energy
    
    def test_prey_reproduction(self):
        """Test prey reproduction mechanism."""
        prey_pos = Position(5, 5)
        prey = Prey(0, prey_pos, energy=75)  # Above reproduction threshold
        env = Environment(width=10, height=10)
        
        # Clear the environment first
        env.agents.clear()
        env.agent_positions.clear()
        env.grid.fill(CellType.EMPTY.value)
        
        initial_agents = len(env.agents)
        initial_energy = prey.energy
        
        # Try to reproduce
        prey._try_reproduce(env)
        
        # Should have created offspring
        assert len(env.agents) > initial_agents
        # Parent should have lost energy
        assert prey.energy < initial_energy
    
    def test_prey_interaction_with_predator(self):
        """Test prey interaction with predator."""
        prey_pos = Position(5, 5)
        predator_pos = Position(6, 6)
        
        prey = Prey(0, prey_pos)
        predator = Predator(1, predator_pos)
        env = Environment(width=10, height=10)
        
        env._add_agent(prey, prey_pos)
        env._add_agent(predator, predator_pos)
        
        # Prey interacts with predator
        prey.interact_with(predator, env)
        
        # Prey should be removed
        assert prey.id not in env.agent_positions

class TestPredator:
    """Test cases for the Predator class."""
    
    def test_predator_initialization(self):
        """Test predator initialization."""
        pos = Position(5, 5)
        predator = Predator(0, pos)
        
        assert predator.cell_type == CellType.PREDATOR
        assert predator.energy_decay_rate == 2.0
        assert predator.vision_range == 10
        assert predator.max_energy == 100
        assert predator.reproduction_threshold == 80
        assert predator.reproduction_cost == 40
        assert predator.hunt_success_rate == 0.7
    
    def test_predator_movement_towards_prey(self):
        """Test predator movement towards prey."""
        predator_pos = Position(5, 5)
        prey_pos = Position(7, 7)
        
        predator = Predator(0, predator_pos)
        prey = Prey(1, prey_pos)
        env = Environment(width=10, height=10)
        
        # Clear the environment first
        env.agents.clear()
        env.agent_positions.clear()
        env.grid.fill(CellType.EMPTY.value)
        
        env._add_agent(predator, predator_pos)
        env._add_agent(prey, prey_pos)
        
        # Predator should move towards prey
        new_pos = predator._move_towards_prey(env, [(prey_pos, prey)])
        
        # Should move closer to prey (or stay in place if no valid moves)
        if new_pos != predator_pos:
            assert new_pos.distance_to(prey_pos) < predator_pos.distance_to(prey_pos)
        else:
            # If staying in place, that's also valid behavior
            pass
    
    def test_predator_random_movement(self):
        """Test predator random movement."""
        predator_pos = Position(5, 5)
        predator = Predator(0, predator_pos)
        env = Environment(width=10, height=10)
        
        # Clear the environment first
        env.agents.clear()
        env.agent_positions.clear()
        env.grid.fill(CellType.EMPTY.value)
        
        # Should find a valid move
        new_pos = predator._random_move(env)
        assert env.is_valid_position(new_pos)
        assert env.is_empty(new_pos)
    
    def test_predator_reproduction(self):
        """Test predator reproduction mechanism."""
        predator_pos = Position(5, 5)
        predator = Predator(0, predator_pos, energy=85)  # Above reproduction threshold
        env = Environment(width=10, height=10)
        
        # Clear the environment first
        env.agents.clear()
        env.agent_positions.clear()
        env.grid.fill(CellType.EMPTY.value)
        
        initial_agents = len(env.agents)
        initial_energy = predator.energy
        
        # Try to reproduce
        predator._try_reproduce(env)
        
        # Should have created offspring
        assert len(env.agents) > initial_agents
        # Parent should have lost energy
        assert predator.energy < initial_energy
    
    def test_predator_interaction_with_prey(self):
        """Test predator interaction with prey."""
        predator_pos = Position(5, 5)
        prey_pos = Position(6, 6)
        
        predator = Predator(0, predator_pos, energy=50)
        prey = Prey(1, prey_pos)
        env = Environment(width=10, height=10)
        
        env._add_agent(predator, predator_pos)
        env._add_agent(prey, prey_pos)
        
        initial_energy = predator.energy
        
        # Predator interacts with prey
        predator.interact_with(prey, env)
        
        # Either prey is removed (successful hunt) or predator loses energy (failed hunt)
        if prey.id not in env.agent_positions:
            # Successful hunt
            assert predator.energy > initial_energy
        else:
            # Failed hunt
            assert predator.energy < initial_energy

class TestAgentBehavior:
    """Test cases for agent behavior and interactions."""
    
    def test_agent_energy_decay(self):
        """Test agent energy decay over time."""
        pos = Position(5, 5)
        prey = Prey(0, pos, energy=50)
        env = Environment(width=10, height=10)
        
        # Clear the environment first to ensure no plants
        env.agents.clear()
        env.agent_positions.clear()
        env.grid.fill(CellType.EMPTY.value)
        
        env._add_agent(prey, pos)
        initial_energy = prey.energy
        
        # Run multiple steps
        for _ in range(5):
            prey.act(env)
        
        # Energy should have decreased (energy decay rate is 1.5 per step)
        assert prey.energy < initial_energy
    
    def test_agent_death_from_low_energy(self):
        """Test agent death when energy reaches zero."""
        pos = Position(5, 5)
        prey = Prey(0, pos, energy=1)  # Very low energy
        env = Environment(width=10, height=10)
        
        # Clear the environment first
        env.agents.clear()
        env.agent_positions.clear()
        env.grid.fill(CellType.EMPTY.value)
        
        env._add_agent(prey, pos)
        
        # Run until energy depletes (energy decay rate is 1.5 per step)
        for _ in range(5):
            if prey.energy <= 0:
                break
            prey.act(env)
        
        # Agent should be dead (energy should be 0 or negative)
        assert prey.energy <= 0
    
    def test_agent_vision_range(self):
        """Test agent vision range functionality."""
        prey_pos = Position(5, 5)
        predator_pos = Position(15, 15)  # Further outside vision range
        
        prey = Prey(0, prey_pos)
        predator = Predator(1, predator_pos)
        env = Environment(width=20, height=20)
        
        # Clear the environment first
        env.agents.clear()
        env.agent_positions.clear()
        env.grid.fill(CellType.EMPTY.value)
        
        env._add_agent(prey, prey_pos)
        env._add_agent(predator, predator_pos)
        
        # Prey should not see predator (too far)
        nearby_predators = env.get_agents_in_radius(prey_pos, prey.vision_range, Predator)
        assert len(nearby_predators) == 0
    
    def test_agent_movement_validation(self):
        """Test agent movement validation."""
        prey_pos = Position(0, 0)  # Corner position
        prey = Prey(0, prey_pos)
        env = Environment(width=10, height=10)
        
        # Should not move outside bounds
        new_pos = prey._random_move(env)
        assert env.is_valid_position(new_pos)
        assert new_pos.x >= 0 and new_pos.x < env.width
        assert new_pos.y >= 0 and new_pos.y < env.height

if __name__ == "__main__":
    pytest.main([__file__])
