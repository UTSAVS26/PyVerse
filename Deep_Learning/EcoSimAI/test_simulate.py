import pytest
import numpy as np
import tempfile
import os
from environment import Environment, CellType
from agents import Prey, Predator
from rl_agent import RLPrey, RLPredator
from simulate import (
    SimulationVisualizer, 
    create_environment_with_rl_agents,
    run_simulation,
    run_headless_simulation,
    save_simulation_stats
)

class TestSimulationVisualizer:
    """Test cases for the SimulationVisualizer class."""
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        env = Environment(width=10, height=10)
        visualizer = SimulationVisualizer(env, cell_size=5)
        
        assert visualizer.environment == env
        assert visualizer.cell_size == 5
        assert visualizer.use_pygame == True
        assert visualizer.show_stats == True
        assert len(visualizer.colors) == 4  # All cell types
        assert len(visualizer.stats_history) == 4  # All stat types
    
    def test_visualizer_colors(self):
        """Test color mapping."""
        env = Environment(width=10, height=10)
        visualizer = SimulationVisualizer(env)
        
        # Check that all cell types have colors
        assert CellType.EMPTY.value in visualizer.colors
        assert CellType.PLANT.value in visualizer.colors
        assert CellType.PREY.value in visualizer.colors
        assert CellType.PREDATOR.value in visualizer.colors
        
        # Check color format (RGB tuples)
        for color in visualizer.colors.values():
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)
    
    def test_visualizer_pygame_initialization(self):
        """Test pygame initialization."""
        env = Environment(width=10, height=10)
        visualizer = SimulationVisualizer(env, use_pygame=True)
        
        # Should have pygame attributes
        assert hasattr(visualizer, 'screen')
        assert hasattr(visualizer, 'font')
        assert visualizer.width == 10 * visualizer.cell_size
        assert visualizer.height == 10 * visualizer.cell_size + 100  # Extra space for stats
    
    def test_visualizer_matplotlib_initialization(self):
        """Test matplotlib initialization."""
        env = Environment(width=10, height=10)
        visualizer = SimulationVisualizer(env, use_pygame=False)
        
        # Should have matplotlib attributes
        assert hasattr(visualizer, 'fig')
        assert hasattr(visualizer, 'ax1')
        assert hasattr(visualizer, 'ax2')
    
    def test_visualizer_update_pygame(self):
        """Test pygame update method."""
        env = Environment(width=5, height=5)
        visualizer = SimulationVisualizer(env, use_pygame=True)
        
        # Add some agents
        prey = Prey(0, env._get_random_empty_position(), energy=50)
        predator = Predator(1, env._get_random_empty_position(), energy=50)
        env._add_agent(prey, env._get_random_empty_position())
        env._add_agent(predator, env._get_random_empty_position())
        
        # Update should return True (continue simulation)
        result = visualizer.update(step=1)
        assert result == True
    
    def test_visualizer_update_matplotlib(self):
        """Test matplotlib update method."""
        env = Environment(width=5, height=5)
        visualizer = SimulationVisualizer(env, use_pygame=False)
        
        # Add some agents
        prey = Prey(0, env._get_random_empty_position(), energy=50)
        predator = Predator(1, env._get_random_empty_position(), energy=50)
        env._add_agent(prey, env._get_random_empty_position())
        env._add_agent(predator, env._get_random_empty_position())
        
        # Update should work without errors
        visualizer.update(step=1)
        
        # Stats should be updated
        assert len(visualizer.stats_history['prey']) > 0
        assert len(visualizer.stats_history['predators']) > 0
    
    def test_visualizer_close(self):
        """Test visualizer cleanup."""
        env = Environment(width=5, height=5)
        
        # Test pygame close
        visualizer_pygame = SimulationVisualizer(env, use_pygame=True)
        visualizer_pygame.close()
        
        # Test matplotlib close
        visualizer_matplotlib = SimulationVisualizer(env, use_pygame=False)
        visualizer_matplotlib.close()

class TestEnvironmentCreation:
    """Test cases for environment creation functions."""
    
    def test_create_environment_with_rl_agents(self):
        """Test creating environment with RL agents."""
        env = create_environment_with_rl_agents(
            width=10, height=10, 
            rl_prey_ratio=0.5, 
            rl_predator_ratio=0.5
        )
        
        assert env.width == 10
        assert env.height == 10
        
        # Should have some agents
        assert len(env.agents) > 0
        
        # Check for RL agents
        rl_prey_count = 0
        rl_predator_count = 0
        
        for agent in env.agents.values():
            if isinstance(agent, RLPrey):
                rl_prey_count += 1
            elif isinstance(agent, RLPredator):
                rl_predator_count += 1
        
        # Should have some RL agents
        assert rl_prey_count > 0 or rl_predator_count > 0
    
    def test_create_environment_no_rl_agents(self):
        """Test creating environment without RL agents."""
        env = create_environment_with_rl_agents(
            width=10, height=10, 
            rl_prey_ratio=0.0, 
            rl_predator_ratio=0.0
        )
        
        # Should have regular agents only
        for agent in env.agents.values():
            assert not isinstance(agent, (RLPrey, RLPredator))
            assert isinstance(agent, (Prey, Predator))

class TestSimulationExecution:
    """Test cases for simulation execution."""
    
    def test_run_headless_simulation(self):
        """Test headless simulation execution."""
        env = Environment(width=10, height=10)
        
        # Add some agents
        prey = Prey(0, env._get_random_empty_position(), energy=50)
        predator = Predator(1, env._get_random_empty_position(), energy=50)
        env._add_agent(prey, env._get_random_empty_position())
        env._add_agent(predator, env._get_random_empty_position())
        
        # Run headless simulation
        stats = run_headless_simulation(env, max_steps=10, print_interval=5)
        
        assert len(stats) > 0
        assert all('step' in stat for stat in stats)
        assert all('plants' in stat for stat in stats)
        assert all('prey' in stat for stat in stats)
        assert all('predators' in stat for stat in stats)
        assert all('total_agents' in stat for stat in stats)
    
    def test_run_headless_simulation_extinction(self):
        """Test headless simulation with extinction event."""
        env = Environment(width=5, height=5)
        
        # Add only one prey with very low energy
        prey = Prey(0, env._get_random_empty_position(), energy=1)
        env._add_agent(prey, env._get_random_empty_position())
        
        # Run simulation - should end due to extinction
        stats = run_headless_simulation(env, max_steps=100, print_interval=10)
        
        # Should have some stats before extinction
        assert len(stats) > 0
        
        # Final stats should show no prey
        final_stats = stats[-1]
        assert final_stats['prey'] == 0
    
    def test_run_simulation_with_visualization(self):
        """Test simulation with visualization (basic test)."""
        env = Environment(width=5, height=5)
        
        # Add some agents
        prey = Prey(0, env._get_random_empty_position(), energy=50)
        predator = Predator(1, env._get_random_empty_position(), energy=50)
        env._add_agent(prey, env._get_random_empty_position())
        env._add_agent(predator, env._get_random_empty_position())
        
        # Run simulation with very few steps and fast speed
        # This is a basic test to ensure no errors occur
        try:
            run_simulation(
                env, 
                max_steps=2, 
                visualization_speed=0.0,  # No delay
                use_pygame=False,  # Use matplotlib for testing
                save_stats=False
            )
        except Exception as e:
            # If pygame is not available, that's okay
            if "pygame" not in str(e).lower():
                raise e

class TestStatistics:
    """Test cases for statistics handling."""
    
    def test_save_simulation_stats(self):
        """Test saving simulation statistics to CSV."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_filename = tmp_file.name
        
        try:
            # Create sample statistics
            stats_list = [
                {'step': 0, 'plants': 10, 'prey': 5, 'predators': 2, 'total_agents': 7},
                {'step': 1, 'plants': 12, 'prey': 4, 'predators': 3, 'total_agents': 7},
                {'step': 2, 'plants': 11, 'prey': 3, 'predators': 2, 'total_agents': 5}
            ]
            
            # Save statistics
            save_simulation_stats(stats_list, tmp_filename)
            
            # Check that file was created and has content
            assert os.path.exists(tmp_filename)
            
            with open(tmp_filename, 'r') as f:
                content = f.read()
                assert 'step,plants,prey,predators,total_agents' in content
                assert '0,10,5,2,7' in content
                assert '1,12,4,3,7' in content
                assert '2,11,3,2,5' in content
        
        finally:
            # Clean up
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
    
    def test_environment_statistics(self):
        """Test environment statistics collection."""
        env = Environment(width=10, height=10)
        
        # Add agents
        prey1 = Prey(0, env._get_random_empty_position(), energy=50)
        prey2 = Prey(1, env._get_random_empty_position(), energy=50)
        predator = Predator(2, env._get_random_empty_position(), energy=50)
        
        env._add_agent(prey1, env._get_random_empty_position())
        env._add_agent(prey2, env._get_random_empty_position())
        env._add_agent(predator, env._get_random_empty_position())
        
        # Add plants
        env.grid[3, 3] = CellType.PLANT.value
        env.grid[4, 4] = CellType.PLANT.value
        env.grid[5, 5] = CellType.PLANT.value
        
        # Get statistics
        stats = env.get_statistics()
        
        assert stats['step'] == 0
        assert stats['plants'] == 3
        assert stats['prey'] == 2
        assert stats['predators'] == 1
        assert stats['total_agents'] == 3

class TestSimulationIntegration:
    """Integration tests for simulation components."""
    
    def test_full_simulation_cycle(self):
        """Test a complete simulation cycle."""
        # Create environment
        env = Environment(width=10, height=10)
        
        # Add agents
        prey = Prey(0, env._get_random_empty_position(), energy=50)
        predator = Predator(1, env._get_random_empty_position(), energy=50)
        env._add_agent(prey, env._get_random_empty_position())
        env._add_agent(predator, env._get_random_empty_position())
        
        # Add plants
        env.grid[3, 3] = CellType.PLANT.value
        env.grid[4, 4] = CellType.PLANT.value
        
        # Run simulation steps
        initial_stats = env.get_statistics()
        
        for step in range(5):
            env.step()
            stats = env.get_statistics()
            assert stats['step'] == step + 1
        
        final_stats = env.get_statistics()
        assert final_stats['step'] == 5
        
        # Agents should have aged
        assert prey.age > 0
        assert predator.age > 0
    
    def test_rl_agent_simulation(self):
        """Test simulation with RL agents."""
        env = create_environment_with_rl_agents(
            width=10, height=10,
            rl_prey_ratio=0.5,
            rl_predator_ratio=0.5
        )
        
        # Run a few steps
        for step in range(3):
            env.step()
            stats = env.get_statistics()
            assert stats['step'] == step + 1
        
        # Check that RL agents are learning
        rl_agents = [agent for agent in env.agents.values() 
                    if isinstance(agent, (RLPrey, RLPredator))]
        
        if rl_agents:
            # At least one RL agent should have some memory
            assert any(len(agent.memory) > 0 for agent in rl_agents)
    
    def test_visualization_statistics_tracking(self):
        """Test that visualization properly tracks statistics."""
        env = Environment(width=5, height=5)
        visualizer = SimulationVisualizer(env, use_pygame=False)
        
        # Add agents
        prey = Prey(0, env._get_random_empty_position(), energy=50)
        predator = Predator(1, env._get_random_empty_position(), energy=50)
        env._add_agent(prey, env._get_random_empty_position())
        env._add_agent(predator, env._get_random_empty_position())
        
        # Update visualization
        visualizer.update(step=0)
        
        # Check that statistics are tracked
        assert len(visualizer.stats_history['prey']) == 1
        assert len(visualizer.stats_history['predators']) == 1
        assert visualizer.stats_history['prey'][0] == 1
        assert visualizer.stats_history['predators'][0] == 1
        
        # Remove an agent and update again
        env._remove_agent(prey.id)
        visualizer.update(step=1)
        
        # Statistics should be updated
        assert visualizer.stats_history['prey'][1] == 0
        assert visualizer.stats_history['predators'][1] == 1

if __name__ == "__main__":
    pytest.main([__file__])
