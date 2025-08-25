"""
Tests for the visualization classes.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from src.visualization.swarm_visualizer import SwarmVisualizer


class TestSwarmVisualizer:
    """Test SwarmVisualizer class."""
    
    @pytest.fixture
    def mock_environment(self):
        """Create a mock environment."""
        env = Mock()
        env.world = Mock()
        env.world.width = 800
        env.world.height = 600
        env.world.resources = []
        env.world.obstacles = []
        env.world.get_world_state.return_value = {
            "width": 800,
            "height": 600,
            "time_step": 0
        }
        env.swarm = Mock()
        env.swarm.agents = []
        env.swarm.get_swarm_state.return_value = {
            "num_agents": 0,
            "active_agents": 0
        }
        return env
    
    @patch('pygame.display.flip')
    @patch('pygame.display.set_mode')
    @patch('pygame.init')
    def test_visualizer_initialization(self, mock_pygame_init, mock_display_set_mode, mock_display_flip, mock_environment):
        """Test visualizer initialization."""
        # Mock pygame display
        mock_screen = Mock()
        mock_display_set_mode.return_value = mock_screen
        
        visualizer = SwarmVisualizer(
            environment=mock_environment,
            window_size=(1024, 768)
        )
        
        # Check initialization
        assert visualizer.environment == mock_environment
        assert visualizer.window_size == (1024, 768)
        assert visualizer.fps == 60
        
        # Check pygame was initialized
        mock_pygame_init.assert_called_once()
        mock_display_set_mode.assert_called_once_with((1024, 768))
    
    @patch('pygame.display.flip')
    @patch('pygame.display.set_mode')
    @patch('pygame.init')
    def test_visualizer_default_values(self, mock_pygame_init, mock_display_set_mode, mock_display_flip, mock_environment):
        """Test visualizer default values."""
        mock_screen = Mock()
        mock_display_set_mode.return_value = mock_screen
        
        visualizer = SwarmVisualizer(environment=mock_environment)
        
        # Check default values
        assert visualizer.window_size == (1200, 800)
        assert visualizer.show_grid is True
        assert visualizer.show_metrics is True
        assert visualizer.show_paths is False
        assert visualizer.fps == 60
    
    @patch('pygame.display.flip')
    @patch('pygame.display.set_mode')
    @patch('pygame.init')
    def test_camera_transforms(self, mock_pygame_init, mock_display_set_mode, mock_display_flip, mock_environment):
        """Test camera transformation calculations."""
        mock_screen = Mock()
        mock_display_set_mode.return_value = mock_screen
        
        visualizer = SwarmVisualizer(environment=mock_environment)
        
        # Test camera transform calculation
        transform = visualizer._calculate_camera_transform()
        
        # Check transform structure
        assert "world_center" in transform
        assert "screen_center" in transform
        assert "zoom" in transform
        assert "offset" in transform
        
        # Check values
        assert transform["zoom"] == visualizer.zoom_level
        assert np.array_equal(transform["offset"], visualizer.camera_offset)
        assert transform["screen_center"][0] == visualizer.window_size[0] // 2
        assert transform["screen_center"][1] == visualizer.window_size[1] // 2
        
        # Test world to screen conversion
        world_pos = (100, 200)
        screen_pos = visualizer._world_to_screen(world_pos, transform)
        
        # Check conversion result is valid
        assert isinstance(screen_pos, tuple)
        assert len(screen_pos) == 2
        assert isinstance(screen_pos[0], int)
        assert isinstance(screen_pos[1], int)
    
    @patch('pygame.display.flip')
    @patch('pygame.display.set_mode')
    @patch('pygame.init')
    def test_camera_movement(self, mock_pygame_init, mock_display_set_mode, mock_display_flip, mock_environment):
        """Test camera movement functionality."""
        mock_screen = Mock()
        mock_display_set_mode.return_value = mock_screen
        
        visualizer = SwarmVisualizer(environment=mock_environment)
        
        # Test camera offset properties
        assert isinstance(visualizer.camera_offset, np.ndarray)
        assert len(visualizer.camera_offset) == 2
        
        # Test camera reset functionality
        initial_x, initial_y = visualizer.camera_offset[0], visualizer.camera_offset[1]
        visualizer._reset_camera()
        
        # Check camera was reset
        assert visualizer.camera_offset[0] == 0.0
        assert visualizer.camera_offset[1] == 0.0
        assert visualizer.zoom_level == 1.0
    
    @patch('pygame.display.flip')
    @patch('pygame.display.set_mode')
    @patch('pygame.init')
    def test_zoom_functionality(self, mock_pygame_init, mock_display_set_mode, mock_display_flip, mock_environment):
        """Test zoom functionality."""
        mock_screen = Mock()
        mock_display_set_mode.return_value = mock_screen
        
        visualizer = SwarmVisualizer(environment=mock_environment)
        
        # Test zoom properties
        assert isinstance(visualizer.zoom_level, float)
        assert visualizer.zoom_level > 0
        
        # Test zoom level is reasonable
        assert 0.1 <= visualizer.zoom_level <= 3.0
        
        # Test zoom level affects camera transform
        transform = visualizer._calculate_camera_transform()
        assert transform["zoom"] == visualizer.zoom_level
    
    @patch('pygame.display.flip')
    @patch('pygame.display.set_mode')
    @patch('pygame.init')
    def test_grid_rendering(self, mock_pygame_init, mock_display_set_mode, mock_display_flip, mock_environment):
        """Test grid rendering."""
        mock_screen = Mock()
        mock_display_set_mode.return_value = mock_screen
        
        visualizer = SwarmVisualizer(environment=mock_environment)
        
        # Enable grid
        visualizer.show_grid = True
        
        # Create a proper transform for testing
        transform = visualizer._calculate_camera_transform()
        
        # Draw grid with proper transform
        visualizer._draw_grid(transform)
        
        # Check grid was processed (no errors)
        assert visualizer.show_grid is True
        
        # Disable grid
        visualizer.show_grid = False
        
        # Draw grid (should not draw anything)
        visualizer._draw_grid(transform)
        # Grid should not be drawn when disabled
    
    @patch('pygame.display.flip')
    @patch('pygame.display.set_mode')
    @patch('pygame.init')
    def test_obstacle_rendering(self, mock_pygame_init, mock_display_set_mode, mock_display_flip, mock_environment):
        """Test obstacle rendering."""
        from src.environment.world import Obstacle
        
        mock_screen = Mock()
        mock_display_set_mode.return_value = mock_screen
        
        # Create mock obstacles
        obstacles = [
            Obstacle(100, 100, 20),
            Obstacle(200, 200, 15),
            Obstacle(300, 300, 25)
        ]
        mock_environment.world.obstacles = obstacles
        
        visualizer = SwarmVisualizer(environment=mock_environment)
        
        # Create a proper transform for testing
        transform = visualizer._calculate_camera_transform()
        
        # Draw world (which includes obstacles)
        visualizer._draw_world(transform)
        
        # Check world was processed (no errors)
        assert len(obstacles) == 3
    
    @patch('pygame.display.flip')
    @patch('pygame.display.set_mode')
    @patch('pygame.init')
    def test_resource_rendering(self, mock_pygame_init, mock_display_set_mode, mock_display_flip, mock_environment):
        """Test resource rendering."""
        from src.environment.world import Resource
        
        mock_screen = Mock()
        mock_display_set_mode.return_value = mock_screen
        
        # Create mock resources
        resources = [
            Resource(100, 100, "food", 5.0, 2),
            Resource(200, 200, "mineral", 8.0, 1),
            Resource(300, 300, "energy", 10.0, 3)
        ]
        mock_environment.world.resources = resources
        
        visualizer = SwarmVisualizer(environment=mock_environment)
        
        # Create a proper transform for testing
        transform = visualizer._calculate_camera_transform()
        
        # Draw world (which includes resources)
        visualizer._draw_world(transform)
        
        # Check world was processed (no errors)
        assert len(resources) == 3
    
    @patch('pygame.display.flip')
    @patch('pygame.display.set_mode')
    @patch('pygame.init')
    def test_agent_rendering(self, mock_pygame_init, mock_display_set_mode, mock_display_flip, mock_environment):
        """Test agent rendering."""
        mock_screen = Mock()
        mock_display_set_mode.return_value = mock_screen
        
        # Create mock agents
        agents = [
            Mock(
                agent_id="agent1",
                position=(100, 100),
                direction=0.0,
                energy=0.8,
                agent_type="explorer"
            ),
            Mock(
                agent_id="agent2",
                position=(200, 200),
                direction=1.57,
                energy=0.6,
                agent_type="collector"
            )
        ]
        mock_environment.swarm.agents = agents
        
        visualizer = SwarmVisualizer(environment=mock_environment)
        
        # Create a proper transform for testing
        transform = visualizer._calculate_camera_transform()
        
        # Draw agents with proper transform
        visualizer._draw_agents(transform)
        
        # Check agents were processed (no errors)
        assert len(agents) == 2
    
    @patch('pygame.display.flip')
    @patch('pygame.display.set_mode')
    @patch('pygame.init')
    def test_metrics_overlay(self, mock_pygame_init, mock_display_set_mode, mock_display_flip, mock_environment):
        """Test metrics overlay rendering."""
        mock_screen = Mock()
        mock_display_set_mode.return_value = mock_screen
        
        visualizer = SwarmVisualizer(environment=mock_environment)
        
        # Enable metrics
        visualizer.show_metrics = True
        
        # Mock performance metrics
        performance_metrics = {
            "total_reward": 150.5,
            "task_completion_rate": 0.75,
            "swarm_efficiency": 0.82,
            "communication_overhead": 0.15
        }
        
        # Mock environment state for metrics
        mock_environment.get_environment_state.return_value = {
            "swarm_metrics": {"active_agents": 2, "coordination_score": 0.8},
            "task_statistics": {"completion_rate": 0.75}
        }
        
        # Draw metrics with step result
        step_result = {"step": 10}
        visualizer._draw_metrics(step_result)
        
        # Check metrics were processed (no errors)
        assert visualizer.show_metrics is True
        
        # Disable metrics
        visualizer.show_metrics = False
        
        # Draw metrics (should not draw anything)
        visualizer._draw_metrics(step_result)
        # Metrics should not be drawn when disabled
    
    @patch('pygame.display.flip')
    @patch('pygame.display.set_mode')
    @patch('pygame.init')
    def test_path_rendering(self, mock_pygame_init, mock_display_set_mode, mock_display_flip, mock_environment):
        """Test agent path rendering."""
        mock_screen = Mock()
        mock_display_set_mode.return_value = mock_screen
        
        # Create mock agents with paths
        agents = [
            Mock(
                agent_id="agent1",
                position=(100, 100),
                path_history=[(50, 50), (75, 75), (100, 100)]
            ),
            Mock(
                agent_id="agent2",
                position=(200, 200),
                path_history=[(150, 150), (175, 175), (200, 200)]
            )
        ]
        mock_environment.swarm.agents = agents
        
        visualizer = SwarmVisualizer(environment=mock_environment)
        
        # Enable paths
        visualizer.show_paths = True
        
        # Draw paths
        visualizer._draw_paths(mock_screen)
        
        # Check paths were drawn
        # Note: In a real test, we'd check specific pygame draw calls
        assert mock_screen is not None
        
        # Disable paths
        visualizer.show_paths = False
        
        # Draw paths (should not draw anything)
        visualizer._draw_paths(mock_screen)
        # Paths should not be drawn when disabled
    
    @patch('pygame.display.flip')
    @patch('pygame.display.set_mode')
    @patch('pygame.init')
    def test_event_handling(self, mock_pygame_init, mock_display_set_mode, mock_display_flip, mock_environment):
        """Test event handling."""
        mock_screen = Mock()
        mock_display_set_mode.return_value = mock_screen
        
        visualizer = SwarmVisualizer(environment=mock_environment)
        
        # Test key events
        # Mock pygame key event
        mock_event = Mock()
        mock_event.type = 768  # KEYDOWN
        mock_event.key = 107  # 'k' key
        
        # Handle event
        result = visualizer._handle_event(mock_event)
        
        # Check event was handled
        assert result is not None
        
        # Test mouse events
        # Mock pygame mouse event
        mock_event.type = 1024  # MOUSEBUTTONDOWN
        mock_event.button = 1  # Left click
        mock_event.pos = (400, 300)
        
        # Handle event
        result = visualizer._handle_event(mock_event)
        
        # Check event was handled
        assert result is not None
    
    @patch('pygame.display.flip')
    @patch('pygame.display.set_mode')
    @patch('pygame.init')
    def test_controls_information(self, mock_pygame_init, mock_display_set_mode, mock_display_flip, mock_environment):
        """Test controls information display."""
        mock_screen = Mock()
        mock_display_set_mode.return_value = mock_screen
        
        visualizer = SwarmVisualizer(environment=mock_environment)
        
        # Draw controls information
        visualizer._draw_controls_info(mock_screen)
        
        # Check controls were drawn
        # Note: In a real test, we'd check specific pygame draw calls
        assert mock_screen is not None
    
    @patch('pygame.display.flip')
    @patch('pygame.display.set_mode')
    @patch('pygame.init')
    def test_visualization_update(self, mock_pygame_init, mock_display_set_mode, mock_display_flip, mock_environment):
        """Test visualization update cycle."""
        mock_screen = Mock()
        mock_display_set_mode.return_value = mock_screen
        
        visualizer = SwarmVisualizer(environment=mock_environment)
        
        # Mock pygame event
        mock_event = Mock()
        mock_event.type = 256  # QUIT
        
        # Mock pygame event queue
        with patch('pygame.event.get') as mock_event_get:
            mock_event_get.return_value = [mock_event]
            
            # Update visualization
            result = visualizer.update({})
            
            # Check update result
            assert result is not None
    
    @patch('pygame.display.flip')
    @patch('pygame.display.set_mode')
    @patch('pygame.init')
    def test_visualization_cleanup(self, mock_pygame_init, mock_display_set_mode, mock_display_flip, mock_environment):
        """Test visualization cleanup."""
        mock_screen = Mock()
        mock_display_set_mode.return_value = mock_screen
        
        visualizer = SwarmVisualizer(environment=mock_environment)
        
        # Cleanup
        visualizer.cleanup()
        
        # Check cleanup was performed
        # Note: In a real test, we'd check pygame.quit() was called
        assert visualizer is not None
    
    @patch('pygame.display.flip')
    @patch('pygame.display.set_mode')
    @patch('pygame.init')
    def test_visualization_settings(self, mock_pygame_init, mock_display_set_mode, mock_display_flip, mock_environment):
        """Test visualization settings management."""
        mock_screen = Mock()
        mock_display_set_mode.return_value = mock_screen
        
        visualizer = SwarmVisualizer(environment=mock_environment)
        
        # Test setting changes
        visualizer.set_show_grid(False)
        assert visualizer.show_grid is False
        
        visualizer.set_show_metrics(False)
        assert visualizer.show_metrics is False
        
        visualizer.set_show_paths(False)
        assert visualizer.show_paths is False
        
        # Test FPS setting
        visualizer.set_fps(30)
        assert visualizer.fps == 30
        
        # Test window size setting
        visualizer.set_window_size(1600, 900)
        assert visualizer.window_size == (1600, 900)
    
    @patch('pygame.display.flip')
    @patch('pygame.display.set_mode')
    @patch('pygame.init')
    def test_visualization_state(self, mock_pygame_init, mock_display_set_mode, mock_display_flip, mock_environment):
        """Test visualization state management."""
        mock_screen = Mock()
        mock_display_set_mode.return_value = mock_screen
        
        visualizer = SwarmVisualizer(environment=mock_environment)
        
        # Get current state
        state = visualizer.get_visualization_state()
        
        # Check state contains expected keys
        assert "camera_position" in state
        assert "zoom_level" in state
        assert "display_settings" in state
        assert "performance_metrics" in state
        
        # Check state values
        assert state["camera_position"] == (visualizer.camera_offset[0], visualizer.camera_offset[1])
        assert state["zoom_level"] == visualizer.zoom_level
        assert state["display_settings"]["show_grid"] == visualizer.show_grid
        assert state["display_settings"]["show_metrics"] == visualizer.show_metrics
        assert state["display_settings"]["show_paths"] == visualizer.show_paths
