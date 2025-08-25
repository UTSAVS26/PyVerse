"""
Tests for the World class.
"""

import pytest
import numpy as np
from src.environment.world import World, Obstacle, Resource


class TestObstacle:
    """Test Obstacle class."""
    
    def test_obstacle_creation(self):
        """Test obstacle creation with default values."""
        obstacle = Obstacle(x=100, y=200, radius=30)
        
        assert obstacle.x == 100
        assert obstacle.y == 200
        assert obstacle.radius == 30
        assert obstacle.obstacle_type == "static"
    
    def test_obstacle_custom_type(self):
        """Test obstacle creation with custom type."""
        obstacle = Obstacle(x=150, y=250, radius=25, obstacle_type="dynamic")
        
        assert obstacle.obstacle_type == "dynamic"


class TestResource:
    """Test Resource class."""
    
    def test_resource_creation(self):
        """Test resource creation."""
        resource = Resource(x=300, y=400, resource_type="food", value=5.0, quantity=3)
        
        assert resource.x == 300
        assert resource.y == 400
        assert resource.resource_type == "food"
        assert resource.value == 5.0
        assert resource.quantity == 3
        assert not resource.collected
    
    def test_resource_collection(self):
        """Test resource collection."""
        resource = Resource(x=300, y=400, resource_type="mineral", value=8.0, quantity=2)
        
        assert not resource.collected
        resource.collected = True
        assert resource.collected


class TestWorld:
    """Test World class."""
    
    @pytest.fixture
    def world(self):
        """Create a test world."""
        return World(width=800, height=600, seed=42)
    
    def test_world_initialization(self, world):
        """Test world initialization."""
        assert world.width == 800
        assert world.height == 600
        assert world.seed == 42
        assert world.time_step == 0
        assert world.weather_conditions == "clear"
        assert len(world.obstacles) > 0
        assert len(world.resources) > 0
    
    def test_world_seed_reproducibility(self):
        """Test that world generation is reproducible with same seed."""
        world1 = World(width=400, height=300, seed=123)
        world2 = World(width=400, height=300, seed=123)
        
        # Check that obstacles are in same positions
        obs1_positions = [(o.x, o.y, o.radius) for o in world1.obstacles]
        obs2_positions = [(o.x, o.y, o.radius) for o in world2.obstacles]
        assert obs1_positions == obs2_positions
        
        # Check that resources are in same positions
        res1_positions = [(r.x, r.y, r.resource_type, r.value) for r in world1.resources]
        res2_positions = [(r.x, r.y, r.resource_type, r.value) for r in world2.resources]
        assert res1_positions == res2_positions
    
    def test_collision_detection(self, world):
        """Test collision detection."""
        # Test no collision
        assert not world.check_collision(50, 50, 10)
        
        # Test collision with obstacle
        for obstacle in world.obstacles:
            # Position inside obstacle boundary (should collide)
            x = obstacle.x + obstacle.radius - 5
            y = obstacle.y
            assert world.check_collision(x, y, 10)
    
    def test_nearby_resources(self, world):
        """Test getting nearby resources."""
        # Get a resource position
        resource = world.resources[0]
        
        # Test getting resources within range
        nearby = world.get_nearby_resources(resource.x, resource.y, 50)
        assert len(nearby) > 0
        assert resource in nearby
        
        # Test getting resources outside range
        far_resources = world.get_nearby_resources(0, 0, 10)
        assert len(far_resources) == 0
    
    def test_nearby_obstacles(self, world):
        """Test getting nearby obstacles."""
        # Get an obstacle position
        obstacle = world.obstacles[0]
        
        # Test getting obstacles within range
        nearby = world.get_nearby_obstacles(obstacle.x, obstacle.y, 100)
        assert len(nearby) > 0
        assert obstacle in nearby
    
    def test_resource_collection(self, world):
        """Test resource collection."""
        # Get an uncollected resource
        uncollected_resources = [r for r in world.resources if not r.collected]
        if uncollected_resources:
            resource = uncollected_resources[0]
            original_count = len([r for r in world.resources if not r.collected])
            
            # Collect resource
            collected = world.collect_resource(resource.x, resource.y, 20)
            assert collected is not None
            assert collected.collected
            
            # Check count decreased
            new_count = len([r for r in world.resources if not r.collected])
            assert new_count == original_count - 1
    
    def test_dynamic_obstacle_creation(self, world):
        """Test dynamic obstacle creation."""
        initial_count = len(world.dynamic_elements)
        
        # Add dynamic obstacle
        world.add_dynamic_obstacle(
            x=100, y=100, radius=15, 
            velocity=(2, 1), lifetime=50
        )
        
        assert len(world.dynamic_elements) == initial_count + 1
        
        # Check obstacle properties
        dynamic_obs = world.dynamic_elements[-1]
        assert dynamic_obs["obstacle"].obstacle_type == "dynamic"
        assert dynamic_obs["velocity"] == (2, 1)
        assert dynamic_obs["lifetime"] == 50
        assert dynamic_obs["age"] == 0
    
    def test_world_update(self, world):
        """Test world update functionality."""
        initial_step = world.time_step
        
        # Add a dynamic obstacle
        world.add_dynamic_obstacle(
            x=200, y=200, radius=20, 
            velocity=(1, 0), lifetime=10
        )
        
        # Update world multiple times
        for _ in range(15):
            world.update()
        
        # Check that dynamic obstacle was removed after lifetime
        dynamic_obstacles = [e for e in world.dynamic_elements if e["obstacle"].obstacle_type == "dynamic"]
        assert len(dynamic_obstacles) == 0
        
        # Check time step increased
        assert world.time_step > initial_step
    
    def test_world_state(self, world):
        """Test getting world state."""
        state = world.get_world_state()
        
        assert "width" in state
        assert "height" in state
        assert "time_step" in state
        assert "num_obstacles" in state
        assert "num_resources" in state
        assert "num_dynamic_elements" in state
        assert "weather_conditions" in state
        
        assert state["width"] == world.width
        assert state["height"] == world.height
        assert state["num_obstacles"] == len(world.obstacles)
        assert state["num_resources"] == len([r for r in world.resources if not r.collected])
    
    def test_world_reset(self, world):
        """Test world reset functionality."""
        # Modify world state
        world.time_step = 100
        world.add_dynamic_obstacle(100, 100, 20, (1, 1), 50)
        
        # Reset world
        world.reset()
        
        assert world.time_step == 0
        assert len(world.dynamic_elements) == 0
        assert len(world.obstacles) > 0  # Obstacles should be regenerated
        assert len(world.resources) > 0   # Resources should be regenerated
    
    def test_spatial_grid(self, world):
        """Test spatial partitioning grid."""
        # Check that spatial grid is populated
        assert len(world.spatial_grid) > 0
        
        # Check that obstacles are in grid
        for obstacle in world.obstacles:
            grid_x = int(obstacle.x // world.grid_size)
            grid_y = int(obstacle.y // world.grid_size)
            grid_key = (grid_x, grid_y)
            
            if grid_key in world.spatial_grid:
                grid_contents = world.spatial_grid[grid_key]
                obstacle_found = any(
                    item[0] == "obstacle" and item[1] == obstacle 
                    for item in grid_contents
                )
                assert obstacle_found
    
    def test_boundary_conditions(self, world):
        """Test world boundary conditions."""
        # Test that obstacles are within bounds
        for obstacle in world.obstacles:
            assert obstacle.x >= obstacle.radius
            assert obstacle.x <= world.width - obstacle.radius
            assert obstacle.y >= obstacle.radius
            assert obstacle.y <= world.height - obstacle.radius
        
        # Test that resources are within bounds
        for resource in world.resources:
            assert resource.x >= 50
            assert resource.x <= world.width - 50
            assert resource.y >= 50
            assert resource.y <= world.height - 50
    
    def test_dynamic_obstacle_movement(self, world):
        """Test dynamic obstacle movement and bouncing."""
        # Add dynamic obstacle near boundary that will hit it
        world.add_dynamic_obstacle(
            x=world.width - 25, y=world.height - 25, 
            radius=20, velocity=(10, 10), lifetime=100
        )
        
        initial_pos = (world.dynamic_elements[-1]["obstacle"].x, world.dynamic_elements[-1]["obstacle"].y)
        initial_vel = world.dynamic_elements[-1]["velocity"]
        
        # Update world
        world.update()
        
        # Check that obstacle moved
        new_pos = (world.dynamic_elements[-1]["obstacle"].x, world.dynamic_elements[-1]["obstacle"].y)
        assert new_pos[0] != initial_pos[0] or new_pos[1] != initial_pos[1]
        
        # Check that velocity changed due to boundary bouncing
        new_vel = world.dynamic_elements[-1]["velocity"]
        assert new_vel != initial_vel
    
    def test_resource_types(self, world):
        """Test that resources have valid types."""
        valid_types = {"food", "mineral", "energy", "water"}
        
        for resource in world.resources:
            assert resource.resource_type in valid_types
            assert resource.value > 0
            assert resource.quantity > 0
    
    def test_obstacle_radius_range(self, world):
        """Test that obstacles have reasonable radius values."""
        for obstacle in world.obstacles:
            assert 20 <= obstacle.radius <= 60  # Based on initialization parameters
