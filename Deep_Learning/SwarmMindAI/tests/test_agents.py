"""
Tests for the agent classes.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from src.agents.base_agent import BaseAgent
from src.agents.agent_types import ExplorerAgent, CollectorAgent, CoordinatorAgent
from src.environment.world import World


class MockWorld:
    """Mock world for testing agents."""
    
    def __init__(self):
        self.width = 800
        self.height = 600
        self.obstacles = []
        self.resources = []
    
    def check_collision(self, x, y, radius):
        return False
    
    def get_nearby_resources(self, x, y, radius):
        return []
    
    def get_nearby_obstacles(self, x, y, radius):
        return []
    
    def collect_resource(self, x, y, radius):
        return None


class TestBaseAgent:
    """Test BaseAgent class."""
    
    @pytest.fixture
    def mock_world(self):
        """Create a mock world."""
        return MockWorld()
    
    @pytest.fixture
    def agent(self, mock_world):
        """Create a test agent."""
        # Create a concrete subclass for testing
        class TestAgent(BaseAgent):
            def decide_action(self, observations):
                return {"move": {"direction": [1, 0], "speed": 5.0}}
        
        return TestAgent("test_agent", (100, 100), "test", mock_world)
    
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.agent_id == "test_agent"
        assert np.array_equal(agent.position, np.array([100, 100]))
        assert agent.agent_type == "test"
        assert agent.energy == 100.0
        assert agent.max_energy == 100.0
        assert agent.radius == 15.0
        assert agent.max_speed == 5.0
    
    def test_agent_capabilities(self, agent):
        """Test agent capabilities."""
        assert agent.capabilities["mobility"] is True
        assert agent.capabilities["sensor_capability"] is True
        assert agent.capabilities["collection_capability"] is False
        assert agent.capabilities["communication"] is True
    
    def test_agent_movement(self, agent):
        """Test agent movement."""
        initial_position = agent.position.copy()
        
        # Move agent
        agent.move((1, 0), 3.0)
        
        # Check position changed
        assert not np.array_equal(agent.position, initial_position)
        assert agent.movement_mode == "moving"
    
    def test_agent_energy_consumption(self, agent):
        """Test agent energy consumption."""
        initial_energy = agent.energy
        
        # Move agent to consume energy
        agent.move((1, 0), 5.0)
        agent.update_energy()
        
        assert agent.energy < initial_energy
    
    def test_agent_sensing(self, agent):
        """Test agent sensing capabilities."""
        observations = agent.sense_environment()
        
        assert "position" in observations
        assert "velocity" in observations
        assert "energy" in observations
        assert "nearby_resources" in observations
        assert "nearby_obstacles" in observations
        assert "nearby_agents" in observations
        assert "messages" in observations
    
    def test_agent_communication(self, agent):
        """Test agent communication."""
        # Send message
        agent.send_message({"type": "test"}, "other_agent")
        assert len(agent.message_queue) == 1
        
        # Receive message (within broadcast range)
        message = {
            "sender": "other_agent",
            "content": {"type": "test"},
            "position": [150, 100]  # Within 80 unit broadcast range
        }
        agent.receive_message(message)
        assert len(agent.local_messages) == 1
    
    def test_agent_state(self, agent):
        """Test getting agent state."""
        state = agent.get_agent_state()
        
        assert "agent_id" in state
        assert "agent_type" in state
        assert "position" in state
        assert "energy" in state
        assert "capabilities" in state
        assert "performance" in state
    
    def test_agent_step(self, agent):
        """Test agent step execution."""
        step_result = agent.step()
        
        assert "agent_id" in step_result
        assert "actions" in step_result
        assert "status" in step_result
        assert step_result["status"] == "active"
    
    def test_agent_reset(self, agent):
        """Test agent reset functionality."""
        # Modify agent state
        agent.energy = 50.0
        agent.steps_alive = 100
        agent.send_message({"type": "test"})
        
        # Reset agent
        agent.reset()
        
        assert agent.energy == agent.max_energy
        assert agent.steps_alive == 0
        assert len(agent.message_queue) == 0
    
    def test_agent_inactive(self, agent):
        """Test agent inactive state."""
        # Deplete energy
        agent.energy = 0.0
        
        assert not agent.is_active()
        
        # Step should return inactive status
        step_result = agent.step()
        assert step_result["status"] == "inactive"


class TestExplorerAgent:
    """Test ExplorerAgent class."""
    
    @pytest.fixture
    def mock_world(self):
        """Create a mock world."""
        world = MockWorld()
        world.width = 800
        world.height = 600
        return world
    
    @pytest.fixture
    def explorer(self, mock_world):
        """Create a test explorer agent."""
        return ExplorerAgent("explorer_1", (100, 100), mock_world)
    
    def test_explorer_initialization(self, explorer):
        """Test explorer agent initialization."""
        assert explorer.agent_type == "explorer"
        assert explorer.exploration_mode == "systematic"
        assert explorer.movement_pattern == "spiral"
        assert explorer.sensor_range == 80.0
        assert explorer.vision_range == 150.0
    
    def test_explorer_capabilities(self, explorer):
        """Test explorer agent capabilities."""
        assert explorer.capabilities["exploration"] is True
        assert explorer.capabilities["collection_capability"] is False
    
    def test_exploration_targets(self, explorer):
        """Test exploration target initialization."""
        assert len(explorer.exploration_targets) > 0
        
        # Check target positions are within world bounds
        for target in explorer.exploration_targets:
            x, y = target
            assert 50 <= x <= explorer.world.width - 50
            assert 50 <= y <= explorer.world.height - 50
    
    def test_spiral_movement(self, explorer):
        """Test spiral movement pattern."""
        initial_angle = explorer.spiral_angle
        
        # Get spiral movement
        movement = explorer._spiral_movement()
        
        assert "direction" in movement
        assert "speed" in movement
        assert explorer.spiral_angle > initial_angle
    
    def test_grid_movement(self, explorer):
        """Test grid movement pattern."""
        if explorer.exploration_targets:
            target = explorer.exploration_targets[0]
            movement = explorer._grid_movement()
            
            assert "direction" in movement
            assert "speed" in movement
    
    def test_explorer_action_decision(self, explorer):
        """Test explorer action decision making."""
        observations = {
            "nearby_resources": [{"type": "food", "position": (150, 150), "value": 5.0, "distance": 30}],
            "nearby_obstacles": [],
            "position": [100, 100]
        }
        
        action = explorer.decide_action(observations)
        
        # Should communicate resource discovery
        assert "communicate" in action
        assert action["communicate"]["message"]["type"] == "resource_discovery"
        
        # Should have movement action
        assert "move" in action
    
    def test_explorer_metrics(self, explorer):
        """Test explorer metrics."""
        metrics = explorer.get_exploration_metrics()
        
        assert "discovered_resources" in metrics
        assert "exploration_targets_remaining" in metrics
        assert "coverage_area" in metrics
        assert "exploration_efficiency" in metrics


class TestCollectorAgent:
    """Test CollectorAgent class."""
    
    @pytest.fixture
    def mock_world(self):
        """Create a mock world."""
        world = MockWorld()
        world.width = 800
        world.height = 600
        return world
    
    @pytest.fixture
    def collector(self, mock_world):
        """Create a test collector agent."""
        return CollectorAgent("collector_1", (100, 100), mock_world)
    
    def test_collector_initialization(self, collector):
        """Test collector agent initialization."""
        assert collector.agent_type == "collector"
        assert collector.collection_mode == "efficient"
        assert collector.carrying_capacity == 5
        assert collector.collection_range == 25.0
        assert collector.collection_speed == 1.5
    
    def test_collector_capabilities(self, collector):
        """Test collector agent capabilities."""
        assert collector.capabilities["collection"] is True
        assert collector.capabilities["collection_capability"] is True
    
    def test_collection_strategies(self, collector):
        """Test collection strategy initialization."""
        assert "energy" in collector.resource_weights
        assert "mineral" in collector.resource_weights
        assert "food" in collector.resource_weights
        assert "water" in collector.resource_weights
        
        # Check weights are reasonable
        assert collector.resource_weights["energy"] > collector.resource_weights["water"]
    
    def test_resource_scoring(self, collector):
        """Test resource scoring system."""
        resource = {
            "type": "energy",
            "position": (150, 150),
            "value": 8.0,
            "distance": 50
        }
        
        score = collector._calculate_resource_score(resource)
        assert score > 0
        
        # Energy should score higher than water
        water_resource = resource.copy()
        water_resource["type"] = "water"
        water_score = collector._calculate_resource_score(water_resource)
        assert score > water_score
    
    def test_collector_action_decision(self, collector):
        """Test collector action decision making."""
        observations = {
            "nearby_resources": [{"type": "energy", "position": (150, 150), "value": 8.0, "distance": 30}],
            "position": [100, 100]
        }
        
        action = collector.decide_action(observations)
        
        # Should attempt to collect resource
        assert "collect" in action
        assert action["collect"]["position"] == (150, 150)
    
    def test_collection_target(self, collector):
        """Test collection target setting."""
        resource_info = {
            "position": (200, 200),
            "type": "mineral",
            "value": 6.0
        }
        
        collector.set_collection_target(resource_info)
        assert collector.collection_target == resource_info
    
    def test_collector_metrics(self, collector):
        """Test collector metrics."""
        metrics = collector.get_collection_metrics()
        
        assert "resources_collected" in metrics
        assert "current_cargo" in metrics
        assert "carrying_capacity" in metrics
        assert "collection_efficiency" in metrics


class TestCoordinatorAgent:
    """Test CoordinatorAgent class."""
    
    @pytest.fixture
    def mock_world(self):
        """Create a mock world."""
        world = MockWorld()
        world.width = 800
        world.height = 600
        return world
    
    @pytest.fixture
    def coordinator(self, mock_world):
        """Create a test coordinator agent."""
        return CoordinatorAgent("coordinator_1", (100, 100), mock_world)
    
    def test_coordinator_initialization(self, coordinator):
        """Test coordinator agent initialization."""
        assert coordinator.agent_type == "coordinator"
        assert coordinator.coordination_mode == "centralized"
        assert coordinator.communication_range == 200.0
        assert coordinator.broadcast_frequency == 10
    
    def test_coordinator_capabilities(self, coordinator):
        """Test coordinator agent capabilities."""
        assert coordinator.capabilities["coordination"] is True
        assert coordinator.capabilities["collection_capability"] is False
    
    def test_coordination_systems(self, coordinator):
        """Test coordination system initialization."""
        assert len(coordinator.coordination_strategies) > 0
        assert len(coordinator.strategic_goals) > 0
        
        # Check for key strategies
        assert "task_optimization" in coordinator.coordination_strategies
        assert "swarm_formation" in coordinator.coordination_strategies
    
    def test_situation_analysis(self, coordinator):
        """Test situation analysis."""
        observations = {
            "nearby_resources": [{"type": "food", "position": (150, 150), "value": 5.0, "distance": 30}],
            "position": [100, 100]
        }
        
        analysis = coordinator._analyze_situation(observations)
        
        assert "resource_distribution" in analysis
        assert "agent_positions" in analysis
        assert "task_progress" in analysis
        assert "swarm_cohesion" in analysis
        assert "performance_metrics" in analysis
    
    def test_coordination_actions(self, coordinator):
        """Test coordination action generation."""
        # Mock situation analysis
        analysis = {
            "task_progress": {"efficiency": 0.5},  # Below threshold
            "swarm_cohesion": 0.6,  # Below threshold
            "performance_metrics": {"coordination_efficiency": 0.8}
        }
        
        actions = coordinator._generate_coordination_actions(analysis)
        
        # Should generate actions for low efficiency and cohesion
        assert len(actions) > 0
        
        action_types = [action["type"] for action in actions]
        assert "task_assignment" in action_types or "formation_control" in action_types
    
    def test_formation_movement(self, coordinator):
        """Test formation movement calculation."""
        movement = coordinator._calculate_formation_movement({
            "nearby_resources": [{"position": (150, 150)}]
        })
        
        assert "direction" in movement
        assert "speed" in movement
    
    def test_coordinator_metrics(self, coordinator):
        """Test coordinator metrics."""
        metrics = coordinator.get_coordination_metrics()
        
        assert "managed_agents" in metrics
        assert "active_assignments" in metrics
        assert "coordination_efficiency" in metrics
        assert "swarm_cohesion" in metrics
        assert "strategic_goals" in metrics
