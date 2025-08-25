"""
Tests for the environment classes.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from src.environment.world import World
from src.environment.tasks import (
    Task, ResourceCollectionTask, SearchAndRescueTask, 
    AreaCoverageTask, TaskManager, TaskStatus
)
from src.environment.swarm_environment import SwarmEnvironment


class TestTaskStatus:
    """Test TaskStatus class."""
    
    def test_task_status_initialization(self):
        """Test task status initialization."""
        status = TaskStatus()
        
        assert not status.completed
        assert status.progress == 0.0
        assert status.assigned_agents == []
        assert status.start_time == 0
        assert status.completion_time is None
        assert status.efficiency_score == 0.0
    
    def test_task_status_custom_values(self):
        """Test task status with custom values."""
        status = TaskStatus(
            completed=True,
            progress=0.8,
            assigned_agents=["agent1", "agent2"],
            start_time=10,
            completion_time=50,
            efficiency_score=0.9
        )
        
        assert status.completed
        assert status.progress == 0.8
        assert status.assigned_agents == ["agent1", "agent2"]
        assert status.start_time == 10
        assert status.completion_time == 50
        assert status.efficiency_score == 0.9


class TestTask:
    """Test abstract Task class."""
    
    def test_task_initialization(self):
        """Test task initialization."""
        # Create a concrete subclass for testing
        class TestTask(Task):
            def can_be_assigned(self, agent_capabilities):
                return agent_capabilities.get("test_capability", False)
            
            def update_progress(self, agent_actions):
                return 0.5
            
            def calculate_reward(self, completion_time, efficiency):
                return 10.0
        
        task = TestTask("test_task", priority=2.0)
        
        assert task.task_id == "test_task"
        assert task.priority == 2.0
        assert isinstance(task.status, TaskStatus)
        assert task.requirements == {}
        assert task.reward_structure == {}
    
    def test_agent_assignment(self):
        """Test agent assignment functionality."""
        class TestTask(Task):
            def can_be_assigned(self, agent_capabilities):
                return agent_capabilities.get("test_capability", False)
            
            def update_progress(self, agent_actions):
                return 0.5
            
            def calculate_reward(self, completion_time, efficiency):
                return 10.0
        
        task = TestTask("test_task")
        
        # Assign agent
        task.assign_agent("agent1")
        assert "agent1" in task.status.assigned_agents
        
        # Unassign agent
        task.unassign_agent("agent1")
        assert "agent1" not in task.status.assigned_agents
    
    def test_task_completion(self):
        """Test task completion status."""
        class TestTask(Task):
            def can_be_assigned(self, agent_capabilities):
                return True
            
            def update_progress(self, agent_actions):
                return 0.5
            
            def calculate_reward(self, completion_time, efficiency):
                return 10.0
        
        task = TestTask("test_task")
        
        assert not task.is_completed()
        task.status.completed = True
        assert task.is_completed()


class TestResourceCollectionTask:
    """Test ResourceCollectionTask class."""
    
    @pytest.fixture
    def mock_resources(self):
        """Create mock resources."""
        from src.environment.world import Resource
        return [
            Resource(100, 100, "food", 5.0, 2),
            Resource(200, 200, "mineral", 8.0, 1),
            Resource(300, 300, "energy", 10.0, 3)
        ]
    
    @pytest.fixture
    def collection_task(self, mock_resources):
        """Create a resource collection task."""
        return ResourceCollectionTask("collection_1", mock_resources)
    
    def test_collection_task_initialization(self, collection_task, mock_resources):
        """Test resource collection task initialization."""
        assert collection_task.task_id == "collection_1"
        assert collection_task.target_resources == mock_resources
        assert collection_task.collection_radius == 20.0
        assert collection_task.priority == 1.0
        
        # Check requirements
        assert collection_task.requirements["collection_capability"] is True
        assert collection_task.requirements["mobility"] is True
        
        # Check reward structure
        assert "base_reward" in collection_task.reward_structure
        assert "efficiency_bonus" in collection_task.reward_structure
        assert "speed_bonus" in collection_task.reward_structure
    
    def test_agent_assignment_check(self, collection_task):
        """Test agent assignment capability check."""
        # Agent with required capabilities
        capable_agent = {
            "collection_capability": True,
            "mobility": True
        }
        assert collection_task.can_be_assigned(capable_agent)
        
        # Agent missing capabilities
        incapable_agent = {
            "collection_capability": False,
            "mobility": True
        }
        assert not collection_task.can_be_assigned(incapable_agent)
    
    def test_progress_update(self, collection_task):
        """Test task progress update."""
        # Initially no resources collected
        progress = collection_task.update_progress([])
        assert progress == 0.0
        
        # Mark some resources as collected
        collection_task.target_resources[0].collected = True
        collection_task.target_resources[1].collected = True
        
        progress = collection_task.update_progress([])
        assert progress == 2/3  # 2 out of 3 resources collected
        
        # Mark all resources as collected
        collection_task.target_resources[2].collected = True
        progress = collection_task.update_progress([])
        assert progress == 1.0
        assert collection_task.status.completed
    
    def test_reward_calculation(self, collection_task):
        """Test reward calculation."""
        reward = collection_task.calculate_reward(completion_time=10, efficiency=0.8)
        
        # Base reward + efficiency bonus + speed bonus
        expected_reward = 10.0 + (5.0 * 0.8) + (3.0 / 10)
        assert abs(reward - expected_reward) < 0.01


class TestSearchAndRescueTask:
    """Test SearchAndRescueTask class."""
    
    @pytest.fixture
    def search_task(self):
        """Create a search and rescue task."""
        target_locations = [(100, 100), (200, 200), (300, 300)]
        return SearchAndRescueTask("search_1", target_locations)
    
    def test_search_task_initialization(self, search_task):
        """Test search and rescue task initialization."""
        assert search_task.task_id == "search_1"
        assert len(search_task.target_locations) == 3
        assert search_task.search_radius == 30.0
        assert search_task.priority == 2.0
        assert len(search_task.discovered_targets) == 0
        
        # Check requirements
        assert search_task.requirements["sensor_capability"] is True
        assert search_task.requirements["mobility"] is True
        assert search_task.requirements["communication"] is True
    
    def test_agent_assignment_check(self, search_task):
        """Test agent assignment capability check."""
        # Agent with required capabilities
        capable_agent = {
            "sensor_capability": True,
            "mobility": True
        }
        assert search_task.can_be_assigned(capable_agent)
        
        # Agent missing capabilities
        incapable_agent = {
            "sensor_capability": False,
            "mobility": True
        }
        assert not search_task.can_be_assigned(incapable_agent)
    
    def test_progress_update(self, search_task):
        """Test task progress update."""
        # Initially no targets discovered
        progress = search_task.update_progress([])
        assert progress == 0.0
        
        # Simulate discovering targets
        agent_actions = [
            {"discovered_target": 0},
            {"discovered_target": 2}
        ]
        
        progress = search_task.update_progress(agent_actions)
        assert progress == 2/3  # 2 out of 3 targets discovered
        assert 0 in search_task.discovered_targets
        assert 2 in search_task.discovered_targets
        
        # Discover all targets
        agent_actions = [{"discovered_target": 1}]
        progress = search_task.update_progress(agent_actions)
        assert progress == 1.0
        assert search_task.status.completed
    
    def test_reward_calculation(self, search_task):
        """Test reward calculation."""
        # Assign some agents
        search_task.assign_agent("agent1")
        search_task.assign_agent("agent2")
        
        reward = search_task.calculate_reward(completion_time=15, efficiency=0.9)
        
        # Base reward + discovery bonus + coordination bonus
        expected_reward = 15.0 + (8.0 * 0.9) + (5.0 * min(2/3, 1.0))
        assert abs(reward - expected_reward) < 0.01


class TestAreaCoverageTask:
    """Test AreaCoverageTask class."""
    
    @pytest.fixture
    def coverage_task(self):
        """Create an area coverage task."""
        target_area = (0, 0, 100, 100)  # 100x100 area
        return AreaCoverageTask("coverage_1", target_area)
    
    def test_coverage_task_initialization(self, coverage_task):
        """Test area coverage task initialization."""
        assert coverage_task.task_id == "coverage_1"
        assert coverage_task.target_area == (0, 0, 100, 100)
        assert coverage_task.coverage_threshold == 0.8
        assert coverage_task.priority == 1.5
        assert len(coverage_task.covered_positions) == 0
        
        # Check requirements
        assert coverage_task.requirements["mobility"] is True
        assert coverage_task.requirements["sensor_capability"] is True
    
    def test_agent_assignment_check(self, coverage_task):
        """Test agent assignment capability check."""
        # Agent with required capabilities
        capable_agent = {
            "mobility": True,
            "sensor_capability": True
        }
        assert coverage_task.can_be_assigned(capable_agent)
        
        # Agent missing capabilities
        incapable_agent = {
            "mobility": False,
            "sensor_capability": True
        }
        assert not coverage_task.can_be_assigned(incapable_agent)
    
    def test_progress_update(self, coverage_task):
        """Test task progress update."""
        # Initially no coverage
        progress = coverage_task.update_progress([])
        assert progress == 0.0
        
        # Simulate agent coverage
        agent_actions = [
            {"position": (25, 25)},
            {"position": (75, 75)}
        ]
        
        progress = coverage_task.update_progress(agent_actions)
        assert progress > 0.0  # Some coverage achieved
        
        # Simulate high coverage
        for x in range(0, 100, 10):
            for y in range(0, 100, 10):
                agent_actions.append({"position": (x, y)})
        
        progress = coverage_task.update_progress(agent_actions)
        assert progress >= coverage_task.coverage_threshold
        assert coverage_task.status.completed
    
    def test_reward_calculation(self, coverage_task):
        """Test reward calculation."""
        # Set some coverage
        coverage_task.status.progress = 0.7
        
        reward = coverage_task.calculate_reward(completion_time=20, efficiency=0.8)
        
        # Base reward + coverage bonus + efficiency bonus
        expected_reward = 12.0 + (6.0 * 0.7) + (4.0 * 0.8)
        assert abs(reward - expected_reward) < 0.01


class TestTaskManager:
    """Test TaskManager class."""
    
    @pytest.fixture
    def task_manager(self):
        """Create a task manager."""
        return TaskManager()
    
    @pytest.fixture
    def mock_resources(self):
        """Create mock resources."""
        from src.environment.world import Resource
        return [
            Resource(100, 100, "food", 5.0, 2),
            Resource(200, 200, "mineral", 8.0, 1)
        ]
    
    def test_task_manager_initialization(self, task_manager):
        """Test task manager initialization."""
        assert len(task_manager.tasks) == 0
        assert len(task_manager.completed_tasks) == 0
        assert task_manager.task_counter == 0
    
    def test_create_resource_collection_task(self, task_manager, mock_resources):
        """Test creating resource collection tasks."""
        task = task_manager.create_resource_collection_task(mock_resources)
        
        assert task.task_id == "resource_collection_0"
        assert task.target_resources == mock_resources
        assert task in task_manager.tasks
        assert task_manager.task_counter == 1
    
    def test_create_search_rescue_task(self, task_manager):
        """Test creating search and rescue tasks."""
        target_locations = [(100, 100), (200, 200)]
        task = task_manager.create_search_rescue_task(target_locations)
        
        assert task.task_id == "search_rescue_0"
        assert task.target_locations == target_locations
        assert task in task_manager.tasks
    
    def test_create_area_coverage_task(self, task_manager):
        """Test creating area coverage tasks."""
        target_area = (0, 0, 200, 200)
        task = task_manager.create_area_coverage_task(target_area)
        
        assert task.task_id == "area_coverage_0"
        assert task.target_area == target_area
        assert task in task_manager.tasks
    
    def test_get_available_tasks(self, task_manager, mock_resources):
        """Test getting available tasks for agents."""
        # Create tasks
        collection_task = task_manager.create_resource_collection_task(mock_resources)
        search_task = task_manager.create_search_rescue_task([(100, 100)])
        
        # Agent with collection capabilities
        collection_agent = {
            "collection_capability": True,
            "mobility": True
        }
        available_tasks = task_manager.get_available_tasks(collection_agent)
        assert collection_task in available_tasks
        assert search_task not in available_tasks
        
        # Agent with sensor capabilities
        sensor_agent = {
            "sensor_capability": True,
            "mobility": True
        }
        available_tasks = task_manager.get_available_tasks(sensor_agent)
        assert search_task in available_tasks
    
    def test_get_high_priority_tasks(self, task_manager):
        """Test getting tasks ordered by priority."""
        # Create tasks with different priorities
        task1 = task_manager.create_resource_collection_task([], priority=1.0)
        task2 = task_manager.create_search_rescue_task([], priority=3.0)
        task3 = task_manager.create_area_coverage_task((0, 0, 100, 100), priority=2.0)
        
        high_priority_tasks = task_manager.get_high_priority_tasks()
        
        # Should be ordered by priority (descending)
        assert high_priority_tasks[0] == task2  # Priority 3.0
        assert high_priority_tasks[1] == task3  # Priority 2.0
        assert high_priority_tasks[2] == task1  # Priority 1.0
    
    def test_update_all_tasks(self, task_manager, mock_resources):
        """Test updating all tasks."""
        # Create a task
        task = task_manager.create_resource_collection_task(mock_resources)
        task.assign_agent("agent1")
        
        # Mark some resources as collected
        mock_resources[0].collected = True
        
        # Update tasks
        agent_actions = {"agent1": [{"action": "collect"}]}
        task_manager.update_all_tasks(agent_actions)
        
        # Task should be updated
        assert task.status.progress > 0.0
    
    def test_task_statistics(self, task_manager, mock_resources):
        """Test getting task statistics."""
        # Initially no tasks
        stats = task_manager.get_task_statistics()
        assert stats["total_tasks"] == 0
        assert stats["completion_rate"] == 0.0
        
        # Create and complete a task
        task = task_manager.create_resource_collection_task(mock_resources)
        task.status.completed = True
        task_manager.completed_tasks.append(task)
        task_manager.tasks.remove(task)
        
        stats = task_manager.get_task_statistics()
        assert stats["total_tasks"] == 1
        assert stats["completed_tasks"] == 1
        assert stats["completion_rate"] == 1.0


class TestSwarmEnvironment:
    """Test SwarmEnvironment class."""
    
    @pytest.fixture
    def mock_world(self):
        """Create a mock world."""
        world = Mock()
        world.width = 800
        world.height = 600
        world.resources = []
        world.get_world_state.return_value = {
            "width": 800,
            "height": 600,
            "time_step": 0
        }
        return world
    
    @pytest.fixture
    def mock_swarm(self):
        """Create a mock swarm."""
        swarm = Mock()
        swarm.agents = []
        swarm.get_swarm_state.return_value = {
            "num_agents": 0,
            "active_agents": 0
        }
        swarm.step.return_value = {}
        swarm.reset.return_value = None
        swarm.close.return_value = None
        return swarm
    
    @patch('src.environment.swarm_environment.HeterogeneousSwarm')
    def test_environment_initialization(self, mock_swarm_class, mock_world):
        """Test swarm environment initialization."""
        mock_swarm_class.return_value = Mock()
        
        env = SwarmEnvironment(
            world_size=(800, 600),
            num_agents=10,
            agent_types=["explorer", "collector"]
        )
        
        assert env.world_size == (800, 600)
        assert env.num_agents == 10
        assert env.agent_types == ["explorer", "collector"]
        assert env.current_step == 0
        assert not env.simulation_running
    
    @patch('src.environment.swarm_environment.HeterogeneousSwarm')
    def test_environment_reset(self, mock_swarm_class, mock_world):
        """Test environment reset functionality."""
        mock_swarm_class.return_value = Mock()
        
        env = SwarmEnvironment(world_size=(800, 600))
        
        # Modify environment state
        env.current_step = 100
        env.simulation_running = True
        
        # Reset environment
        env.reset()
        
        assert env.current_step == 0
        assert not env.simulation_running
        assert len(env.episode_rewards) == 0
    
    @patch('src.environment.swarm_environment.HeterogeneousSwarm')
    def test_environment_start_stop(self, mock_swarm_class, mock_world):
        """Test environment start and stop functionality."""
        mock_swarm_class.return_value = Mock()
        
        env = SwarmEnvironment(world_size=(800, 600))
        
        # Start simulation
        env.start_simulation()
        assert env.simulation_running
        assert env.current_step == 0
        
        # Stop simulation
        env.stop_simulation()
        assert not env.simulation_running
    
    @patch('src.environment.swarm_environment.HeterogeneousSwarm')
    def test_environment_step(self, mock_swarm_class, mock_world):
        """Test environment step execution."""
        mock_swarm = Mock()
        mock_swarm.step.return_value = {"agent1": [{"action": "move"}]}
        mock_swarm.agents = [Mock(), Mock(), Mock()]  # Mock agents
        for agent in mock_swarm.agents:
            agent.is_active.return_value = True
            agent.energy = 0.8
            agent.position = (100, 100)
        mock_swarm_class.return_value = mock_swarm
        
        env = SwarmEnvironment(world_size=(800, 600))
        env.start_simulation()
        
        # Execute step
        step_result = env.step()
        
        assert "step" in step_result
        assert "rewards" in step_result
        assert "task_progress" in step_result
        assert "swarm_metrics" in step_result
        assert "done" in step_result
        
        assert env.current_step == 1
    
    @patch('src.environment.swarm_environment.HeterogeneousSwarm')
    def test_environment_state(self, mock_swarm_class, mock_world):
        """Test getting environment state."""
        mock_swarm = Mock()
        mock_swarm.get_swarm_state.return_value = {"num_agents": 10}
        mock_swarm.agents = [Mock(), Mock(), Mock()]  # Mock agents
        for agent in mock_swarm.agents:
            agent.is_active.return_value = True
            agent.energy = 0.8
            agent.position = (100, 100)
        mock_swarm_class.return_value = mock_swarm
        
        env = SwarmEnvironment(world_size=(800, 600))
        state = env.get_environment_state()
        
        assert "world_state" in state
        assert "task_statistics" in state
        assert "swarm_metrics" in state
        assert "performance_metrics" in state
        assert "current_step" in state
        assert "simulation_running" in state
    
    @patch('src.environment.swarm_environment.HeterogeneousSwarm')
    def test_environment_termination(self, mock_swarm_class, mock_world):
        """Test environment termination conditions."""
        mock_swarm = Mock()
        mock_swarm.step.return_value = {}
        mock_swarm.agents = [Mock(), Mock(), Mock()]  # Mock agents
        for agent in mock_swarm.agents:
            agent.is_active.return_value = True
            agent.energy = 0.8
            agent.position = (100, 100)
        mock_swarm_class.return_value = mock_swarm
        
        env = SwarmEnvironment(world_size=(800, 600), max_steps=5)
        env.start_simulation()
        
        # Run until termination
        for _ in range(10):
            step_result = env.step()
            if step_result.get("done", False):
                break
        
        # Should terminate after max_steps
        assert env.current_step >= 5
