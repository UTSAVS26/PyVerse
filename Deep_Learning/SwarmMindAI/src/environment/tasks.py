"""
Task definitions for SwarmMindAI simulation environment.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from .world import Resource, Obstacle


@dataclass
class TaskStatus:
    """Status information for a task."""
    completed: bool = False
    progress: float = 0.0
    assigned_agents: List[str] = None
    start_time: int = 0
    completion_time: Optional[int] = None
    efficiency_score: float = 0.0
    
    def __post_init__(self):
        if self.assigned_agents is None:
            self.assigned_agents = []


class Task(ABC):
    """Abstract base class for all tasks."""
    
    def __init__(self, task_id: str, priority: float = 1.0):
        """
        Initialize a task.
        
        Args:
            task_id: Unique identifier for the task
            priority: Task priority (higher = more important)
        """
        self.task_id = task_id
        self.priority = priority
        self.status = TaskStatus()
        self.requirements = {}
        self.reward_structure = {}
    
    @abstractmethod
    def can_be_assigned(self, agent_capabilities: Dict) -> bool:
        """Check if an agent can be assigned to this task."""
        pass
    
    @abstractmethod
    def update_progress(self, agent_actions: List[Dict]) -> float:
        """Update task progress based on agent actions."""
        pass
    
    @abstractmethod
    def calculate_reward(self, completion_time: int, efficiency: float) -> float:
        """Calculate reward for task completion."""
        pass
    
    def assign_agent(self, agent_id: str):
        """Assign an agent to this task."""
        if agent_id not in self.status.assigned_agents:
            self.status.assigned_agents.append(agent_id)
    
    def unassign_agent(self, agent_id: str):
        """Unassign an agent from this task."""
        if agent_id in self.status.assigned_agents:
            self.status.assigned_agents.remove(agent_id)
    
    def is_completed(self) -> bool:
        """Check if the task is completed."""
        return self.status.completed


class ResourceCollectionTask(Task):
    """Task for collecting resources from the environment."""
    
    def __init__(self, task_id: str, target_resources: List[Resource], 
                 collection_radius: float = 20.0, priority: float = 1.0):
        """
        Initialize a resource collection task.
        
        Args:
            task_id: Unique identifier for the task
            target_resources: List of resources to collect
            collection_radius: Radius within which resources can be collected
            priority: Task priority
        """
        super().__init__(task_id, priority)
        self.target_resources = target_resources
        self.collection_radius = collection_radius
        self.requirements = {
            "collection_capability": True,
            "mobility": True
        }
        self.reward_structure = {
            "base_reward": 10.0,
            "efficiency_bonus": 5.0,
            "speed_bonus": 3.0
        }
    
    def can_be_assigned(self, agent_capabilities: Dict) -> bool:
        """Check if an agent can be assigned to this task."""
        return (agent_capabilities.get("collection_capability", False) and 
                agent_capabilities.get("mobility", False))
    
    def update_progress(self, agent_actions: List[Dict]) -> float:
        """Update task progress based on agent actions."""
        collected_count = 0
        total_count = len(self.target_resources)
        
        for resource in self.target_resources:
            if resource.collected:
                collected_count += 1
        
        progress = collected_count / total_count
        self.status.progress = progress
        
        if progress >= 1.0:
            self.status.completed = True
            self.status.completion_time = self.status.start_time
        
        return progress
    
    def calculate_reward(self, completion_time: int, efficiency: float) -> float:
        """Calculate reward for task completion."""
        base_reward = self.reward_structure["base_reward"]
        efficiency_bonus = self.reward_structure["efficiency_bonus"] * efficiency
        speed_bonus = self.reward_structure["speed_bonus"] / max(completion_time, 1)
        
        return base_reward + efficiency_bonus + speed_bonus


class SearchAndRescueTask(Task):
    """Task for searching and rescuing targets in the environment."""
    
    def __init__(self, task_id: str, target_locations: List[Tuple[float, float]], 
                 search_radius: float = 30.0, priority: float = 2.0):
        """
        Initialize a search and rescue task.
        
        Args:
            task_id: Unique identifier for the task
            target_locations: List of target locations to search
            search_radius: Radius within which targets can be detected
            priority: Task priority
        """
        super().__init__(task_id, priority)
        self.target_locations = target_locations
        self.search_radius = search_radius
        self.discovered_targets = set()
        self.requirements = {
            "sensor_capability": True,
            "mobility": True,
            "communication": True
        }
        self.reward_structure = {
            "base_reward": 15.0,
            "discovery_bonus": 8.0,
            "coordination_bonus": 5.0
        }
    
    def can_be_assigned(self, agent_capabilities: Dict) -> bool:
        """Check if an agent can be assigned to this task."""
        return (agent_capabilities.get("sensor_capability", False) and 
                agent_capabilities.get("mobility", False))
    
    def update_progress(self, agent_actions: List[Dict]) -> float:
        """Update task progress based on agent actions."""
        total_count = len(self.target_locations)
        
        # Check if agents discovered new targets
        for action in agent_actions:
            if "discovered_target" in action:
                target_idx = action["discovered_target"]
                if target_idx < total_count:
                    self.discovered_targets.add(target_idx)
        
        # Calculate progress after updating discovered targets
        discovered_count = len(self.discovered_targets)
        progress = discovered_count / total_count
        self.status.progress = progress
        
        if progress >= 1.0:
            self.status.completed = True
            self.status.completion_time = self.status.start_time
        
        return progress
    
    def calculate_reward(self, completion_time: int, efficiency: float) -> float:
        """Calculate reward for task completion."""
        base_reward = self.reward_structure["base_reward"]
        discovery_bonus = self.reward_structure["discovery_bonus"] * efficiency
        coordination_bonus = self.reward_structure["coordination_bonus"] * min(
            len(self.status.assigned_agents) / 3.0, 1.0
        )
        
        return base_reward + discovery_bonus + coordination_bonus


class AreaCoverageTask(Task):
    """Task for covering a specific area of the environment."""
    
    def __init__(self, task_id: str, target_area: Tuple[float, float, float, float], 
                 coverage_threshold: float = 0.8, priority: float = 1.5):
        """
        Initialize an area coverage task.
        
        Args:
            task_id: Unique identifier for the task
            target_area: (x1, y1, x2, y2) defining the area to cover
            coverage_threshold: Minimum coverage percentage required
            priority: Task priority
        """
        super().__init__(task_id, priority)
        self.target_area = target_area
        self.coverage_threshold = coverage_threshold
        self.covered_positions = set()
        self.requirements = {
            "mobility": True,
            "sensor_capability": True
        }
        self.reward_structure = {
            "base_reward": 12.0,
            "coverage_bonus": 6.0,
            "efficiency_bonus": 4.0
        }
    
    def can_be_assigned(self, agent_capabilities: Dict) -> bool:
        """Check if an agent can be assigned to this task."""
        return (agent_capabilities.get("mobility", False) and 
                agent_capabilities.get("sensor_capability", False))
    
    def update_progress(self, agent_actions: List[Dict]) -> float:
        """Update task progress based on agent actions."""
        # Calculate coverage based on agent positions
        x1, y1, x2, y2 = self.target_area
        area_width = x2 - x1
        area_height = y2 - y1
        total_area = area_width * area_height
        
        # Grid-based coverage calculation
        grid_size = 10
        covered_cells = 0
        total_cells = 0
        
        for x in np.arange(x1, x2, grid_size):
            for y in np.arange(y1, y2, grid_size):
                total_cells += 1
                # Check if any agent is near this cell
                for action in agent_actions:
                    if "position" in action:
                        ax, ay = action["position"]
                        if (abs(ax - x) < grid_size and abs(ay - y) < grid_size):
                            covered_cells += 1
                            break
        
        coverage = covered_cells / total_cells if total_cells > 0 else 0.0
        self.status.progress = coverage
        
        if coverage >= self.coverage_threshold:
            self.status.completed = True
            self.status.completion_time = self.status.start_time
        
        return coverage
    
    def calculate_reward(self, completion_time: int, efficiency: float) -> float:
        """Calculate reward for task completion."""
        base_reward = self.reward_structure["base_reward"]
        coverage_bonus = self.reward_structure["coverage_bonus"] * self.status.progress
        efficiency_bonus = self.reward_structure["efficiency_bonus"] * efficiency
        
        return base_reward + coverage_bonus + efficiency_bonus


class TaskManager:
    """Manages task creation, assignment, and monitoring."""
    
    def __init__(self):
        """Initialize the task manager."""
        self.tasks: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.task_counter = 0
    
    def create_resource_collection_task(self, resources: List[Resource], 
                                      priority: float = 1.0) -> ResourceCollectionTask:
        """Create a new resource collection task."""
        task_id = f"resource_collection_{self.task_counter}"
        self.task_counter += 1
        
        task = ResourceCollectionTask(task_id, resources, priority=priority)
        self.tasks.append(task)
        return task
    
    def create_search_rescue_task(self, target_locations: List[Tuple[float, float]], 
                                priority: float = 2.0) -> SearchAndRescueTask:
        """Create a new search and rescue task."""
        task_id = f"search_rescue_{self.task_counter}"
        self.task_counter += 1
        
        task = SearchAndRescueTask(task_id, target_locations, priority=priority)
        self.tasks.append(task)
        return task
    
    def create_area_coverage_task(self, target_area: Tuple[float, float, float, float], 
                                priority: float = 1.5) -> AreaCoverageTask:
        """Create a new area coverage task."""
        task_id = f"area_coverage_{self.task_counter}"
        self.task_counter += 1
        
        task = AreaCoverageTask(task_id, target_area, priority=priority)
        self.tasks.append(task)
        return task
    
    def get_available_tasks(self, agent_capabilities: Dict) -> List[Task]:
        """Get tasks that can be assigned to an agent."""
        return [task for task in self.tasks 
                if not task.is_completed() and task.can_be_assigned(agent_capabilities)]
    
    def get_high_priority_tasks(self) -> List[Task]:
        """Get tasks ordered by priority."""
        return sorted(self.tasks, key=lambda t: t.priority, reverse=True)
    
    def update_all_tasks(self, agent_actions: Dict[str, List[Dict]]):
        """Update all tasks based on agent actions."""
        for task in self.tasks:
            if not task.is_completed():
                # Collect actions from agents assigned to this task
                relevant_actions = []
                for agent_id in task.status.assigned_agents:
                    if agent_id in agent_actions:
                        relevant_actions.extend(agent_actions[agent_id])
                
                task.update_progress(relevant_actions)
                
                # Move completed tasks
                if task.is_completed():
                    self.completed_tasks.append(task)
                    self.tasks.remove(task)
    
    def get_task_statistics(self) -> Dict:
        """Get statistics about all tasks."""
        total_tasks = len(self.tasks) + len(self.completed_tasks)
        completed_count = len(self.completed_tasks)
        completion_rate = completed_count / total_tasks if total_tasks > 0 else 0.0
        
        return {
            "total_tasks": total_tasks,
            "active_tasks": len(self.tasks),
            "completed_tasks": completed_count,
            "completion_rate": completion_rate,
            "average_priority": np.mean([t.priority for t in self.tasks]) if self.tasks else 0.0
        }
