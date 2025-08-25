"""
Main simulation environment for SwarmMindAI.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
from .world import World
from .tasks import TaskManager, Task
from ..agents import BaseAgent, HeterogeneousSwarm


class SwarmEnvironment:
    """
    Main simulation environment that coordinates the world, tasks, and agents.
    
    Features:
    - Dynamic task generation and management
    - Multi-agent coordination and communication
    - Real-time performance monitoring
    - Configurable simulation parameters
    """
    
    def __init__(self, world_size: Tuple[int, int] = (1000, 1000), 
                 num_agents: int = 20, agent_types: List[str] = None,
                 seed: Optional[int] = None, max_steps: int = 1000):
        """
        Initialize the swarm environment.
        
        Args:
            world_size: (width, height) of the simulation world
            num_agents: Number of agents in the swarm
            agent_types: List of agent types to create
            seed: Random seed for reproducibility
            max_steps: Maximum number of simulation steps
        """
        self.world_size = world_size
        self.num_agents = num_agents
        self.agent_types = agent_types or ["explorer", "collector", "coordinator"]
        self.seed = seed
        self.max_steps = max_steps
        
        # Initialize components
        self.world = World(world_size[0], world_size[1], seed)
        self.task_manager = TaskManager()
        self.swarm = HeterogeneousSwarm(
            world=self.world,
            num_agents=num_agents,
            agent_types=self.agent_types
        )
        
        # Simulation state
        self.current_step = 0
        self.simulation_running = False
        self.episode_rewards = []
        self.performance_metrics = {}
        
        # Communication and coordination
        self.agent_actions: Dict[str, List[Dict]] = {}
        self.global_messages: List[Dict] = []
        
        # Initialize tasks
        self._initialize_tasks()
    
    def _initialize_tasks(self):
        """Initialize tasks for the simulation."""
        # Create resource collection tasks
        available_resources = [r for r in self.world.resources if not r.collected]
        if available_resources:
            # Group resources by type for efficient collection
            resources_by_type = {}
            for resource in available_resources:
                if resource.resource_type not in resources_by_type:
                    resources_by_type[resource.resource_type] = []
                resources_by_type[resource.resource_type].append(resource)
            
            for resource_type, resources in resources_by_type.items():
                task = self.task_manager.create_resource_collection_task(
                    resources, priority=1.0
                )
        
        # Create search and rescue tasks
        target_locations = [
            (100, 100), (900, 100), (100, 900), (900, 900),
            (500, 500), (300, 700), (700, 300)
        ]
        search_task = self.task_manager.create_search_rescue_task(
            target_locations, priority=2.0
        )
        
        # Create area coverage tasks
        coverage_areas = [
            (0, 0, 500, 500),      # Top-left quadrant
            (500, 0, 1000, 500),   # Top-right quadrant
            (0, 500, 500, 1000),   # Bottom-left quadrant
            (500, 500, 1000, 1000) # Bottom-right quadrant
        ]
        
        for area in coverage_areas:
            coverage_task = self.task_manager.create_area_coverage_task(
                area, priority=1.5
            )
    
    def reset(self):
        """Reset the environment to initial state."""
        self.current_step = 0
        self.simulation_running = False
        self.episode_rewards = []
        self.performance_metrics = {}
        self.agent_actions.clear()
        self.global_messages.clear()
        
        # Reset components
        self.world.reset()
        self.task_manager = TaskManager()
        self.swarm.reset()
        
        # Reinitialize tasks
        self._initialize_tasks()
    
    def step(self) -> Dict[str, Any]:
        """
        Execute one simulation step.
        
        Returns:
            Dictionary containing step information and metrics
        """
        if not self.simulation_running:
            return {}
        
        self.current_step += 1
        
        # Update world
        self.world.update()
        
        # Get agent actions
        agent_actions = self.swarm.step()
        self.agent_actions = agent_actions
        
        # Update tasks based on agent actions
        self.task_manager.update_all_tasks(agent_actions)
        
        # Calculate rewards and update metrics
        step_rewards = self._calculate_step_rewards()
        self.episode_rewards.append(step_rewards)
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Check termination conditions
        done = self._check_termination()
        
        # Generate new tasks if needed
        if self.current_step % 100 == 0:
            self._generate_new_tasks()
        
        return {
            "step": self.current_step,
            "rewards": step_rewards,
            "task_progress": self._get_task_progress(),
            "swarm_metrics": self._get_swarm_metrics(),
            "done": done
        }
    
    def _calculate_step_rewards(self) -> Dict[str, float]:
        """Calculate rewards for the current step."""
        step_rewards = {}
        
        # Calculate rewards for each agent
        for agent_id, actions in self.agent_actions.items():
            agent_reward = 0.0
            
            # Reward for successful actions
            for action in actions:
                if "resource_collected" in action:
                    agent_reward += 5.0
                if "target_discovered" in action:
                    agent_reward += 3.0
                if "area_covered" in action:
                    agent_reward += 2.0
                if "collision_avoided" in action:
                    agent_reward += 1.0
            
            # Penalty for collisions
            if any("collision" in action for action in actions):
                agent_reward -= 10.0
            
            # Efficiency bonus
            if len(actions) > 0:
                efficiency = len([a for a in actions if "success" in a]) / len(actions)
                agent_reward += efficiency * 2.0
            
            step_rewards[agent_id] = agent_reward
        
        return step_rewards
    
    def _update_performance_metrics(self):
        """Update performance metrics for the simulation."""
        # Task completion metrics
        task_stats = self.task_manager.get_task_statistics()
        
        # Swarm efficiency metrics
        total_rewards = sum(sum(rewards.values()) for rewards in self.episode_rewards[-100:])
        avg_reward_per_step = total_rewards / min(100, len(self.episode_rewards))
        
        # Agent coordination metrics
        active_agents = len([a for a in self.swarm.agents if a.is_active()])
        coordination_score = self._calculate_coordination_score()
        
        self.performance_metrics = {
            "task_completion_rate": task_stats["completion_rate"],
            "active_tasks": task_stats["active_tasks"],
            "total_tasks": task_stats["total_tasks"],
            "avg_reward_per_step": avg_reward_per_step,
            "active_agents": active_agents,
            "coordination_score": coordination_score,
            "current_step": self.current_step
        }
    
    def _calculate_coordination_score(self) -> float:
        """Calculate how well agents are coordinating."""
        if not self.agent_actions:
            return 0.0
        
        # Count coordinated actions
        coordinated_actions = 0
        total_actions = 0
        
        for agent_id, actions in self.agent_actions.items():
            for action in actions:
                total_actions += 1
                if "coordinated" in action:
                    coordinated_actions += 1
        
        return coordinated_actions / max(total_actions, 1)
    
    def _get_task_progress(self) -> Dict[str, float]:
        """Get progress of all active tasks."""
        progress = {}
        for task in self.task_manager.tasks:
            progress[task.task_id] = task.status.progress
        return progress
    
    def _get_swarm_metrics(self) -> Dict[str, Any]:
        """Get current swarm performance metrics."""
        return {
            "num_agents": len(self.swarm.agents),
            "active_agents": len([a for a in self.swarm.agents if a.is_active()]),
            "avg_energy": np.mean([a.energy for a in self.swarm.agents]),
            "avg_position": np.mean([a.position for a in self.swarm.agents], axis=0).tolist(),
            "swarm_density": self._calculate_swarm_density()
        }
    
    def _calculate_swarm_density(self) -> float:
        """Calculate how clustered the swarm is."""
        if len(self.swarm.agents) < 2:
            return 0.0
        
        positions = np.array([a.position for a in self.swarm.agents])
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        avg_distance = np.mean(distances)
        
        # Normalize by world size
        world_diagonal = np.sqrt(self.world_size[0]**2 + self.world_size[1]**2)
        return 1.0 - (avg_distance / world_diagonal)
    
    def _check_termination(self) -> bool:
        """Check if the simulation should terminate."""
        # Check step limit
        if self.current_step >= self.max_steps:
            return True
        
        # Check if all tasks are completed
        if len(self.task_manager.tasks) == 0:
            return True
        
        # Check if all agents are inactive
        if all(not agent.is_active() for agent in self.swarm.agents):
            return True
        
        return False
    
    def _generate_new_tasks(self):
        """Generate new tasks during simulation."""
        # Add new resources
        if random.random() < 0.1:  # 10% chance per 100 steps
            x = random.uniform(50, self.world_size[0] - 50)
            y = random.uniform(50, self.world_size[1] - 50)
            resource_type = random.choice(["food", "mineral", "energy", "water"])
            value = random.uniform(1.0, 10.0)
            quantity = random.randint(1, 3)
            
            from .world import Resource
            new_resource = Resource(x, y, resource_type, value, quantity)
            self.world.resources.append(new_resource)
            
            # Create task for the new resource
            task = self.task_manager.create_resource_collection_task(
                [new_resource], priority=1.0
            )
    
    def start_simulation(self):
        """Start the simulation."""
        self.simulation_running = True
        self.current_step = 0
    
    def stop_simulation(self):
        """Stop the simulation."""
        self.simulation_running = False
    
    def get_environment_state(self) -> Dict[str, Any]:
        """Get the current state of the environment."""
        return {
            "world_state": self.world.get_world_state(),
            "task_statistics": self.task_manager.get_task_statistics(),
            "swarm_metrics": self._get_swarm_metrics(),
            "performance_metrics": self.performance_metrics,
            "current_step": self.current_step,
            "simulation_running": self.simulation_running
        }
    
    def render(self, mode: str = "human"):
        """Render the environment."""
        if mode == "human":
            # This will be implemented in the visualization module
            pass
        elif mode == "rgb_array":
            # Return RGB array for headless rendering
            return np.zeros((self.world_size[1], self.world_size[0], 3), dtype=np.uint8)
    
    def close(self):
        """Clean up resources."""
        self.stop_simulation()
        self.swarm.close()


# Import at the end to avoid circular imports
import random
