"""
Multi-Agent Deep Q-Network (DQN) implementation for SwarmMindAI.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random


class DQNetwork(nn.Module):
    """Neural network for DQN value function approximation."""
    
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize the DQN network.
        
        Args:
            input_dim: Dimension of input observations
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
        """
        super(DQNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Q-values for all actions
        """
        return self.network(x)


class MultiAgentDQN:
    """
    Multi-Agent Deep Q-Network implementation.
    
    Features:
    - Experience replay with prioritized sampling
    - Target network for stable training
    - Multi-agent coordination through shared experience
    - Adaptive exploration strategies
    """
    
    def __init__(self,
                 observation_dim: int,
                 action_dim: int,
                 num_agents: int = 20,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 100,
                 device: str = "cpu"):
        """
        Initialize Multi-Agent DQN.
        
        Args:
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
            num_agents: Number of agents in the swarm
            learning_rate: Learning rate for optimization
            gamma: Discount factor
            epsilon_start: Starting exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            memory_size: Size of experience replay buffer
            batch_size: Training batch size
            target_update_freq: Frequency of target network updates
            device: Device to run computations on
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        
        # Initialize networks
        self.q_network = DQNetwork(observation_dim, action_dim).to(device)
        self.target_network = DQNetwork(observation_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        self.priorities = deque(maxlen=memory_size)
        
        # Training parameters
        self.epsilon = epsilon_start
        self.step_count = 0
        self.update_count = 0
        
        # Performance tracking
        self.training_losses = []
        self.epsilon_history = []
        self.q_values_history = []
        
        # Multi-agent coordination
        self.shared_experience = True
        self.coordination_bonus = 0.1
        self.agent_coordination = {}
    
    def select_action(self, observation: np.ndarray, agent_id: str) -> Tuple[int, float]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            observation: Current observation
            agent_id: ID of the agent
            
        Returns:
            Tuple of (action, q_value)
        """
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            q_value = 0.0
        else:
            # Get Q-values from network
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(obs_tensor)
                action = q_values.argmax().item()
                q_value = q_values.max().item()
        
        return action, q_value
    
    def store_experience(self,
                        observation: np.ndarray,
                        action: int,
                        reward: float,
                        next_observation: np.ndarray,
                        done: bool,
                        agent_id: str):
        """
        Store experience in replay buffer.
        
        Args:
            observation: Current observation
            action: Taken action
            reward: Received reward
            next_observation: Next observation
            done: Episode done flag
            agent_id: ID of the agent
        """
        # Add coordination bonus to reward
        if self.shared_experience:
            reward += self._compute_coordination_bonus(agent_id, action)
        
        # Store experience
        experience = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "done": done,
            "agent_id": agent_id
        }
        
        self.memory.append(experience)
        self.priorities.append(1.0)  # Default priority
        
        # Update agent coordination tracking
        if agent_id not in self.agent_coordination:
            self.agent_coordination[agent_id] = []
        self.agent_coordination[agent_id].append({
            "action": action,
            "reward": reward,
            "step": self.step_count
        })
    
    def _compute_coordination_bonus(self, agent_id: str, action: int) -> float:
        """Compute coordination bonus based on agent behavior."""
        if agent_id not in self.agent_coordination:
            return 0.0
        
        recent_actions = self.agent_coordination[agent_id][-10:]  # Last 10 actions
        
        # Check for coordinated behavior patterns
        coordination_score = 0.0
        
        for action_data in recent_actions:
            if action_data["action"] == action:
                coordination_score += 0.1
            if action_data["reward"] > 0:
                coordination_score += 0.05
        
        return coordination_score * self.coordination_bonus
    
    def update_policy(self) -> Dict[str, float]:
        """
        Update the Q-network using experience replay.
        
        Returns:
            Dictionary containing training loss
        """
        if len(self.memory) < self.batch_size:
            return {}
        
        # Sample batch from memory
        batch_indices = self._sample_batch()
        batch = [self.memory[i] for i in batch_indices]
        
        # Prepare batch data
        observations = torch.FloatTensor([exp["observation"] for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp["action"] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp["reward"] for exp in batch]).to(self.device)
        next_observations = torch.FloatTensor([exp["next_observation"] for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp["done"] for exp in batch]).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(observations)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_observations)
            max_next_q = next_q_values.max(1)[0]
            target_q = rewards + (self.gamma * max_next_q * ~dones)
        
        # Compute loss
        loss = self.criterion(current_q, target_q)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update priorities based on TD error
        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
        for i, idx in enumerate(batch_indices):
            self.priorities[idx] = td_errors[i] + 1e-6  # Add small constant to avoid zero priority
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Update exploration rate
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Store training statistics
        self.training_losses.append(loss.item())
        self.epsilon_history.append(self.epsilon)
        
        return {"loss": loss.item(), "epsilon": self.epsilon}
    
    def _sample_batch(self) -> List[int]:
        """Sample batch indices using prioritized experience replay."""
        if len(self.memory) <= self.batch_size:
            return list(range(len(self.memory)))
        
        # Convert priorities to numpy array
        priorities = np.array(self.priorities)
        
        # Compute sampling probabilities
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.memory), self.batch_size, p=probabilities)
        return indices.tolist()
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "training_losses": self.training_losses[-100:] if self.training_losses else [],
            "epsilon_history": self.epsilon_history[-100:] if self.epsilon_history else [],
            "q_values_history": self.q_values_history[-100:] if self.q_values_history else [],
            "avg_loss": np.mean(self.training_losses[-100:]) if self.training_losses else 0.0,
            "current_epsilon": self.epsilon,
            "memory_size": len(self.memory),
            "update_count": self.update_count
        }
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_stats": self.get_training_stats(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "update_count": self.update_count
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "training_stats" in checkpoint:
            stats = checkpoint["training_stats"]
            self.training_losses = stats.get("training_losses", [])
            self.epsilon_history = stats.get("epsilon_history", [])
            self.q_values_history = stats.get("q_values_history", [])
        
        if "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]
        if "step_count" in checkpoint:
            self.step_count = checkpoint["step_count"]
        if "update_count" in checkpoint:
            self.update_count = checkpoint["update_count"]
    
    def reset(self):
        """Reset the algorithm state."""
        self.memory.clear()
        self.priorities.clear()
        self.agent_coordination.clear()
        self.training_losses.clear()
        self.epsilon_history.clear()
        self.q_values_history.clear()
        self.epsilon = self.epsilon_start
        self.step_count = 0
        self.update_count = 0
        
        # Reset target network
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_q_values(self, observation: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions given an observation."""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
        
        return q_values.squeeze(0).cpu().numpy()
    
    def compute_coordination_metrics(self) -> Dict[str, float]:
        """Compute coordination metrics across all agents."""
        if not self.agent_coordination:
            return {"coordination_score": 0.0, "action_diversity": 0.0}
        
        # Calculate coordination score
        total_actions = 0
        coordinated_actions = 0
        
        for agent_actions in self.agent_coordination.values():
            for action_data in agent_actions:
                total_actions += 1
                if action_data["reward"] > 0:
                    coordinated_actions += 1
        
        coordination_score = coordinated_actions / max(total_actions, 1)
        
        # Calculate action diversity
        all_actions = []
        for agent_actions in self.agent_coordination.values():
            all_actions.extend([a["action"] for a in agent_actions])
        
        action_diversity = len(set(all_actions)) / max(len(all_actions), 1)
        
        return {
            "coordination_score": coordination_score,
            "action_diversity": action_diversity
        }
