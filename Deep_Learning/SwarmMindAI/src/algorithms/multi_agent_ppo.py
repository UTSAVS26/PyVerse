"""
Multi-Agent Proximal Policy Optimization (PPO) implementation for SwarmMindAI.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random


class PPONetwork(nn.Module):
    """Neural network for PPO policy and value functions."""
    
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize the PPO network.
        
        Args:
            input_dim: Dimension of input observations
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
        """
        super(PPONetwork, self).__init__()
        
        # Policy network (actor)
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value network (critic)
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (action_probs, value)
        """
        action_probs = self.policy_net(x)
        value = self.value_net(x)
        return action_probs, value
    
    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get action probabilities."""
        return self.policy_net(x)
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get state value."""
        return self.value_net(x)


class MultiAgentPPO:
    """
    Multi-Agent Proximal Policy Optimization implementation.
    
    Features:
    - Centralized training with decentralized execution
    - Experience replay and advantage estimation
    - Multi-agent coordination through shared experience
    - Adaptive learning rates and clipping
    """
    
    def __init__(self, 
                 observation_dim: int,
                 action_dim: int,
                 num_agents: int = 20,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 device: str = "cpu"):
        """
        Initialize Multi-Agent PPO.
        
        Args:
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
            num_agents: Number of agents in the swarm
            learning_rate: Learning rate for optimization
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clipping ratio
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy coefficient for exploration
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run computations on
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Initialize networks
        self.policy_net = PPONetwork(observation_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience buffers
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.action_probs = []
        self.dones = []
        
        # Training parameters
        self.batch_size = 64
        self.epochs_per_update = 4
        self.target_kl = 0.01
        
        # Performance tracking
        self.training_losses = []
        self.value_losses = []
        self.policy_losses = []
        self.entropy_losses = []
        
        # Multi-agent coordination
        self.shared_experience = True
        self.coordination_reward = 0.1
        self.agent_communication = {}
    
    def select_action(self, observation: np.ndarray, agent_id: str) -> Tuple[int, float, float]:
        """
        Select action for a specific agent.
        
        Args:
            observation: Current observation
            agent_id: ID of the agent
            
        Returns:
            Tuple of (action, action_prob, value)
        """
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Get action probabilities and value
        with torch.no_grad():
            action_probs, value = self.policy_net(obs_tensor)
            action_probs = action_probs.squeeze(0)
            value = value.squeeze(0)
        
        # Sample action from distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        action_prob = action_probs[action].item()
        
        return action.item(), action_prob, value.item()
    
    def store_experience(self, 
                        observation: np.ndarray,
                        action: int,
                        reward: float,
                        value: float,
                        action_prob: float,
                        done: bool,
                        agent_id: str):
        """
        Store experience for training.
        
        Args:
            observation: Agent observation
            action: Taken action
            reward: Received reward
            value: Predicted value
            action_prob: Action probability
            done: Episode done flag
            agent_id: ID of the agent
        """
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.action_probs.append(action_prob)
        self.dones.append(done)
        
        # Store agent communication for coordination
        if agent_id not in self.agent_communication:
            self.agent_communication[agent_id] = []
        self.agent_communication[agent_id].append({
            "observation": observation,
            "action": action,
            "reward": reward
        })
    
    def compute_advantages(self, rewards: List[float], values: List[float], 
                          dones: List[bool]) -> np.ndarray:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: List of predicted values
            dones: List of done flags
            
        Returns:
            Array of advantages
        """
        advantages = np.zeros(len(rewards))
        last_advantage = 0
        last_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
            last_value = values[t]
        
        return advantages
    
    def update_policy(self) -> Dict[str, float]:
        """
        Update the policy using stored experiences.
        
        Returns:
            Dictionary containing training losses
        """
        if len(self.observations) < self.batch_size:
            return {}
        
        # Convert experiences to tensors
        observations = torch.FloatTensor(np.array(self.observations)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_action_probs = torch.FloatTensor(self.action_probs).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        dones = torch.BoolTensor(self.dones).to(self.device)
        
        # Compute advantages
        advantages = self.compute_advantages(self.rewards, self.values, self.dones)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute returns
        returns = advantages + torch.FloatTensor(self.values).to(self.device)
        
        # Training loop
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for epoch in range(self.epochs_per_update):
            # Forward pass
            action_probs, values = self.policy_net(observations)
            
            # Compute action distribution
            action_dist = torch.distributions.Categorical(action_probs)
            new_action_probs = action_dist.probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Compute ratio
            ratio = new_action_probs / (old_action_probs + 1e-8)
            
            # Compute clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Compute entropy loss
            entropy_loss = -action_dist.entropy().mean()
            
            # Total loss
            total_loss = (policy_loss + 
                         self.value_loss_coef * value_loss + 
                         self.entropy_coef * entropy_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # Store losses
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())
            
            # Early stopping if KL divergence is too high
            with torch.no_grad():
                kl_div = (old_action_probs * (torch.log(old_action_probs + 1e-8) - 
                                            torch.log(new_action_probs + 1e-8))).sum(-1).mean()
                if kl_div > self.target_kl:
                    break
        
        # Store average losses
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        avg_entropy_loss = np.mean(entropy_losses)
        
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.entropy_losses.append(avg_entropy_loss)
        
        # Clear experience buffer
        self._clear_experience_buffer()
        
        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy_loss": avg_entropy_loss,
            "kl_divergence": kl_div.item()
        }
    
    def _clear_experience_buffer(self):
        """Clear the experience buffer after training."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.action_probs.clear()
        self.dones.clear()
    
    def compute_coordination_reward(self, agent_actions: Dict[str, List[Dict]]) -> float:
        """
        Compute coordination reward based on agent actions.
        
        Args:
            agent_actions: Dictionary mapping agent IDs to their actions
            
        Returns:
            Coordination reward value
        """
        if not self.shared_experience:
            return 0.0
        
        coordination_score = 0.0
        total_actions = 0
        
        # Analyze coordination patterns
        for agent_id, actions in agent_actions.items():
            for action in actions:
                total_actions += 1
                
                # Check for coordinated actions
                if "coordinated" in action:
                    coordination_score += 1.0
                elif "resource_collected" in action:
                    coordination_score += 0.5
                elif "collision_avoided" in action:
                    coordination_score += 0.3
        
        if total_actions > 0:
            coordination_score = coordination_score / total_actions
            return coordination_score * self.coordination_reward
        
        return 0.0
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "policy_losses": self.policy_losses[-100:] if self.policy_losses else [],
            "value_losses": self.value_losses[-100:] if self.value_losses else [],
            "entropy_losses": self.entropy_losses[-100:] if self.entropy_losses else [],
            "avg_policy_loss": np.mean(self.policy_losses[-100:]) if self.policy_losses else 0.0,
            "avg_value_loss": np.mean(self.value_losses[-100:]) if self.value_losses else 0.0,
            "avg_entropy_loss": np.mean(self.entropy_losses[-100:]) if self.entropy_losses else 0.0
        }
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            "policy_net_state_dict": self.policy_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_stats": self.get_training_stats()
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "training_stats" in checkpoint:
            stats = checkpoint["training_stats"]
            self.policy_losses = stats.get("policy_losses", [])
            self.value_losses = stats.get("value_losses", [])
            self.entropy_losses = stats.get("entropy_losses", [])
    
    def reset(self):
        """Reset the algorithm state."""
        self._clear_experience_buffer()
        self.agent_communication.clear()
        self.training_losses.clear()
        self.value_losses.clear()
        self.policy_losses.clear()
        self.entropy_losses.clear()
