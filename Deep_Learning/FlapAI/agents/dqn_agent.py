import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import Dict, Any, List, Tuple, Optional
import pickle
import os
from .base_agent import BaseAgent

class DQNNetwork(nn.Module):
    """Neural network for DQN agent."""
    
    def __init__(self, input_size: int = 7, hidden_size: int = 64, output_size: int = 2):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: Dict[str, Any], action: int, 
             reward: float, next_state: Dict[str, Any], done: bool) -> None:
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample a batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent(BaseAgent):
    """DQN (Deep Q-Network) agent for FlapAI."""
    
    def __init__(self, learning_rate: float = 0.001, epsilon: float = 1.0, 
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 memory_size: int = 10000, batch_size: int = 32):
        super().__init__("DQNAgent")
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork().to(self.device)
        self.target_network = DQNNetwork().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = ReplayBuffer(memory_size)
        
        # Training variables
        self.update_target_counter = 0
        self.update_target_freq = 100
        
    def get_action(self, state: Dict[str, Any]) -> int:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: Current game state
            
        Returns:
            action: 0 for no flap, 1 for flap
        """
        if self.training and random.random() < self.epsilon:
            # Random action
            return random.randint(0, 1)
        else:
            # Greedy action
            state_tensor = self._state_to_tensor(state)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
                
    def update(self, state: Dict[str, Any], action: int, 
               reward: float, next_state: Dict[str, Any], done: bool) -> None:
        """
        Update the agent with experience and train the network.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Store experience in replay buffer
        self.memory.push(state, action, reward, next_state, done)
        
        # Train if enough samples
        if len(self.memory) >= self.batch_size:
            self._train()
            
        # Update target network periodically
        self.update_target_counter += 1
        if self.update_target_counter % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        # Decay epsilon
        if self.training and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def _train(self) -> None:
        """Train the Q-network using experience replay."""
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        
        # Prepare tensors
        states = torch.stack([self._state_to_tensor(s) for s, _, _, _, _ in batch]).to(self.device)
        actions = torch.tensor([a for _, a, _, _, _ in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([r for _, _, r, _, _ in batch], dtype=torch.float).to(self.device)
        next_states = torch.stack([self._state_to_tensor(s) for _, _, _, s, _ in batch]).to(self.device)
        dones = torch.tensor([d for _, _, _, _, d in batch], dtype=torch.bool).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values (from target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (0.99 * next_q_values * ~dones)
            
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def _state_to_tensor(self, state: Dict[str, Any]) -> torch.Tensor:
        """
        Convert state dictionary to tensor.
        
        Args:
            state: Game state dictionary
            
        Returns:
            tensor: PyTorch tensor
        """
        inputs = [
            state.get('bird_y', 0.5),
            state.get('bird_velocity', 0.0),
            state.get('pipe_x', 1.0),
            state.get('pipe_gap_y', 0.5),
            state.get('pipe_gap_size', 0.25),
            state.get('distance_to_pipe', 1.0),
            state.get('bird_alive', 1.0)
        ]
        
        return torch.FloatTensor(inputs).to(self.device)
        
    def save(self, filepath: str) -> None:
        """Save the agent to a file."""
        save_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'best_score': self.best_score
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
            
    def load(self, filepath: str) -> None:
        """Load the agent from a file."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
            
        self.q_network.load_state_dict(save_data['q_network_state_dict'])
        self.target_network.load_state_dict(save_data['target_network_state_dict'])
        self.optimizer.load_state_dict(save_data['optimizer_state_dict'])
        self.epsilon = save_data.get('epsilon', self.epsilon)
        self.episode_count = save_data.get('episode_count', 0)
        self.best_score = save_data.get('best_score', 0)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get current agent statistics."""
        stats = super().get_stats()
        stats.update({
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'device': str(self.device)
        })
        return stats
        
    def reset_episode(self) -> None:
        """Reset episode-specific variables."""
        super().reset_episode()
        # Don't reset epsilon here as it should decay over time 