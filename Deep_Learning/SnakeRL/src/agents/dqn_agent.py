import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Dict, Any, List, Tuple
from .base_agent import BaseAgent
from models.dqn_model import DQNModel

class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent with experience replay and target network
    
    Features:
    - Experience replay buffer for stable training
    - Target network for stable Q-value estimation
    - Gradient clipping to prevent exploding gradients
    """
    
    def __init__(self, state_size: int, action_size: int, **kwargs):
        """
        Initialize DQN agent
        
        Args:
            state_size: Dimension of the state space
            action_size: Number of possible actions
            **kwargs: Additional parameters including:
                - learning_rate: Learning rate for optimizer
                - discount_factor: Discount factor for future rewards
                - memory_size: Size of experience replay buffer
                - batch_size: Batch size for training
                - target_update: Frequency of target network updates
                - hidden_size: Size of hidden layers in neural network
        """
        super().__init__(state_size, action_size, **kwargs)
        
        # DQN specific parameters
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.discount_factor = kwargs.get('discount_factor', 0.95)
        self.memory_size = kwargs.get('memory_size', 10000)
        self.batch_size = kwargs.get('batch_size', 32)
        self.target_update = kwargs.get('target_update', 100)
        self.hidden_size = kwargs.get('hidden_size', 64)
        
        # Device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = DQNModel(state_size, action_size, self.hidden_size).to(self.device)
        self.target_network = DQNModel(state_size, action_size, self.hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Experience replay buffer
        self.memory = deque(maxlen=self.memory_size)
        
        # Training counter
        self.train_step = 0
        
    def act(self, state: np.ndarray) -> int:
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state observation
            
        Returns:
            Action to take (0=left, 1=straight, 2=right)
        """
        return self.q_network.get_action(state, self.epsilon)
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool) -> None:
        """
        Store experience in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, done: bool) -> None:
        """
        Store experience and train if enough samples
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        # Store experience
        self.remember(state, action, reward, next_state, done)
        
        # Train if enough samples
        if len(self.memory) >= self.batch_size:
            self._train_step()
    
    def _train_step(self) -> None:
        """Perform one training step using experience replay"""
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values (using target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            next_q_values[dones] = 0.0
            target_q_values = rewards + self.discount_factor * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath: str) -> None:
        """
        Save Q-network to file
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load Q-network from file
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information including memory size and training step"""
        info = super().get_info()
        info.update({
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'memory_size': len(self.memory),
            'batch_size': self.batch_size,
            'target_update': self.target_update,
            'train_step': self.train_step,
            'device': str(self.device)
        })
        return info
