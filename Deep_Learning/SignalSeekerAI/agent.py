import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import copy


class BaseAgent(ABC):
    """Base class for spectrum scanning agents."""
    
    def __init__(self, num_channels: int, scan_window: int = 50):
        """
        Initialize the base agent.
        
        Args:
            num_channels: Number of frequency channels
            scan_window: Number of channels to scan at once
        """
        self.num_channels = num_channels
        self.scan_window = scan_window
        self.current_position = 0
        self.scan_history = []
        self.detection_history = []
    
    @abstractmethod
    def select_action(self, spectrum: np.ndarray, frequencies: np.ndarray) -> int:
        """
        Select the next frequency to scan.
        
        Args:
            spectrum: Current spectrum power levels
            frequencies: Corresponding frequencies
            
        Returns:
            Index of the next frequency to scan
        """
        pass
    
    def scan_spectrum(self, spectrum: np.ndarray, frequencies: np.ndarray) -> Dict:
        """
        Perform a spectrum scan and return results.
        
        Args:
            spectrum: Current spectrum power levels
            frequencies: Corresponding frequencies
            
        Returns:
            Scan results including position, power, and detection info
        """
        # Select next position to scan
        next_position = self.select_action(spectrum, frequencies)
        self.current_position = next_position
        
        # Get power at current position
        power = spectrum[next_position]
        frequency = frequencies[next_position]
        
        # Record scan
        scan_result = {
            'position': next_position,
            'frequency': frequency,
            'power': power,
            'timestamp': len(self.scan_history)
        }
        self.scan_history.append(scan_result)
        
        return scan_result
    
    def get_scan_statistics(self) -> Dict:
        """Get statistics about the agent's scanning behavior."""
        if not self.scan_history:
            return {}
        
        powers = [scan['power'] for scan in self.scan_history]
        positions = [scan['position'] for scan in self.scan_history]
        
        return {
            'total_scans': len(self.scan_history),
            'avg_power': np.mean(powers),
            'max_power': np.max(powers),
            'min_power': np.min(powers),
            'position_variance': np.var(positions),
            'coverage': len(set(positions)) / self.num_channels
        }


class RandomAgent(BaseAgent):
    """Random scanning agent (baseline)."""
    
    def select_action(self, spectrum: np.ndarray, frequencies: np.ndarray) -> int:
        """Select a random frequency to scan."""
        return random.randint(0, self.num_channels - 1)


class SequentialAgent(BaseAgent):
    """Sequential scanning agent."""
    
    def select_action(self, spectrum: np.ndarray, frequencies: np.ndarray) -> int:
        """Scan frequencies sequentially."""
        next_position = (self.current_position + 1) % self.num_channels
        return next_position


class AdaptiveAgent(BaseAgent):
    """Adaptive scanning agent that focuses on high-power regions."""
    
    def __init__(self, num_channels: int, scan_window: int = 50, 
                 power_threshold: float = -70, exploration_rate: float = 0.1):
        super().__init__(num_channels, scan_window)
        self.power_threshold = power_threshold
        self.exploration_rate = exploration_rate
        self.high_power_regions = []
    
    def select_action(self, spectrum: np.ndarray, frequencies: np.ndarray) -> int:
        """Select next position based on power levels and exploration."""
        # Exploration: random choice
        if random.random() < self.exploration_rate:
            return random.randint(0, self.num_channels - 1)
        
        # Exploitation: focus on high-power regions
        high_power_indices = np.where(spectrum > self.power_threshold)[0]
        
        if len(high_power_indices) > 0:
            # Choose from high-power regions
            return random.choice(high_power_indices)
        else:
            # If no high-power regions, scan sequentially
            return (self.current_position + 1) % self.num_channels


class DQNAgent(BaseAgent):
    """Deep Q-Network agent for reinforcement learning."""
    
    def __init__(self, num_channels: int, scan_window: int = 50,
                 learning_rate: float = 0.001, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, memory_size: int = 10000):
        super().__init__(num_channels, scan_window)
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.batch_size = 32
        
        # Neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNetwork(num_channels, scan_window).to(self.device)
        self.target_network = DQNetwork(num_channels, scan_window).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Update target network
        self.update_target_network()
    
    def select_action(self, spectrum: np.ndarray, frequencies: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_channels - 1)
        
        # Get state representation
        state = self._get_state_representation(spectrum)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        # Select action with highest Q-value
        return q_values.argmax().item()
    
    def _get_state_representation(self, spectrum: np.ndarray) -> np.ndarray:
        """Convert spectrum to state representation for the neural network."""
        # Normalize spectrum to [0, 1] range
        normalized_spectrum = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum) + 1e-8)
        
        # Add current position as one-hot encoding
        position_one_hot = np.zeros(self.num_channels)
        position_one_hot[self.current_position] = 1.0
        
        # Combine spectrum and position
        state = np.concatenate([normalized_spectrum, position_one_hot])
        return state
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network with current network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']


class DQNetwork(nn.Module):
    """Neural network for DQN agent."""
    
    def __init__(self, num_channels: int, scan_window: int):
        super(DQNetwork, self).__init__()
        
        # Input: normalized spectrum + position one-hot
        input_size = num_channels * 2
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_channels)  # Q-values for each channel
        )
    
    def forward(self, x):
        return self.network(x)


class MultiAgentEnvironment:
    """Environment for multiple agents competing for spectrum access."""
    
    def __init__(self, num_agents: int, num_channels: int, scan_window: int = 50):
        """
        Initialize multi-agent environment.
        
        Args:
            num_agents: Number of competing agents
            num_channels: Number of frequency channels
            scan_window: Number of channels to scan at once
        """
        self.num_agents = num_agents
        self.num_channels = num_channels
        self.scan_window = scan_window
        
        # Create agents
        self.agents = []
        agent_types = [RandomAgent, SequentialAgent, AdaptiveAgent, DQNAgent]
        
        for i in range(num_agents):
            agent_type = agent_types[i % len(agent_types)]
            agent = agent_type(num_channels, scan_window)
            self.agents.append(agent)
        
        # Competition tracking
        self.agent_positions = [0] * num_channels
        self.collision_history = []
    
    def step(self, spectrum: np.ndarray, frequencies: np.ndarray) -> List[Dict]:
        """
        Perform one step with all agents.
        
        Args:
            spectrum: Current spectrum power levels
            frequencies: Corresponding frequencies
            
        Returns:
            List of scan results for each agent
        """
        results = []
        
        # Get actions from all agents
        actions = []
        for agent in self.agents:
            action = agent.select_action(spectrum, frequencies)
            actions.append(action)
        
        # Check for collisions
        collisions = self._check_collisions(actions)
        
        # Execute scans
        for i, agent in enumerate(self.agents):
            scan_result = agent.scan_spectrum(spectrum, frequencies)
            scan_result['agent_id'] = i
            scan_result['collision'] = collisions[i]
            results.append(scan_result)
        
        return results
    
    def _check_collisions(self, actions: List[int]) -> List[bool]:
        """Check for collisions between agents."""
        collisions = [False] * len(actions)
        
        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                if actions[i] == actions[j]:
                    collisions[i] = True
                    collisions[j] = True
        
        return collisions
    
    def get_competition_statistics(self) -> Dict:
        """Get statistics about agent competition."""
        stats = {}
        
        for i, agent in enumerate(self.agents):
            agent_stats = agent.get_scan_statistics()
            agent_stats['agent_id'] = i
            agent_stats['agent_type'] = type(agent).__name__
            stats[f'agent_{i}'] = agent_stats
        
        return stats
