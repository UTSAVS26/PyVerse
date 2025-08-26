import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Dict, List, Tuple, Optional
from environment import Position, CellType
from agents import Agent, Prey, Predator

class QNetwork(nn.Module):
    """Neural network for Q-learning."""
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class RLAgent(Agent):
    """Base class for reinforcement learning agents."""
    
    def __init__(self, agent_id: int, position: Position, energy: float = 50,
                 learning_rate: float = 0.001, epsilon: float = 0.1, gamma: float = 0.95):
        super().__init__(agent_id, position, energy)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.update_frequency = 100
        self.step_count = 0
        
    def get_state_vector(self, environment) -> np.ndarray:
        """Convert environment state to neural network input vector."""
        # Base state features
        state = [
            self.energy / self.max_energy,
            self.age / 1000,  # Normalize age
            self.position.x / environment.width,
            self.position.y / environment.height
        ]
        
        # Vision-based features
        vision_range = min(self.vision_range, 5)  # Limit for computational efficiency
        
        # Count nearby entities
        nearby_plants = len(environment.get_plants_in_radius(self.position, vision_range))
        nearby_prey = len(environment.get_agents_in_radius(self.position, vision_range, Prey))
        nearby_predators = len(environment.get_agents_in_radius(self.position, vision_range, Predator))
        
        # Normalize counts
        max_entities = vision_range * vision_range
        state.extend([
            nearby_plants / max_entities,
            nearby_prey / max_entities,
            nearby_predators / max_entities
        ])
        
        # Direction to nearest entities
        nearest_plant_dir = self._get_direction_to_nearest(environment, CellType.PLANT, vision_range)
        nearest_prey_dir = self._get_direction_to_nearest(environment, Prey, vision_range)
        nearest_predator_dir = self._get_direction_to_nearest(environment, Predator, vision_range)
        
        state.extend(nearest_plant_dir + nearest_prey_dir + nearest_predator_dir)
        
        return np.array(state, dtype=np.float32)
    
    def _get_direction_to_nearest(self, environment, entity_type, vision_range) -> List[float]:
        """Get normalized direction to nearest entity of given type."""
        if entity_type == CellType.PLANT:
            entities = environment.get_plants_in_radius(self.position, vision_range)
            if entities:
                nearest = min(entities, key=lambda p: self.position.distance_to(p))
                dx = (nearest.x - self.position.x) / vision_range
                dy = (nearest.y - self.position.y) / vision_range
                return [dx, dy]
        else:
            entities = environment.get_agents_in_radius(self.position, vision_range, entity_type)
            if entities:
                nearest_pos, _ = min(entities, key=lambda x: self.position.distance_to(x[0]))
                dx = (nearest_pos.x - self.position.x) / vision_range
                dy = (nearest_pos.y - self.position.y) / vision_range
                return [dx, dy]
        
        return [0.0, 0.0]  # No entity found
    
    def get_available_actions(self, environment) -> List[Position]:
        """Get list of available actions (positions to move to)."""
        actions = [self.position]  # Stay in place is always available
        
        # Add adjacent positions
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_pos = Position(self.position.x + dx, self.position.y + dy)
                if environment.is_valid_position(new_pos) and environment.is_empty(new_pos):
                    actions.append(new_pos)
        
        return actions
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, environment) -> Optional[Position]:
        """Choose action using epsilon-greedy policy."""
        super().act(environment)
        
        state = self.get_state_vector(environment)
        available_actions = self.get_available_actions(environment)
        
        if random.random() < self.epsilon:
            # Random action
            action = random.choice(available_actions)
        else:
            # Best action according to Q-network
            action = self._get_best_action(state, available_actions)
        
        return action
    
    def _get_best_action(self, state: np.ndarray, available_actions: List[Position]) -> Position:
        """Get the best action according to the Q-network."""
        # This should be implemented by subclasses
        return random.choice(available_actions)
    
    def learn(self):
        """Learn from stored experiences."""
        # This should be implemented by subclasses
        pass

class QLearningAgent(RLAgent):
    """Q-learning agent using a simple Q-table approach."""
    
    def __init__(self, agent_id: int, position: Position, energy: float = 50,
                 learning_rate: float = 0.1, epsilon: float = 0.1, gamma: float = 0.95):
        super().__init__(agent_id, position, energy, learning_rate, epsilon, gamma)
        self.q_table = {}
        
    def _get_state_action_key(self, state: np.ndarray, action: Position) -> str:
        """Create a key for the Q-table."""
        # Discretize state and action for Q-table
        state_discrete = tuple(np.round(state * 10).astype(int))
        action_discrete = (action.x, action.y)
        return str((state_discrete, action_discrete))
    
    def _get_q_value(self, state: np.ndarray, action: Position) -> float:
        """Get Q-value for state-action pair."""
        key = self._get_state_action_key(state, action)
        return self.q_table.get(key, 0.0)
    
    def _set_q_value(self, state: np.ndarray, action: Position, value: float):
        """Set Q-value for state-action pair."""
        key = self._get_state_action_key(state, action)
        self.q_table[key] = value
    
    def _get_best_action(self, state: np.ndarray, available_actions: List[Position]) -> Position:
        """Get the best action according to Q-table."""
        best_action = available_actions[0]
        best_value = self._get_q_value(state, best_action)
        
        for action in available_actions[1:]:
            value = self._get_q_value(state, action)
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action
    
    def learn(self):
        """Learn from stored experiences using Q-learning."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            current_q = self._get_q_value(state, action)
            
            if done:
                target_q = reward
            else:
                # Find best Q-value for next state
                next_actions = self.get_available_actions_from_state(next_state)
                next_q_values = [self._get_q_value(next_state, next_action) 
                               for next_action in next_actions]
                max_next_q = max(next_q_values) if next_q_values else 0
                target_q = reward + self.gamma * max_next_q
            
            # Update Q-value
            new_q = current_q + self.learning_rate * (target_q - current_q)
            self._set_q_value(state, action, new_q)
    
    def get_available_actions_from_state(self, state: np.ndarray) -> List[Position]:
        """Get available actions from a state (simplified)."""
        # This is a simplified version - in practice, you'd need the environment
        return [Position(0, 0)]  # Placeholder

class DQNAgent(RLAgent):
    """Deep Q-Network agent using neural networks."""
    
    def __init__(self, agent_id: int, position: Position, energy: float = 50,
                 learning_rate: float = 0.001, epsilon: float = 0.1, gamma: float = 0.95):
        super().__init__(agent_id, position, energy, learning_rate, epsilon, gamma)
        
        # Neural network parameters
        self.input_size = 13  # Size of state vector
        self.output_size = 9   # 8 directions + stay
        self.hidden_size = 64
        
        # Networks
        self.q_network = QNetwork(self.input_size, self.output_size, self.hidden_size)
        self.target_network = QNetwork(self.input_size, self.output_size, self.hidden_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def _action_to_index(self, action: Position) -> int:
        """Convert action position to network output index."""
        dx = action.x - self.position.x
        dy = action.y - self.position.y
        
        # Map to 8 directions + stay
        if dx == 0 and dy == 0:
            return 0  # Stay
        elif dx == -1 and dy == -1:
            return 1
        elif dx == 0 and dy == -1:
            return 2
        elif dx == 1 and dy == -1:
            return 3
        elif dx == 1 and dy == 0:
            return 4
        elif dx == 1 and dy == 1:
            return 5
        elif dx == 0 and dy == 1:
            return 6
        elif dx == -1 and dy == 1:
            return 7
        elif dx == -1 and dy == 0:
            return 8
        else:
            return 0  # Default to stay
    
    def _index_to_action(self, index: int, available_actions: List[Position]) -> Position:
        """Convert network output index to action position."""
        if index == 0:
            return self.position  # Stay
        
        # Map index to direction
        directions = [
            (-1, -1), (0, -1), (1, -1),
            (1, 0), (1, 1), (0, 1),
            (-1, 1), (-1, 0)
        ]
        
        if 1 <= index <= 8:
            dx, dy = directions[index - 1]
            target_pos = Position(self.position.x + dx, self.position.y + dy)
            
            # Check if target position is available
            if target_pos in available_actions:
                return target_pos
        
        # Fallback to random available action
        return random.choice(available_actions)
    
    def _get_best_action(self, state: np.ndarray, available_actions: List[Position]) -> Position:
        """Get the best action according to Q-network."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor).squeeze()
            
            # Convert available actions to indices and get their Q-values
            action_q_values = []
            for action in available_actions:
                action_idx = self._action_to_index(action)
                action_q_values.append((action, q_values[action_idx].item()))
            
            # Return action with highest Q-value
            best_action = max(action_q_values, key=lambda x: x[1])[0]
            return best_action
    
    def learn(self):
        """Learn from stored experiences using DQN."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        
        # Convert to numpy arrays first for better performance
        states = np.array([exp[0] for exp in batch], dtype=np.float32)
        actions = np.array([self._action_to_index(exp[1]) for exp in batch], dtype=np.int64)
        rewards = np.array([exp[2] for exp in batch], dtype=np.float32)
        next_states = np.array([exp[3] for exp in batch], dtype=np.float32)
        dones = np.array([exp[4] for exp in batch], dtype=bool)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
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
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

class RLPrey(DQNAgent, Prey):
    """Reinforcement learning prey agent."""
    
    def __init__(self, agent_id: int, position: Position, energy: float = 50):
        DQNAgent.__init__(self, agent_id, position, energy)
        Prey.__init__(self, agent_id, position, energy)
        self.cell_type = CellType.PREY
        
    def act(self, environment) -> Optional[Position]:
        """RL prey behavior with learning."""
        # Get current state
        state = self.get_state_vector(environment)
        
        # Choose action
        new_position = super().act(environment)
        
        # Get reward based on outcome
        reward = self._calculate_reward(environment, new_position)
        
        # Get next state
        next_state = self.get_state_vector(environment)
        
        # Check if agent died
        done = self.energy <= 0
        
        # Store experience
        self.remember(state, new_position, reward, next_state, done)
        
        # Learn
        self.learn()
        
        return new_position
    
    def _calculate_reward(self, environment, action: Position) -> float:
        """Calculate reward for the action taken."""
        reward = 0.0
        
        # Reward for staying alive
        reward += 1.0
        
        # Reward for eating plants
        if environment.grid[action.y, action.x] == CellType.PLANT.value:
            reward += 10.0
        
        # Penalty for being near predators
        nearby_predators = environment.get_agents_in_radius(action, 3, Predator)
        reward -= len(nearby_predators) * 5.0
        
        # Penalty for low energy
        if self.energy < 20:
            reward -= 5.0
        
        # Reward for high energy
        if self.energy > 60:
            reward += 2.0
        
        return reward

class RLPredator(DQNAgent, Predator):
    """Reinforcement learning predator agent."""
    
    def __init__(self, agent_id: int, position: Position, energy: float = 50):
        DQNAgent.__init__(self, agent_id, position, energy)
        Predator.__init__(self, agent_id, position, energy)
        self.cell_type = CellType.PREDATOR
        
    def act(self, environment) -> Optional[Position]:
        """RL predator behavior with learning."""
        # Get current state
        state = self.get_state_vector(environment)
        
        # Choose action
        new_position = super().act(environment)
        
        # Get reward based on outcome
        reward = self._calculate_reward(environment, new_position)
        
        # Get next state
        next_state = self.get_state_vector(environment)
        
        # Check if agent died
        done = self.energy <= 0
        
        # Store experience
        self.remember(state, new_position, reward, next_state, done)
        
        # Learn
        self.learn()
        
        return new_position
    
    def _calculate_reward(self, environment, action: Position) -> float:
        """Calculate reward for the action taken."""
        reward = 0.0
        
        # Reward for staying alive
        reward += 1.0
        
        # Reward for being near prey
        nearby_prey = environment.get_agents_in_radius(action, 3, Prey)
        reward += len(nearby_prey) * 3.0
        
        # Penalty for low energy
        if self.energy < 30:
            reward -= 5.0
        
        # Reward for high energy
        if self.energy > 70:
            reward += 2.0
        
        return reward
