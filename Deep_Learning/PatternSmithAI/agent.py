"""
PatternSmithAI - AI Agent for Pattern Generation
Handles reinforcement learning for pattern aesthetics and generation rules.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
import random
import json
from collections import deque
import math
from canvas import PatternCanvas, ColorPalette
from patterns import PatternFactory


class PatternState:
    """Represents the state of a pattern for the AI agent."""
    
    def __init__(self, canvas: PatternCanvas):
        self.canvas = canvas
        self.pixel_data = canvas.get_pixel_data()
        self.features = self._extract_features()
    
    def _extract_features(self) -> np.ndarray:
        """Extract features from the pattern for the AI agent."""
        # Convert to grayscale and normalize
        if len(self.pixel_data.shape) == 3:
            gray = np.mean(self.pixel_data, axis=2)
        else:
            gray = self.pixel_data
        
        # Resize to fixed size for neural network
        gray_resized = self._resize_image(gray, (64, 64))
        
        # Calculate additional features
        symmetry_score = self._calculate_symmetry()
        complexity_score = self._calculate_complexity()
        color_variety = self._calculate_color_variety()
        
        # Combine features
        features = np.concatenate([
            gray_resized.flatten() / 255.0,  # Normalized pixel values
            [symmetry_score, complexity_score, color_variety]
        ])
        
        return features
    
    def _resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image to fixed size."""
        # Simple nearest neighbor resizing
        h, w = image.shape
        new_h, new_w = size
        
        resized = np.zeros(size)
        for i in range(new_h):
            for j in range(new_w):
                old_i = int(i * h / new_h)
                old_j = int(j * w / new_w)
                resized[i, j] = image[old_i, old_j]
        
        return resized
    
    def _calculate_symmetry(self) -> float:
        """Calculate symmetry score of the pattern."""
        if len(self.pixel_data.shape) == 3:
            gray = np.mean(self.pixel_data, axis=2)
        else:
            gray = self.pixel_data
        
        h, w = gray.shape
        center_x, center_y = w // 2, h // 2
        
        # Check horizontal symmetry
        horizontal_symmetry = 0
        for y in range(h):
            for x in range(w // 2):
                if abs(gray[y, x] - gray[y, w - 1 - x]) < 10:
                    horizontal_symmetry += 1
        
        # Check vertical symmetry
        vertical_symmetry = 0
        for y in range(h // 2):
            for x in range(w):
                if abs(gray[y, x] - gray[h - 1 - y, x]) < 10:
                    vertical_symmetry += 1
        
        total_pixels = h * w
        symmetry_score = (horizontal_symmetry + vertical_symmetry) / (2 * total_pixels)
        
        return symmetry_score
    
    def _calculate_complexity(self) -> float:
        """Calculate complexity score of the pattern."""
        if len(self.pixel_data.shape) == 3:
            gray = np.mean(self.pixel_data, axis=2)
        else:
            gray = self.pixel_data
        
        # Calculate gradient magnitude
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Complexity is the average gradient magnitude
        complexity = np.mean(gradient_magnitude)
        
        return complexity / 255.0  # Normalize
    
    def _calculate_color_variety(self) -> float:
        """Calculate color variety score."""
        if len(self.pixel_data.shape) != 3:
            return 0.0
        
        # Count unique colors
        unique_colors = len(np.unique(self.pixel_data.reshape(-1, 3), axis=0))
        max_colors = 256 * 256 * 256  # Maximum possible colors
        
        return unique_colors / max_colors


class PatternAction:
    """Represents an action the AI agent can take."""
    
    def __init__(self, action_type: str, parameters: Dict[str, Any]):
        self.action_type = action_type
        self.parameters = parameters
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary."""
        return {
            'action_type': self.action_type,
            'parameters': self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternAction':
        """Create action from dictionary."""
        return cls(data['action_type'], data['parameters'])


class PatternNeuralNetwork(nn.Module):
    """Neural network for pattern generation decisions."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 64):
        super(PatternNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class PatternAgent:
    """AI agent for pattern generation using reinforcement learning."""
    
    def __init__(self, canvas: PatternCanvas, color_palette: ColorPalette):
        self.canvas = canvas
        self.color_palette = color_palette
        self.pattern_factory = PatternFactory()
        
        # Neural network
        self.input_size = 64 * 64 + 3  # Image features + additional features
        self.hidden_size = 128
        self.output_size = 64  # Number of possible actions
        
        self.policy_net = PatternNeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        self.target_net = PatternNeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Training parameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99  # Discount factor
        
        # Action space
        self.action_space = self._create_action_space()
        
        # Training history
        self.training_history = []
    
    def _create_action_space(self) -> List[PatternAction]:
        """Create the action space for the agent."""
        actions = []
        
        # Geometric pattern actions
        for pattern_type in ["circles", "squares", "polygons", "stars"]:
            actions.append(PatternAction("geometric", {
                "pattern_type": pattern_type,
                "count": random.randint(5, 30),
                "palette": random.choice(["rainbow", "pastel", "monochrome", "earth", "ocean"])
            }))
        
        # Mandala actions
        for base_shape in ["circle", "square", "star"]:
            actions.append(PatternAction("mandala", {
                "base_shape": base_shape,
                "layers": random.randint(3, 10),
                "elements_per_layer": random.randint(8, 16)
            }))
        
        # Fractal actions
        for fractal_type in ["sierpinski", "koch", "tree"]:
            actions.append(PatternAction("fractal", {
                "fractal_type": fractal_type,
                "depth": random.randint(3, 6)
            }))
        
        # Tiling actions
        for tile_type in ["hexagonal", "square", "triangular"]:
            actions.append(PatternAction("tiling", {
                "tile_type": tile_type,
                "tile_size": random.randint(30, 80)
            }))
        
        return actions
    
    def get_state(self) -> PatternState:
        """Get current pattern state."""
        return PatternState(self.canvas)
    
    def select_action(self, state: PatternState) -> PatternAction:
        """Select an action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        
        state_tensor = torch.FloatTensor(state.features).unsqueeze(0)
        q_values = self.policy_net(state_tensor)
        action_idx = q_values.argmax().item()
        
        return self.action_space[action_idx % len(self.action_space)]
    
    def execute_action(self, action: PatternAction) -> Tuple[PatternState, float]:
        """Execute an action and return new state and reward."""
        # Create generator based on action type
        generator = self.pattern_factory.create_generator(
            action.action_type, self.canvas, self.color_palette
        )
        
        # Execute the action
        generator.generate(**action.parameters)
        
        # Get new state
        new_state = self.get_state()
        
        # Calculate reward
        reward = self._calculate_reward(new_state)
        
        return new_state, reward
    
    def _calculate_reward(self, state: PatternState) -> float:
        """Calculate reward for a pattern state."""
        features = state.features
        
        # Extract feature values
        symmetry_score = features[-3]
        complexity_score = features[-2]
        color_variety = features[-1]
        
        # Reward components
        symmetry_reward = symmetry_score * 2.0  # Higher symmetry is better
        complexity_reward = complexity_score * 1.5  # Moderate complexity is good
        color_reward = color_variety * 1.0  # Some color variety is good
        
        # Penalize extreme values
        if complexity_score > 0.8:
            complexity_reward *= 0.5  # Too complex
        if color_variety < 0.1:
            color_reward *= 0.5  # Too monochrome
        
        total_reward = symmetry_reward + complexity_reward + color_reward
        
        return total_reward
    
    def remember(self, state: PatternState, action: PatternAction, 
                reward: float, next_state: PatternState, done: bool):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the agent using experience replay."""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        state_tensor = torch.FloatTensor([s.features for s in states])
        next_state_tensor = torch.FloatTensor([s.features for s in next_states])
        reward_tensor = torch.FloatTensor(rewards)
        done_tensor = torch.BoolTensor(dones)
        
        # Get action indices
        action_indices = torch.LongTensor([
            self.action_space.index(action) for action in actions
        ])
        
        # Current Q values
        current_q_values = self.policy_net(state_tensor).gather(1, action_indices.unsqueeze(1))
        
        # Next Q values
        next_q_values = self.target_net(next_state_tensor).max(1)[0].detach()
        target_q_values = reward_tensor + (self.gamma * next_q_values * ~done_tensor)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, filename: str):
        """Save the trained model."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_history': self.training_history
        }, filename)
    
    def load_model(self, filename: str):
        """Load a trained model."""
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_history = checkpoint['training_history']
    
    def generate_pattern(self, steps: int = 10) -> PatternCanvas:
        """Generate a pattern using the trained agent."""
        self.canvas.clear()
        
        for step in range(steps):
            state = self.get_state()
            action = self.select_action(state)
            next_state, reward = self.execute_action(action)
            
            # Store experience
            self.remember(state, action, reward, next_state, step == steps - 1)
            
            # Train
            self.replay()
            
            # Update target network occasionally
            if step % 100 == 0:
                self.update_target_network()
        
        return self.canvas
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'training_history': self.training_history
        }


class PatternEvaluator:
    """Evaluates pattern quality and provides feedback."""
    
    def __init__(self):
        self.evaluation_criteria = {
            'symmetry': 0.3,
            'complexity': 0.25,
            'color_harmony': 0.25,
            'balance': 0.2
        }
    
    def evaluate_pattern(self, canvas: PatternCanvas) -> Dict[str, float]:
        """Evaluate a pattern and return scores."""
        state = PatternState(canvas)
        
        scores = {
            'symmetry': state._calculate_symmetry(),
            'complexity': state._calculate_complexity(),
            'color_harmony': state._calculate_color_variety(),
            'balance': self._calculate_balance(canvas)
        }
        
        return scores
    
    def _calculate_balance(self, canvas: PatternCanvas) -> float:
        """Calculate visual balance of the pattern."""
        pixel_data = canvas.get_pixel_data()
        
        if len(pixel_data.shape) == 3:
            gray = np.mean(pixel_data, axis=2)
        else:
            gray = pixel_data
        
        h, w = gray.shape
        
        # Calculate center of mass
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        total_mass = np.sum(gray)
        
        if total_mass == 0:
            return 0.5
        
        center_x = np.sum(x_coords * gray) / total_mass
        center_y = np.sum(y_coords * gray) / total_mass
        
        # Distance from center
        ideal_center_x = w / 2
        ideal_center_y = h / 2
        
        distance = np.sqrt((center_x - ideal_center_x)**2 + (center_y - ideal_center_y)**2)
        max_distance = np.sqrt((w/2)**2 + (h/2)**2)
        
        balance = 1.0 - (distance / max_distance)
        return max(0.0, min(1.0, balance))
    
    def get_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate overall pattern score."""
        overall_score = 0.0
        for criterion, weight in self.evaluation_criteria.items():
            overall_score += scores[criterion] * weight
        
        return overall_score
