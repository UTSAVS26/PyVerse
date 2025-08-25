"""
Agent module for ArtForgeAI - RL painter agent
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import random
from collections import deque
import math


class PaintingEnvironment:
    """Environment for the painting agent"""
    
    def __init__(self, canvas_width: int = 800, canvas_height: int = 600, max_strokes: int = 50):
        """
        Initialize the painting environment
        
        Args:
            canvas_width: Width of the canvas
            canvas_height: Height of the canvas
            max_strokes: Maximum number of strokes per episode
        """
        from canvas import Canvas
        from strokes import StrokeGenerator
        
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.max_strokes = max_strokes
        
        self.canvas = Canvas(canvas_width, canvas_height)
        self.stroke_generator = StrokeGenerator(canvas_width, canvas_height)
        
        self.current_stroke_count = 0
        self.episode_reward = 0.0
        
    def reset(self):
        """Reset the environment for a new episode"""
        self.canvas.reset()
        self.current_stroke_count = 0
        self.episode_reward = 0.0
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        # Normalize canvas image to [0, 1] and flatten
        canvas_img = self.canvas.get_image().astype(np.float32) / 255.0
        
        # Add additional state information
        coverage = self.canvas.get_coverage()
        color_diversity = self.canvas.get_color_diversity()
        stroke_count = self.current_stroke_count / self.max_strokes
        
        # Combine image and metadata, ensuring float32 dtype
        metadata = np.array([coverage, color_diversity, stroke_count], dtype=np.float32)
        state = np.concatenate([
            canvas_img.flatten(),
            metadata
        ])
        
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment
        
        Args:
            action: Action vector [stroke_type, x, y, angle, color_r, color_g, color_b, thickness]
        
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Decode action
        stroke_type_idx = min(3, int(action[0] * 4))  # 0-3 for stroke types
        x = int(action[1] * self.canvas_width)
        y = int(action[2] * self.canvas_height)
        angle = action[3] * 360  # 0-360 degrees
        color_r = int(action[4] * 255)
        color_g = int(action[5] * 255)
        color_b = int(action[6] * 255)
        thickness = max(1, int(action[7] * 10))
        
        # Map stroke type index to stroke type
        stroke_types = ['line', 'curve', 'dot', 'splash']
        stroke_type = stroke_types[stroke_type_idx]
        
        # Generate stroke based on action
        if stroke_type == 'line':
            # Calculate end position based on angle and random length
            max_length = min(self.canvas_width, self.canvas_height) // 4
            min_length = min(20, max_length)
            length = random.randint(min_length, max_length)
            angle_rad = math.radians(angle)
            end_x = x + int(length * math.cos(angle_rad))
            end_y = y + int(length * math.sin(angle_rad))
            
            # Ensure end position is within canvas bounds
            end_x = max(0, min(end_x, self.canvas_width - 1))
            end_y = max(0, min(end_y, self.canvas_height - 1))
            
            stroke_data = {
                'type': 'line',
                'start_pos': (x, y),
                'end_pos': (end_x, end_y),
                'color': (color_r, color_g, color_b),
                'thickness': thickness,
                'angle': angle
            }
            
        elif stroke_type == 'curve':
            # Generate control points
            control_points = []
            for _ in range(random.randint(1, 2)):
                t = random.uniform(0.2, 0.8)
                base_x = x + t * (random.randint(0, self.canvas_width) - x)
                base_y = y + t * (random.randint(0, self.canvas_height) - y)
                control_x = max(0, min(int(base_x), self.canvas_width - 1))
                control_y = max(0, min(int(base_y), self.canvas_height - 1))
                control_points.append((control_x, control_y))
            
            stroke_data = {
                'type': 'curve',
                'start_pos': (x, y),
                'end_pos': (random.randint(0, self.canvas_width), random.randint(0, self.canvas_height)),
                'control_points': control_points,
                'color': (color_r, color_g, color_b),
                'thickness': thickness
            }
            
        elif stroke_type == 'dot':
            radius = random.randint(3, 20)
            stroke_data = {
                'type': 'dot',
                'start_pos': (x, y),
                'color': (color_r, color_g, color_b),
                'radius': radius,
                'thickness': 1
            }
            
        elif stroke_type == 'splash':
            radius = random.randint(10, 40)
            stroke_data = {
                'type': 'splash',
                'start_pos': (x, y),
                'color': (color_r, color_g, color_b),
                'radius': radius,
                'thickness': 1
            }
        
        # Apply stroke to canvas
        success = self.canvas.apply_stroke(stroke_data)
        
        # Calculate reward
        reward = self._calculate_reward(success)
        self.episode_reward += reward
        
        # Update stroke count
        self.current_stroke_count += 1
        
        # Check if episode is done
        done = self.current_stroke_count >= self.max_strokes
        
        # Get next state
        next_state = self._get_state()
        
        info = {
            'stroke_type': stroke_type,
            'success': success,
            'coverage': self.canvas.get_coverage(),
            'color_diversity': self.canvas.get_color_diversity()
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, success: bool) -> float:
        """Calculate reward for the current state"""
        if not success:
            return -1.0  # Penalty for failed stroke
        
        # Base reward for successful stroke
        reward = 1.0
        
        # Coverage reward (encourage more coverage)
        coverage = self.canvas.get_coverage()
        reward += coverage * 2.0
        
        # Color diversity reward
        color_diversity = self.canvas.get_color_diversity()
        reward += color_diversity * 1.5
        
        # Stroke diversity reward
        stroke_types = [stroke['type'] for stroke in self.canvas.stroke_history]
        unique_strokes = len(set(stroke_types))
        reward += unique_strokes * 0.5
        
        # Balance reward (encourage strokes across the canvas)
        if len(self.canvas.stroke_history) > 1:
            # Calculate spatial distribution
            positions = [stroke['start_pos'] for stroke in self.canvas.stroke_history]
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            x_std = np.std(x_coords) / self.canvas_width
            y_std = np.std(y_coords) / self.canvas_height
            balance_score = (x_std + y_std) / 2.0
            reward += balance_score * 1.0
        
        return reward


class PaintingActor(nn.Module):
    """Actor network for the painting agent"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize the actor network
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Number of hidden units
        """
        super(PaintingActor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the actor network"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output in [0, 1] range
        return x


class PaintingCritic(nn.Module):
    """Critic network for the painting agent"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize the critic network
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Number of hidden units
        """
        super(PaintingCritic, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through the critic network"""
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PaintingAgent:
    """Reinforcement learning agent for painting"""
    
    def __init__(self, state_dim: int, action_dim: int, lr_actor: float = 1e-4, 
                 lr_critic: float = 1e-3, gamma: float = 0.99, tau: float = 0.005):
        """
        Initialize the painting agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            gamma: Discount factor
            tau: Soft update parameter
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        
        # Networks
        self.actor = PaintingActor(state_dim, action_dim)
        self.critic = PaintingCritic(state_dim, action_dim)
        self.target_actor = PaintingActor(state_dim, action_dim)
        self.target_critic = PaintingCritic(state_dim, action_dim)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize target networks
        self._update_target_networks(tau=1.0)
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        
        # Noise for exploration
        self.noise_std = 0.1
        
    def _update_target_networks(self, tau: float):
        """Update target networks using soft update"""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Select an action using the actor network
        
        Args:
            state: Current state
            add_noise: Whether to add exploration noise
        
        Returns:
            Action array
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).squeeze(0).detach().numpy()
        
        if add_noise:
            noise = np.random.normal(0, self.noise_std, self.action_dim)
            action = np.clip(action + noise, 0, 1)
        
        return action
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, done: bool):
        """Store transition in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self, batch_size: int = 64):
        """Train the agent on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones).unsqueeze(1)
        
        # Update critic
        next_actions = self.target_actor(next_states)
        target_q = self.target_critic(next_states, next_actions)
        target_q = rewards + (self.gamma * target_q * ~dones)
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self._update_target_networks(self.tau)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }
    
    def save_model(self, filepath: str):
        """Save the agent's models"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load the agent's models"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
