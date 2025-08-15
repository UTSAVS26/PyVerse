import numpy as np
from typing import Dict, Any, List, Tuple

class StateEncoder:
    """Utility class for encoding game state into standardized inputs for AI agents."""
    
    def __init__(self, game_width: int = 800, game_height: int = 600):
        self.game_width = game_width
        self.game_height = game_height
        
    def encode_state(self, state: Dict[str, Any]) -> List[float]:
        """
        Encode game state into a standardized input vector.
        
        Args:
            state: Game state dictionary from FlappyBirdGame
            
        Returns:
            List of float values representing the encoded state
        """
        # Extract basic state information
        bird_y = state.get('bird_y', 0.5)
        bird_velocity = state.get('bird_velocity', 0.0)
        bird_alive = state.get('bird_alive', 1.0)
        
        # Extract pipe information
        pipe_x = state.get('pipe_x', 1.0)
        pipe_gap_y = state.get('pipe_gap_y', 0.5)
        pipe_gap_size = state.get('pipe_gap_size', 0.25)
        distance_to_pipe = state.get('distance_to_pipe', 1.0)
        
        # Create normalized input vector
        inputs = [
            bird_y,                    # Bird's Y position (normalized)
            bird_velocity,             # Bird's velocity (normalized)
            pipe_x,                    # Closest pipe X position (normalized)
            pipe_gap_y,                # Pipe gap Y position (normalized)
            pipe_gap_size,             # Pipe gap size (normalized)
            distance_to_pipe,          # Distance to closest pipe (normalized)
            bird_alive                 # Bird alive status (1.0 or 0.0)
        ]
        
        return inputs
        
    def encode_state_advanced(self, state: Dict[str, Any]) -> List[float]:
        """
        Encode game state with additional features for more sophisticated agents.
        
        Args:
            state: Game state dictionary from FlappyBirdGame
            
        Returns:
            List of float values representing the advanced encoded state
        """
        # Basic features
        bird_y = state.get('bird_y', 0.5)
        bird_velocity = state.get('bird_velocity', 0.0)
        bird_alive = state.get('bird_alive', 1.0)
        
        # Pipe features
        pipe_x = state.get('pipe_x', 1.0)
        pipe_gap_y = state.get('pipe_gap_y', 0.5)
        pipe_gap_size = state.get('pipe_gap_size', 0.25)
        distance_to_pipe = state.get('distance_to_pipe', 1.0)
        
        # Additional derived features
        bird_to_gap_center = abs(bird_y - pipe_gap_y - pipe_gap_size/2)
        bird_to_gap_top = abs(bird_y - pipe_gap_y)
        bird_to_gap_bottom = abs(bird_y - (pipe_gap_y + pipe_gap_size))
        
        # Velocity features
        velocity_positive = max(0, bird_velocity)
        velocity_negative = max(0, -bird_velocity)
        
        # Safety features
        safety_margin = min(bird_to_gap_top, bird_to_gap_bottom) / pipe_gap_size
        
        # Create advanced input vector
        inputs = [
            bird_y,                    # Bird's Y position
            bird_velocity,             # Bird's velocity
            pipe_x,                    # Closest pipe X position
            pipe_gap_y,                # Pipe gap Y position
            pipe_gap_size,             # Pipe gap size
            distance_to_pipe,          # Distance to closest pipe
            bird_alive,                # Bird alive status
            bird_to_gap_center,        # Distance to gap center
            bird_to_gap_top,           # Distance to gap top
            bird_to_gap_bottom,        # Distance to gap bottom
            velocity_positive,          # Positive velocity component
            velocity_negative,          # Negative velocity component
            safety_margin              # Safety margin
        ]
        
        return inputs
        
    def decode_action(self, action: int) -> str:
        """
        Decode action integer to human-readable string.
        
        Args:
            action: Action integer (0 or 1)
            
        Returns:
            String description of the action
        """
        return "FLAP" if action == 1 else "NO_FLAP"
        
    def get_state_description(self, state: Dict[str, Any]) -> str:
        """
        Get a human-readable description of the game state.
        
        Args:
            state: Game state dictionary
            
        Returns:
            String description of the state
        """
        bird_y = state.get('bird_y', 0.5)
        bird_velocity = state.get('bird_velocity', 0.0)
        pipe_x = state.get('pipe_x', 1.0)
        pipe_gap_y = state.get('pipe_gap_y', 0.5)
        bird_alive = state.get('bird_alive', 1.0)
        
        description = f"Bird Y: {bird_y:.2f}, "
        description += f"Velocity: {bird_velocity:.2f}, "
        description += f"Pipe X: {pipe_x:.2f}, "
        description += f"Gap Y: {pipe_gap_y:.2f}, "
        description += f"Alive: {bird_alive}"
        
        return description
        
    def normalize_value(self, value: float, min_val: float, max_val: float) -> float:
        """
        Normalize a value to [0, 1] range.
        
        Args:
            value: Value to normalize
            min_val: Minimum value in range
            max_val: Maximum value in range
            
        Returns:
            Normalized value in [0, 1]
        """
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)
        
    def denormalize_value(self, normalized_value: float, min_val: float, max_val: float) -> float:
        """
        Denormalize a value from [0, 1] range.
        
        Args:
            normalized_value: Normalized value in [0, 1]
            min_val: Minimum value in range
            max_val: Maximum value in range
            
        Returns:
            Denormalized value
        """
        return min_val + normalized_value * (max_val - min_val)

def create_state_encoder(game_width: int = 800, game_height: int = 600) -> StateEncoder:
    """
    Factory function to create a state encoder.
    
    Args:
        game_width: Width of the game window
        game_height: Height of the game window
        
    Returns:
        StateEncoder instance
    """
    return StateEncoder(game_width, game_height) 