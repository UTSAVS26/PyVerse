"""
Strokes module for ArtForgeAI - Primitive brushstroke implementations
"""

import numpy as np
import math
from typing import Tuple, List, Dict, Any
import random


class StrokeGenerator:
    """Generator for different types of brushstrokes"""
    
    def __init__(self, canvas_width: int, canvas_height: int):
        """
        Initialize stroke generator
        
        Args:
            canvas_width: Width of the canvas
            canvas_height: Height of the canvas
        """
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        # Predefined color palettes
        self.color_palettes = {
            'monochrome': [(0, 0, 0), (64, 64, 64), (128, 128, 128), (192, 192, 192)],
            'warm': [(255, 0, 0), (255, 165, 0), (255, 255, 0), (255, 192, 203)],
            'cool': [(0, 0, 255), (0, 255, 255), (128, 0, 128), (0, 255, 0)],
            'earth': [(139, 69, 19), (160, 82, 45), (210, 180, 140), (244, 164, 96)],
            'vibrant': [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        }
    
    def generate_random_color(self, palette: str = 'vibrant') -> Tuple[int, int, int]:
        """Generate a random color from the specified palette"""
        if palette in self.color_palettes:
            return random.choice(self.color_palettes[palette])
        else:
            # Generate random RGB color
            return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    def generate_line_stroke(self, **kwargs) -> Dict[str, Any]:
        """
        Generate a line stroke
        
        Args:
            start_pos: Starting position (x, y) - if None, random
            end_pos: Ending position (x, y) - if None, random
            color: RGB color tuple - if None, random
            thickness: Stroke thickness - if None, random
            angle: Angle in degrees - if None, random
            palette: Color palette name
        
        Returns:
            Dictionary with stroke parameters
        """
        start_pos = kwargs.get('start_pos')
        end_pos = kwargs.get('end_pos')
        color = kwargs.get('color')
        thickness = kwargs.get('thickness')
        angle = kwargs.get('angle')
        palette = kwargs.get('palette', 'vibrant')
        
        # Generate random start position if not provided
        if start_pos is None:
            start_pos = (random.randint(0, self.canvas_width), 
                        random.randint(0, self.canvas_height))
        
        # Generate random angle if not provided
        if angle is None:
            angle = random.uniform(0, 360)
        
        # Generate end position based on angle and random length
        if end_pos is None:
            length = random.randint(20, min(self.canvas_width, self.canvas_height) // 4)
            angle_rad = math.radians(angle)
            end_x = start_pos[0] + int(length * math.cos(angle_rad))
            end_y = start_pos[1] + int(length * math.sin(angle_rad))
            
            # Ensure end position is within canvas bounds
            end_x = max(0, min(end_x, self.canvas_width - 1))
            end_y = max(0, min(end_y, self.canvas_height - 1))
            end_pos = (end_x, end_y)
        
        # Generate random color if not provided
        if color is None:
            color = self.generate_random_color(palette)
        
        # Generate random thickness if not provided
        if thickness is None:
            thickness = random.randint(1, 8)
        
        return {
            'type': 'line',
            'start_pos': start_pos,
            'end_pos': end_pos,
            'color': color,
            'thickness': thickness,
            'angle': angle
        }
    
    def generate_curve_stroke(self, **kwargs) -> Dict[str, Any]:
        """
        Generate a curved stroke
        
        Args:
            start_pos: Starting position (x, y) - if None, random
            end_pos: Ending position (x, y) - if None, random
            color: RGB color tuple - if None, random
            thickness: Stroke thickness - if None, random
            control_points: List of control points - if None, random
            palette: Color palette name
        
        Returns:
            Dictionary with stroke parameters
        """
        start_pos = kwargs.get('start_pos')
        end_pos = kwargs.get('end_pos')
        color = kwargs.get('color')
        thickness = kwargs.get('thickness')
        control_points = kwargs.get('control_points')
        palette = kwargs.get('palette', 'vibrant')
        
        # Generate random start position if not provided
        if start_pos is None:
            start_pos = (random.randint(0, self.canvas_width), 
                        random.randint(0, self.canvas_height))
        
        # Generate random end position if not provided
        if end_pos is None:
            end_pos = (random.randint(0, self.canvas_width), 
                      random.randint(0, self.canvas_height))
        
        # Generate random control points if not provided
        if control_points is None:
            num_control_points = random.randint(1, 3)
            control_points = []
            
            for _ in range(num_control_points):
                # Generate control point between start and end
                t = random.uniform(0.2, 0.8)
                base_x = start_pos[0] + t * (end_pos[0] - start_pos[0])
                base_y = start_pos[1] + t * (end_pos[1] - start_pos[1])
                
                # Add some random offset
                offset_x = random.randint(-50, 50)
                offset_y = random.randint(-50, 50)
                
                control_x = int(base_x + offset_x)
                control_y = int(base_y + offset_y)
                
                # Ensure control point is within canvas bounds
                control_x = max(0, min(control_x, self.canvas_width - 1))
                control_y = max(0, min(control_y, self.canvas_height - 1))
                
                control_points.append((control_x, control_y))
        
        # Generate random color if not provided
        if color is None:
            color = self.generate_random_color(palette)
        
        # Generate random thickness if not provided
        if thickness is None:
            thickness = random.randint(1, 6)
        
        return {
            'type': 'curve',
            'start_pos': start_pos,
            'end_pos': end_pos,
            'control_points': control_points,
            'color': color,
            'thickness': thickness
        }
    
    def generate_dot_stroke(self, **kwargs) -> Dict[str, Any]:
        """
        Generate a dot/blob stroke
        
        Args:
            center: Center position (x, y) - if None, random
            color: RGB color tuple - if None, random
            radius: Dot radius - if None, random
            palette: Color palette name
        
        Returns:
            Dictionary with stroke parameters
        """
        center = kwargs.get('center')
        color = kwargs.get('color')
        radius = kwargs.get('radius')
        palette = kwargs.get('palette', 'vibrant')
        
        # Generate random radius if not provided
        if radius is None:
            radius = random.randint(3, 20)
        
        # Generate random center position if not provided
        if center is None:
            # Ensure center allows the entire dot to be drawn within bounds
            min_x = radius
            max_x = self.canvas_width - radius - 1
            min_y = radius
            max_y = self.canvas_height - radius - 1
            
            # Handle case where canvas is too small for the radius
            if min_x > max_x:
                min_x = max_x = self.canvas_width // 2
            if min_y > max_y:
                min_y = max_y = self.canvas_height // 2
                
            center = (random.randint(min_x, max_x), 
                     random.randint(min_y, max_y))
        
        # Generate random color if not provided
        if color is None:
            color = self.generate_random_color(palette)
        
        return {
            'type': 'dot',
            'start_pos': center,  # Use start_pos for consistency
            'color': color,
            'radius': radius,
            'thickness': 1  # Not used for dots
        }
    
    def generate_splash_stroke(self, **kwargs) -> Dict[str, Any]:
        """
        Generate a splash stroke
        
        Args:
            center: Center position (x, y) - if None, random
            color: RGB color tuple - if None, random
            radius: Splash radius - if None, random
            palette: Color palette name
        
        Returns:
            Dictionary with stroke parameters
        """
        center = kwargs.get('center')
        color = kwargs.get('color')
        radius = kwargs.get('radius')
        palette = kwargs.get('palette', 'vibrant')
        
        # Generate random radius if not provided
        if radius is None:
            radius = random.randint(10, 40)
        
        # Generate random center position if not provided
        if center is None:
            # Ensure center allows the entire splash to be drawn within bounds
            min_x = radius
            max_x = self.canvas_width - radius - 1
            min_y = radius
            max_y = self.canvas_height - radius - 1
            
            # Handle case where canvas is too small for the radius
            if min_x > max_x:
                min_x = max_x = self.canvas_width // 2
            if min_y > max_y:
                min_y = max_y = self.canvas_height // 2
                
            center = (random.randint(min_x, max_x), 
                     random.randint(min_y, max_y))
        
        # Generate random color if not provided
        if color is None:
            color = self.generate_random_color(palette)
        
        return {
            'type': 'splash',
            'start_pos': center,  # Use start_pos for consistency
            'color': color,
            'radius': radius,
            'thickness': 1  # Not used for splashes
        }
    
    def generate_random_stroke(self, stroke_types: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate a random stroke of any type
        
        Args:
            stroke_types: List of stroke types to choose from
            **kwargs: Additional parameters passed to specific stroke generators
        
        Returns:
            Dictionary with stroke parameters
        """
        if stroke_types is None:
            stroke_types = ['line', 'curve', 'dot', 'splash']
        
        stroke_type = random.choice(stroke_types)
        
        if stroke_type == 'line':
            return self.generate_line_stroke(**kwargs)
        elif stroke_type == 'curve':
            return self.generate_curve_stroke(**kwargs)
        elif stroke_type == 'dot':
            return self.generate_dot_stroke(**kwargs)
        elif stroke_type == 'splash':
            return self.generate_splash_stroke(**kwargs)
        else:
            raise ValueError(f"Unknown stroke type: {stroke_type}")
    
    def generate_stroke_sequence(self, num_strokes: int, stroke_types: List[str] = None, 
                                palette: str = 'vibrant') -> List[Dict[str, Any]]:
        """
        Generate a sequence of random strokes
        
        Args:
            num_strokes: Number of strokes to generate
            stroke_types: List of stroke types to choose from
            palette: Color palette name
        
        Returns:
            List of stroke dictionaries
        """
        strokes = []
        for _ in range(num_strokes):
            stroke = self.generate_random_stroke(stroke_types, palette=palette)
            strokes.append(stroke)
        return strokes
    
    def get_stroke_bounds(self, stroke: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """
        Get the bounding box of a stroke
        
        Args:
            stroke: Stroke dictionary
        
        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        stroke_type = stroke['type']
        
        if stroke_type == 'line':
            start_pos = stroke['start_pos']
            end_pos = stroke['end_pos']
            min_x = min(start_pos[0], end_pos[0])
            min_y = min(start_pos[1], end_pos[1])
            max_x = max(start_pos[0], end_pos[0])
            max_y = max(start_pos[1], end_pos[1])
            
        elif stroke_type == 'curve':
            points = [stroke['start_pos']] + stroke['control_points'] + [stroke['end_pos']]
            min_x = min(p[0] for p in points)
            min_y = min(p[1] for p in points)
            max_x = max(p[0] for p in points)
            max_y = max(p[1] for p in points)
            
        elif stroke_type in ['dot', 'splash']:
            center = stroke['start_pos']
            radius = stroke['radius']
            min_x = center[0] - radius
            min_y = center[1] - radius
            max_x = center[0] + radius
            max_y = center[1] + radius
            
        else:
            raise ValueError(f"Unknown stroke type: {stroke_type}")
        
        return (min_x, min_y, max_x, max_y)
