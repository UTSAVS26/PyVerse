"""
Canvas module for ArtForgeAI - Painting canvas and rendering logic
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


class Canvas:
    """A digital canvas for painting with brushstrokes"""
    
    def __init__(self, width: int = 800, height: int = 600, background_color: Tuple[int, int, int] = (255, 255, 255)):
        """
        Initialize the canvas
        
        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
            background_color: RGB background color (default: white)
        """
        self.width = width
        self.height = height
        self.background_color = background_color
        self.reset()
    
    def reset(self):
        """Reset canvas to background color"""
        self.image = np.full((self.height, self.width, 3), self.background_color, dtype=np.uint8)
        self.stroke_history = []
    
    def get_image(self) -> np.ndarray:
        """Get current canvas as numpy array"""
        return self.image.copy()
    
    def get_pil_image(self) -> Image.Image:
        """Get current canvas as PIL Image"""
        return Image.fromarray(self.image)
    
    def apply_stroke(self, stroke_data: dict) -> bool:
        """
        Apply a brushstroke to the canvas
        
        Args:
            stroke_data: Dictionary containing stroke parameters
                - type: 'line', 'curve', 'dot', 'splash'
                - start_pos: (x, y) starting position
                - end_pos: (x, y) ending position (for line/curve)
                - color: (r, g, b) stroke color
                - thickness: stroke thickness
                - angle: angle for line strokes
                - radius: radius for dot/splash strokes
        
        Returns:
            bool: True if stroke was applied successfully
        """
        try:
            stroke_type = stroke_data.get('type', 'line')
            color = stroke_data.get('color', (0, 0, 0))
            thickness = stroke_data.get('thickness', 2)
            
            if stroke_type == 'line':
                self._draw_line(stroke_data, color, thickness)
            elif stroke_type == 'curve':
                self._draw_curve(stroke_data, color, thickness)
            elif stroke_type == 'dot':
                self._draw_dot(stroke_data, color, thickness)
            elif stroke_type == 'splash':
                self._draw_splash(stroke_data, color, thickness)
            else:
                return False
            
            # Record stroke in history
            self.stroke_history.append(stroke_data)
            return True
            
        except Exception as e:
            print(f"Error applying stroke: {e}")
            return False
    
    def _draw_line(self, stroke_data: dict, color: Tuple[int, int, int], thickness: int):
        """Draw a line stroke"""
        start_pos = stroke_data.get('start_pos', (0, 0))
        end_pos = stroke_data.get('end_pos', (100, 100))
        
        # Convert to integer coordinates
        start_x, start_y = int(start_pos[0]), int(start_pos[1])
        end_x, end_y = int(end_pos[0]), int(end_pos[1])
        
        # Ensure coordinates are within canvas bounds
        start_x = max(0, min(start_x, self.width - 1))
        start_y = max(0, min(start_y, self.height - 1))
        end_x = max(0, min(end_x, self.width - 1))
        end_y = max(0, min(end_y, self.height - 1))
        
        cv2.line(self.image, (start_x, start_y), (end_x, end_y), color, thickness)
    
    def _draw_curve(self, stroke_data: dict, color: Tuple[int, int, int], thickness: int):
        """Draw a curved stroke"""
        start_pos = stroke_data.get('start_pos', (0, 0))
        end_pos = stroke_data.get('end_pos', (100, 100))
        control_points = stroke_data.get('control_points', [])
        
        if len(control_points) < 2:
            # Simple curve with one control point
            mid_x = (start_pos[0] + end_pos[0]) // 2
            mid_y = (start_pos[1] + end_pos[1]) // 2
            control_points = [(mid_x, mid_y)]
        
        # Create curve points
        points = [start_pos] + control_points + [end_pos]
        points = np.array(points, dtype=np.int32)
        
        # Ensure all points are within canvas bounds
        points[:, 0] = np.clip(points[:, 0], 0, self.width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, self.height - 1)
        
        cv2.polylines(self.image, [points], False, color, thickness)
    
    def _draw_dot(self, stroke_data: dict, color: Tuple[int, int, int], thickness: int):
        """Draw a dot/blob stroke"""
        center = stroke_data.get('start_pos', (100, 100))
        radius = stroke_data.get('radius', thickness * 2)
        
        center_x, center_y = int(center[0]), int(center[1])
        radius = int(radius)
        
        # Ensure circle is within canvas bounds
        center_x = max(radius, min(center_x, self.width - radius - 1))
        center_y = max(radius, min(center_y, self.height - radius - 1))
        
        cv2.circle(self.image, (center_x, center_y), radius, color, -1)
    
    def _draw_splash(self, stroke_data: dict, color: Tuple[int, int, int], thickness: int):
        """Draw a random splash stroke"""
        center = stroke_data.get('start_pos', (100, 100))
        radius = stroke_data.get('radius', thickness * 3)
        
        center_x, center_y = int(center[0]), int(center[1])
        radius = int(radius)
        
        # Ensure splash is within canvas bounds
        center_x = max(radius, min(center_x, self.width - radius - 1))
        center_y = max(radius, min(center_y, self.height - radius - 1))
        
        # Create random splash pattern
        num_points = np.random.randint(5, 15)
        angles = np.linspace(0, 2 * np.pi, num_points)
        radii = np.random.uniform(0, radius, num_points)
        
        points = []
        for angle, r in zip(angles, radii):
            x = center_x + int(r * np.cos(angle))
            y = center_y + int(r * np.sin(angle))
            x = max(0, min(x, self.width - 1))
            y = max(0, min(y, self.height - 1))
            points.append((x, y))
        
        if len(points) > 2:
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(self.image, [points], color)
    
    def get_coverage(self) -> float:
        """Calculate the percentage of canvas covered by strokes"""
        # Count non-background pixels
        background_mask = np.all(self.image == self.background_color, axis=2)
        covered_pixels = np.sum(~background_mask)
        total_pixels = self.width * self.height
        return covered_pixels / total_pixels
    
    def get_color_diversity(self) -> float:
        """Calculate color diversity score"""
        if len(self.stroke_history) == 0:
            return 0.0
        
        colors = [stroke['color'] for stroke in self.stroke_history]
        unique_colors = len(set(colors))
        return unique_colors / len(colors)
    
    def save_image(self, filename: str):
        """Save canvas to file"""
        pil_image = self.get_pil_image()
        pil_image.save(filename)
    
    def display(self, title: str = "ArtForgeAI Canvas"):
        """Display canvas using matplotlib"""
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()
    
    def get_stats(self) -> dict:
        """Get canvas statistics"""
        return {
            'width': self.width,
            'height': self.height,
            'num_strokes': len(self.stroke_history),
            'coverage': self.get_coverage(),
            'color_diversity': self.get_color_diversity(),
            'stroke_types': [stroke['type'] for stroke in self.stroke_history]
        }
