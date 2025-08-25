"""
PatternSmithAI - Canvas Drawing Engine
Handles shape rendering, symmetry operations, and transformations.
"""

import math
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict, Any, Optional
import json


class PatternCanvas:
    """Main canvas class for drawing patterns and shapes."""
    
    def __init__(self, width: int = 800, height: int = 800, bg_color: str = "white"):
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.image = Image.new('RGB', (width, height), bg_color)
        self.draw = ImageDraw.Draw(self.image)
        self.center_x = width // 2
        self.center_y = height // 2
        
    def clear(self):
        """Clear the canvas."""
        self.image = Image.new('RGB', (self.width, self.height), self.bg_color)
        self.draw = ImageDraw.Draw(self.image)
    
    def draw_circle(self, x: int, y: int, radius: int, color: str = "black", fill: str = None):
        """Draw a circle."""
        bbox = [x - radius, y - radius, x + radius, y + radius]
        self.draw.ellipse(bbox, fill=fill, outline=color)
    
    def draw_square(self, x: int, y: int, size: int, color: str = "black", fill: str = None):
        """Draw a square."""
        bbox = [x - size//2, y - size//2, x + size//2, y + size//2]
        self.draw.rectangle(bbox, fill=fill, outline=color)
    
    def draw_polygon(self, points: List[Tuple[int, int]], color: str = "black", fill: str = None):
        """Draw a polygon."""
        self.draw.polygon(points, fill=fill, outline=color)
    
    def draw_star(self, x: int, y: int, outer_radius: int, inner_radius: int, 
                  points: int = 5, color: str = "black", fill: str = None):
        """Draw a star."""
        star_points = []
        for i in range(points * 2):
            angle = i * math.pi / points
            radius = outer_radius if i % 2 == 0 else inner_radius
            px = x + radius * math.cos(angle)
            py = y + radius * math.sin(angle)
            star_points.append((px, py))
        self.draw.polygon(star_points, fill=fill, outline=color)
    
    def draw_curve(self, points: List[Tuple[int, int]], color: str = "black", width: int = 1):
        """Draw a curve through points."""
        if len(points) < 2:
            return
        self.draw.line(points, fill=color, width=width)
    
    def apply_rotation(self, points: List[Tuple[int, int]], center: Tuple[int, int], 
                      angle: float) -> List[Tuple[int, int]]:
        """Apply rotation transformation to points."""
        cx, cy = center
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        rotated_points = []
        for x, y in points:
            dx = x - cx
            dy = y - cy
            new_x = cx + dx * cos_a - dy * sin_a
            new_y = cy + dx * sin_a + dy * cos_a
            rotated_points.append((new_x, new_y))
        
        return rotated_points
    
    def apply_reflection(self, points: List[Tuple[int, int]], 
                        line_start: Tuple[int, int], line_end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Apply reflection transformation across a line."""
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Calculate line equation: ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        
        reflected_points = []
        for x, y in points:
            # Distance from point to line
            dist = (a * x + b * y + c) / math.sqrt(a * a + b * b)
            
            # Reflected point
            new_x = x - 2 * a * dist / math.sqrt(a * a + b * b)
            new_y = y - 2 * b * dist / math.sqrt(a * a + b * b)
            reflected_points.append((new_x, new_y))
        
        return reflected_points
    
    def create_tiling(self, shape_func, *args, **kwargs):
        """Create a tiling pattern by repeating a shape."""
        tile_size = kwargs.get('tile_size', 100)
        for x in range(0, self.width, tile_size):
            for y in range(0, self.height, tile_size):
                shape_func(x + tile_size//2, y + tile_size//2, *args, **kwargs)
    
    def create_mandala(self, base_shape_func, layers: int = 5, 
                      rotation_angle: float = math.pi / 6, **kwargs):
        """Create a mandala by rotating a base shape."""
        for layer in range(layers):
            for i in range(int(2 * math.pi / rotation_angle)):
                angle = i * rotation_angle
                x = self.center_x + layer * 50 * math.cos(angle)
                y = self.center_y + layer * 50 * math.sin(angle)
                base_shape_func(x, y, **kwargs)
    
    def create_fractal(self, shape_func, depth: int = 3, scale_factor: float = 0.5, **kwargs):
        """Create a fractal pattern by recursively scaling shapes."""
        def fractal_recursive(x, y, size, current_depth):
            if current_depth <= 0:
                return
            
            shape_func(x, y, size, **kwargs)
            
            if current_depth > 1:
                new_size = size * scale_factor
                offsets = [(0, -size), (size, 0), (0, size), (-size, 0)]
                for dx, dy in offsets:
                    fractal_recursive(x + dx, y + dy, new_size, current_depth - 1)
        
        fractal_recursive(self.center_x, self.center_y, 100, depth)
    
    def save(self, filename: str, format: str = "PNG"):
        """Save the canvas to a file."""
        self.image.save(filename, format)
    
    def get_image(self) -> Image.Image:
        """Get the current image."""
        return self.image.copy()
    
    def get_pixel_data(self) -> np.ndarray:
        """Get pixel data as numpy array."""
        return np.array(self.image)


class ColorPalette:
    """Manages color palettes for patterns."""
    
    def __init__(self):
        self.palettes = {
            'rainbow': ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#9400D3'],
            'pastel': ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFB3F7', '#F7B3FF'],
            'monochrome': ['#000000', '#333333', '#666666', '#999999', '#CCCCCC', '#FFFFFF'],
            'earth': ['#8B4513', '#A0522D', '#CD853F', '#DEB887', '#F5DEB3', '#F4A460'],
            'ocean': ['#000080', '#0000CD', '#4169E1', '#87CEEB', '#B0E0E6', '#E0F6FF']
        }
    
    def get_palette(self, name: str) -> List[str]:
        """Get a specific color palette."""
        return self.palettes.get(name, self.palettes['rainbow'])
    
    def get_random_color(self, palette_name: str = 'rainbow') -> str:
        """Get a random color from a palette."""
        palette = self.get_palette(palette_name)
        return np.random.choice(palette)
    
    def generate_harmonic_colors(self, base_color: str, count: int = 5) -> List[str]:
        """Generate harmonic colors based on a base color."""
        # Simple harmonic color generation
        colors = [base_color]
        for i in range(1, count):
            # Create variations by adjusting hue
            colors.append(f"#{base_color[1:3]}{base_color[3:5]}{base_color[5:7]}")
        return colors
