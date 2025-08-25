"""
PatternSmithAI - Procedural Pattern Generation
Handles the generation of various pattern types using mathematical rules.
"""

import math
import random
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from canvas import PatternCanvas, ColorPalette


class PatternGenerator:
    """Base class for pattern generation."""
    
    def __init__(self, canvas: PatternCanvas, color_palette: ColorPalette):
        self.canvas = canvas
        self.color_palette = color_palette
        self.rng = random.Random()
    
    def set_seed(self, seed: int):
        """Set random seed for reproducible patterns."""
        self.rng.seed(seed)
        np.random.seed(seed)
    
    def generate(self, **kwargs) -> PatternCanvas:
        """Generate a pattern. Override in subclasses."""
        raise NotImplementedError


class GeometricPatternGenerator(PatternGenerator):
    """Generates geometric patterns."""
    
    def generate(self, pattern_type: str = "random", **kwargs) -> PatternCanvas:
        """Generate geometric patterns."""
        self.canvas.clear()
        
        if pattern_type == "random":
            pattern_type = self.rng.choice(["circles", "squares", "polygons", "stars"])
        
        if pattern_type == "circles":
            self._generate_circles(**kwargs)
        elif pattern_type == "squares":
            self._generate_squares(**kwargs)
        elif pattern_type == "polygons":
            self._generate_polygons(**kwargs)
        elif pattern_type == "stars":
            self._generate_stars(**kwargs)
        
        return self.canvas
    
    def _generate_circles(self, count: int = 20, min_radius: int = 10, 
                         max_radius: int = 50, palette: str = "rainbow"):
        """Generate random circles pattern."""
        for _ in range(count):
            x = self.rng.randint(0, self.canvas.width)
            y = self.rng.randint(0, self.canvas.height)
            radius = self.rng.randint(min_radius, max_radius)
            color = self.color_palette.get_random_color(palette)
            fill = self.color_palette.get_random_color(palette) if self.rng.random() > 0.5 else None
            
            self.canvas.draw_circle(x, y, radius, color, fill)
    
    def _generate_squares(self, count: int = 15, min_size: int = 20, 
                         max_size: int = 80, palette: str = "rainbow"):
        """Generate random squares pattern."""
        for _ in range(count):
            x = self.rng.randint(0, self.canvas.width)
            y = self.rng.randint(0, self.canvas.height)
            size = self.rng.randint(min_size, max_size)
            color = self.color_palette.get_random_color(palette)
            fill = self.color_palette.get_random_color(palette) if self.rng.random() > 0.5 else None
            
            self.canvas.draw_square(x, y, size, color, fill)
    
    def _generate_polygons(self, count: int = 10, min_sides: int = 3, 
                          max_sides: int = 8, min_size: int = 30, 
                          max_size: int = 60, palette: str = "rainbow"):
        """Generate random polygons pattern."""
        for _ in range(count):
            x = self.rng.randint(0, self.canvas.width)
            y = self.rng.randint(0, self.canvas.height)
            sides = self.rng.randint(min_sides, max_sides)
            size = self.rng.randint(min_size, max_size)
            
            points = []
            for i in range(sides):
                angle = i * 2 * math.pi / sides
                px = x + size * math.cos(angle)
                py = y + size * math.sin(angle)
                points.append((px, py))
            
            color = self.color_palette.get_random_color(palette)
            fill = self.color_palette.get_random_color(palette) if self.rng.random() > 0.5 else None
            
            self.canvas.draw_polygon(points, color, fill)
    
    def _generate_stars(self, count: int = 8, min_points: int = 5, 
                       max_points: int = 8, min_outer_radius: int = 20, 
                       max_outer_radius: int = 60, palette: str = "rainbow"):
        """Generate random stars pattern."""
        for _ in range(count):
            x = self.rng.randint(0, self.canvas.width)
            y = self.rng.randint(0, self.canvas.height)
            points = self.rng.randint(min_points, max_points)
            outer_radius = self.rng.randint(min_outer_radius, max_outer_radius)
            inner_radius = outer_radius // 2
            color = self.color_palette.get_random_color(palette)
            fill = self.color_palette.get_random_color(palette) if self.rng.random() > 0.5 else None
            
            self.canvas.draw_star(x, y, outer_radius, inner_radius, points, color, fill)


class MandalaGenerator(PatternGenerator):
    """Generates mandala patterns."""
    
    def generate(self, layers: int = 8, elements_per_layer: int = 12, 
                base_shape: str = "circle", **kwargs) -> PatternCanvas:
        """Generate mandala pattern."""
        self.canvas.clear()
        
        for layer in range(layers):
            radius = 50 + layer * 40
            angle_step = 2 * math.pi / elements_per_layer
            
            for i in range(elements_per_layer):
                angle = i * angle_step
                x = self.canvas.center_x + radius * math.cos(angle)
                y = self.canvas.center_y + radius * math.sin(angle)
                
                color = self.color_palette.get_random_color("rainbow")
                fill = self.color_palette.get_random_color("rainbow") if self.rng.random() > 0.3 else None
                
                if base_shape == "circle":
                    size = 20 - layer * 2
                    self.canvas.draw_circle(x, y, size, color, fill)
                elif base_shape == "square":
                    size = 15 - layer * 1.5
                    self.canvas.draw_square(x, y, size, color, fill)
                elif base_shape == "star":
                    outer_radius = 15 - layer * 1.5
                    inner_radius = outer_radius // 2
                    self.canvas.draw_star(x, y, outer_radius, inner_radius, 5, color, fill)
        
        return self.canvas


class FractalGenerator(PatternGenerator):
    """Generates fractal patterns."""
    
    def generate(self, fractal_type: str = "sierpinski", depth: int = 4, **kwargs) -> PatternCanvas:
        """Generate fractal pattern."""
        self.canvas.clear()
        
        if fractal_type == "sierpinski":
            self._generate_sierpinski_triangle(depth, **kwargs)
        elif fractal_type == "koch":
            self._generate_koch_snowflake(depth, **kwargs)
        elif fractal_type == "tree":
            self._generate_fractal_tree(depth, **kwargs)
        
        return self.canvas
    
    def _generate_sierpinski_triangle(self, depth: int, **kwargs):
        """Generate Sierpinski triangle."""
        def sierpinski(x1, y1, x2, y2, x3, y3, current_depth):
            if current_depth <= 0:
                points = [(x1, y1), (x2, y2), (x3, y3)]
                color = self.color_palette.get_random_color("monochrome")
                self.canvas.draw_polygon(points, color, color)
                return
            
            # Calculate midpoints
            mx1, my1 = (x1 + x2) / 2, (y1 + y2) / 2
            mx2, my2 = (x2 + x3) / 2, (y2 + y3) / 2
            mx3, my3 = (x3 + x1) / 2, (y3 + y1) / 2
            
            # Recursive calls
            sierpinski(x1, y1, mx1, my1, mx3, my3, current_depth - 1)
            sierpinski(mx1, my1, x2, y2, mx2, my2, current_depth - 1)
            sierpinski(mx3, my3, mx2, my2, x3, y3, current_depth - 1)
        
        # Initial triangle
        size = 300
        x1, y1 = self.canvas.center_x, self.canvas.center_y - size // 2
        x2, y2 = self.canvas.center_x - size // 2, self.canvas.center_y + size // 2
        x3, y3 = self.canvas.center_x + size // 2, self.canvas.center_y + size // 2
        
        sierpinski(x1, y1, x2, y2, x3, y3, depth)
    
    def _generate_koch_snowflake(self, depth: int, **kwargs):
        """Generate Koch snowflake."""
        def koch_curve(points, current_depth):
            if current_depth <= 0:
                return points
            
            new_points = []
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                
                # Calculate the two new points
                dx = x2 - x1
                dy = y2 - y1
                
                x3 = x1 + dx / 3
                y3 = y1 + dy / 3
                x4 = x1 + 2 * dx / 3
                y4 = y1 + 2 * dy / 3
                
                # Calculate the peak point
                angle = math.atan2(dy, dx) - math.pi / 3
                length = math.sqrt(dx**2 + dy**2) / 3
                x5 = x3 + length * math.cos(angle)
                y5 = y3 + length * math.sin(angle)
                
                new_points.extend([(x1, y1), (x3, y3), (x5, y5), (x4, y4)])
            
            new_points.append(points[-1])
            return koch_curve(new_points, current_depth - 1)
        
        # Initial triangle
        size = 200
        points = [
            (self.canvas.center_x, self.canvas.center_y - size // 2),
            (self.canvas.center_x - size // 2, self.canvas.center_y + size // 2),
            (self.canvas.center_x + size // 2, self.canvas.center_y + size // 2),
            (self.canvas.center_x, self.canvas.center_y - size // 2)
        ]
        
        final_points = koch_curve(points, depth)
        color = self.color_palette.get_random_color("ocean")
        self.canvas.draw_curve(final_points, color, 2)
    
    def _generate_fractal_tree(self, depth: int, **kwargs):
        """Generate fractal tree."""
        def draw_tree(x, y, angle, length, current_depth):
            if current_depth <= 0:
                return
            
            # Draw trunk
            end_x = x + length * math.cos(angle)
            end_y = y + length * math.sin(angle)
            
            color = self.color_palette.get_random_color("earth")
            self.canvas.draw_curve([(x, y), (end_x, end_y)], color, current_depth)
            
            # Recursive branches
            new_length = length * 0.7
            draw_tree(end_x, end_y, angle + math.pi / 6, new_length, current_depth - 1)
            draw_tree(end_x, end_y, angle - math.pi / 6, new_length, current_depth - 1)
        
        draw_tree(self.canvas.center_x, self.canvas.height - 50, -math.pi / 2, 100, depth)


class TilingGenerator(PatternGenerator):
    """Generates tiling patterns."""
    
    def generate(self, tile_type: str = "hexagonal", tile_size: int = 50, **kwargs) -> PatternCanvas:
        """Generate tiling pattern."""
        self.canvas.clear()
        
        if tile_type == "hexagonal":
            self._generate_hexagonal_tiling(tile_size, **kwargs)
        elif tile_type == "square":
            self._generate_square_tiling(tile_size, **kwargs)
        elif tile_type == "triangular":
            self._generate_triangular_tiling(tile_size, **kwargs)
        
        return self.canvas
    
    def _generate_hexagonal_tiling(self, tile_size: int, **kwargs):
        """Generate hexagonal tiling."""
        hex_width = tile_size * 2
        hex_height = tile_size * math.sqrt(3)
        
        for row in range(-2, int(self.canvas.height / hex_height) + 3):
            for col in range(-2, int(self.canvas.width / hex_width) + 3):
                x = col * hex_width + (row % 2) * hex_width / 2
                y = row * hex_height
                
                # Generate hexagon points
                points = []
                for i in range(6):
                    angle = i * math.pi / 3
                    px = x + tile_size * math.cos(angle)
                    py = y + tile_size * math.sin(angle)
                    points.append((px, py))
                
                color = self.color_palette.get_random_color("pastel")
                fill = self.color_palette.get_random_color("pastel") if self.rng.random() > 0.5 else None
                
                self.canvas.draw_polygon(points, color, fill)
    
    def _generate_square_tiling(self, tile_size: int, **kwargs):
        """Generate square tiling."""
        for x in range(0, self.canvas.width, tile_size):
            for y in range(0, self.canvas.height, tile_size):
                color = self.color_palette.get_random_color("monochrome")
                fill = self.color_palette.get_random_color("monochrome") if self.rng.random() > 0.5 else None
                
                self.canvas.draw_square(x + tile_size//2, y + tile_size//2, tile_size, color, fill)
    
    def _generate_triangular_tiling(self, tile_size: int, **kwargs):
        """Generate triangular tiling."""
        for row in range(-1, int(self.canvas.height / (tile_size * math.sqrt(3))) + 2):
            for col in range(-1, int(self.canvas.width / tile_size) + 2):
                x = col * tile_size
                y = row * tile_size * math.sqrt(3)
                
                # Generate triangle points
                points = [
                    (x, y),
                    (x + tile_size, y),
                    (x + tile_size / 2, y + tile_size * math.sqrt(3) / 2)
                ]
                
                color = self.color_palette.get_random_color("ocean")
                fill = self.color_palette.get_random_color("ocean") if self.rng.random() > 0.5 else None
                
                self.canvas.draw_polygon(points, color, fill)


class PatternFactory:
    """Factory class for creating different types of pattern generators."""
    
    @staticmethod
    def create_generator(pattern_type: str, canvas: PatternCanvas, 
                        color_palette: ColorPalette) -> PatternGenerator:
        """Create a pattern generator based on type."""
        if pattern_type == "geometric":
            return GeometricPatternGenerator(canvas, color_palette)
        elif pattern_type == "mandala":
            return MandalaGenerator(canvas, color_palette)
        elif pattern_type == "fractal":
            return FractalGenerator(canvas, color_palette)
        elif pattern_type == "tiling":
            return TilingGenerator(canvas, color_palette)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
