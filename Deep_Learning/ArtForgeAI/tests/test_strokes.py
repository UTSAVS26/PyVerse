"""
Tests for the strokes module
"""

import pytest
import numpy as np
import math

# Import the modules to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strokes import StrokeGenerator


class TestStrokeGenerator:
    """Test cases for the StrokeGenerator class"""
    
    def test_stroke_generator_initialization(self):
        """Test stroke generator initialization"""
        generator = StrokeGenerator(800, 600)
        assert generator.canvas_width == 800
        assert generator.canvas_height == 600
        assert 'monochrome' in generator.color_palettes
        assert 'warm' in generator.color_palettes
        assert 'cool' in generator.color_palettes
        assert 'earth' in generator.color_palettes
        assert 'vibrant' in generator.color_palettes
    
    def test_generate_random_color(self):
        """Test random color generation"""
        generator = StrokeGenerator(800, 600)
        
        # Test with known palette
        color = generator.generate_random_color('warm')
        assert color in generator.color_palettes['warm']
        
        # Test with unknown palette (should generate random RGB)
        color = generator.generate_random_color('unknown_palette')
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)
    
    def test_generate_line_stroke(self):
        """Test line stroke generation"""
        generator = StrokeGenerator(400, 300)
        
        # Test with default parameters
        stroke = generator.generate_line_stroke()
        assert stroke['type'] == 'line'
        assert 'start_pos' in stroke
        assert 'end_pos' in stroke
        assert 'color' in stroke
        assert 'thickness' in stroke
        assert 'angle' in stroke
        
        # Test with custom parameters
        stroke = generator.generate_line_stroke(
            start_pos=(100, 100),
            end_pos=(200, 200),
            color=(255, 0, 0),
            thickness=5,
            angle=45
        )
        assert stroke['start_pos'] == (100, 100)
        assert stroke['end_pos'] == (200, 200)
        assert stroke['color'] == (255, 0, 0)
        assert stroke['thickness'] == 5
        assert stroke['angle'] == 45
    
    def test_generate_line_stroke_bounds(self):
        """Test line stroke generation respects canvas bounds"""
        generator = StrokeGenerator(100, 100)
        
        # Test with position outside bounds
        stroke = generator.generate_line_stroke(start_pos=(150, 150))
        
        # End position should be within bounds
        end_x, end_y = stroke['end_pos']
        assert 0 <= end_x < 100
        assert 0 <= end_y < 100
    
    def test_generate_curve_stroke(self):
        """Test curve stroke generation"""
        generator = StrokeGenerator(400, 300)
        
        # Test with default parameters
        stroke = generator.generate_curve_stroke()
        assert stroke['type'] == 'curve'
        assert 'start_pos' in stroke
        assert 'end_pos' in stroke
        assert 'control_points' in stroke
        assert 'color' in stroke
        assert 'thickness' in stroke
        
        # Test with custom parameters
        control_points = [(150, 150), (200, 100)]
        stroke = generator.generate_curve_stroke(
            start_pos=(50, 50),
            end_pos=(250, 250),
            control_points=control_points,
            color=(0, 255, 0),
            thickness=3
        )
        assert stroke['start_pos'] == (50, 50)
        assert stroke['end_pos'] == (250, 250)
        assert stroke['control_points'] == control_points
        assert stroke['color'] == (0, 255, 0)
        assert stroke['thickness'] == 3
    
    def test_generate_curve_stroke_control_points(self):
        """Test curve stroke generates control points when not provided"""
        generator = StrokeGenerator(400, 300)
        
        stroke = generator.generate_curve_stroke(
            start_pos=(50, 50),
            end_pos=(350, 250)
        )
        
        assert len(stroke['control_points']) >= 1
        assert len(stroke['control_points']) <= 3
        
        # Check control points are within bounds
        for point in stroke['control_points']:
            x, y = point
            assert 0 <= x < 400
            assert 0 <= y < 300
    
    def test_generate_dot_stroke(self):
        """Test dot stroke generation"""
        generator = StrokeGenerator(400, 300)
        
        # Test with default parameters
        stroke = generator.generate_dot_stroke()
        assert stroke['type'] == 'dot'
        assert 'start_pos' in stroke
        assert 'color' in stroke
        assert 'radius' in stroke
        assert 'thickness' in stroke
        
        # Test with custom parameters
        stroke = generator.generate_dot_stroke(
            center=(200, 150),
            color=(0, 0, 255),
            radius=15
        )
        assert stroke['start_pos'] == (200, 150)  # center becomes start_pos
        assert stroke['color'] == (0, 0, 255)
        assert stroke['radius'] == 15
        assert stroke['thickness'] == 1
    
    def test_generate_dot_stroke_bounds(self):
        """Test dot stroke generation respects canvas bounds"""
        generator = StrokeGenerator(100, 100)
        
        # Test with center within bounds
        stroke = generator.generate_dot_stroke(center=(50, 50))
        
        # Center should be within bounds
        center_x, center_y = stroke['start_pos']
        assert 0 <= center_x < 100
        assert 0 <= center_y < 100
    
    def test_generate_splash_stroke(self):
        """Test splash stroke generation"""
        generator = StrokeGenerator(400, 300)
        
        # Test with default parameters
        stroke = generator.generate_splash_stroke()
        assert stroke['type'] == 'splash'
        assert 'start_pos' in stroke
        assert 'color' in stroke
        assert 'radius' in stroke
        assert 'thickness' in stroke
        
        # Test with custom parameters
        stroke = generator.generate_splash_stroke(
            center=(200, 150),
            color=(255, 255, 0),
            radius=25
        )
        assert stroke['start_pos'] == (200, 150)  # center becomes start_pos
        assert stroke['color'] == (255, 255, 0)
        assert stroke['radius'] == 25
        assert stroke['thickness'] == 1
    
    def test_generate_splash_stroke_bounds(self):
        """Test splash stroke generation respects canvas bounds"""
        generator = StrokeGenerator(100, 100)
        
        # Test with center within bounds
        stroke = generator.generate_splash_stroke(center=(50, 50))
        
        # Center should be within bounds
        center_x, center_y = stroke['start_pos']
        assert 0 <= center_x < 100
        assert 0 <= center_y < 100
    
    def test_generate_random_stroke(self):
        """Test random stroke generation"""
        generator = StrokeGenerator(400, 300)
        
        # Test with default stroke types
        stroke = generator.generate_random_stroke()
        assert stroke['type'] in ['line', 'curve', 'dot', 'splash']
        
        # Test with specific stroke types
        stroke = generator.generate_random_stroke(stroke_types=['line', 'dot'])
        assert stroke['type'] in ['line', 'dot']
        
        # Test with single stroke type
        stroke = generator.generate_random_stroke(stroke_types=['curve'])
        assert stroke['type'] == 'curve'
    
    def test_generate_random_stroke_invalid_type(self):
        """Test random stroke generation with invalid stroke type"""
        generator = StrokeGenerator(400, 300)
        
        with pytest.raises(ValueError):
            generator.generate_random_stroke(stroke_types=['invalid_type'])
    
    def test_generate_stroke_sequence(self):
        """Test stroke sequence generation"""
        generator = StrokeGenerator(400, 300)
        
        # Test sequence generation
        strokes = generator.generate_stroke_sequence(5)
        assert len(strokes) == 5
        
        # Check all strokes are valid
        for stroke in strokes:
            assert stroke['type'] in ['line', 'curve', 'dot', 'splash']
            assert 'color' in stroke
        
        # Test with specific stroke types
        strokes = generator.generate_stroke_sequence(3, stroke_types=['line', 'dot'])
        assert len(strokes) == 3
        for stroke in strokes:
            assert stroke['type'] in ['line', 'dot']
        
        # Test with specific palette
        strokes = generator.generate_stroke_sequence(2, palette='warm')
        assert len(strokes) == 2
        for stroke in strokes:
            assert stroke['color'] in generator.color_palettes['warm']
    
    def test_get_stroke_bounds_line(self):
        """Test getting bounds for line stroke"""
        generator = StrokeGenerator(400, 300)
        
        stroke = {
            'type': 'line',
            'start_pos': (50, 50),
            'end_pos': (150, 100)
        }
        
        bounds = generator.get_stroke_bounds(stroke)
        min_x, min_y, max_x, max_y = bounds
        
        assert min_x == 50
        assert min_y == 50
        assert max_x == 150
        assert max_y == 100
    
    def test_get_stroke_bounds_curve(self):
        """Test getting bounds for curve stroke"""
        generator = StrokeGenerator(400, 300)
        
        stroke = {
            'type': 'curve',
            'start_pos': (50, 50),
            'end_pos': (150, 100),
            'control_points': [(100, 25), (75, 75)]
        }
        
        bounds = generator.get_stroke_bounds(stroke)
        min_x, min_y, max_x, max_y = bounds
        
        assert min_x == 50
        assert min_y == 25
        assert max_x == 150
        assert max_y == 100
    
    def test_get_stroke_bounds_dot(self):
        """Test getting bounds for dot stroke"""
        generator = StrokeGenerator(400, 300)
        
        stroke = {
            'type': 'dot',
            'start_pos': (100, 100),
            'radius': 15
        }
        
        bounds = generator.get_stroke_bounds(stroke)
        min_x, min_y, max_x, max_y = bounds
        
        assert min_x == 85  # 100 - 15
        assert min_y == 85  # 100 - 15
        assert max_x == 115  # 100 + 15
        assert max_y == 115  # 100 + 15
    
    def test_get_stroke_bounds_splash(self):
        """Test getting bounds for splash stroke"""
        generator = StrokeGenerator(400, 300)
        
        stroke = {
            'type': 'splash',
            'start_pos': (100, 100),
            'radius': 20
        }
        
        bounds = generator.get_stroke_bounds(stroke)
        min_x, min_y, max_x, max_y = bounds
        
        assert min_x == 80  # 100 - 20
        assert min_y == 80  # 100 - 20
        assert max_x == 120  # 100 + 20
        assert max_y == 120  # 100 + 20
    
    def test_get_stroke_bounds_invalid_type(self):
        """Test getting bounds for invalid stroke type"""
        generator = StrokeGenerator(400, 300)
        
        stroke = {
            'type': 'invalid_type',
            'start_pos': (100, 100)
        }
        
        with pytest.raises(ValueError):
            generator.get_stroke_bounds(stroke)
    
    def test_color_palettes(self):
        """Test all color palettes"""
        generator = StrokeGenerator(400, 300)
        
        # Test monochrome palette
        color = generator.generate_random_color('monochrome')
        assert color in generator.color_palettes['monochrome']
        
        # Test warm palette
        color = generator.generate_random_color('warm')
        assert color in generator.color_palettes['warm']
        
        # Test cool palette
        color = generator.generate_random_color('cool')
        assert color in generator.color_palettes['cool']
        
        # Test earth palette
        color = generator.generate_random_color('earth')
        assert color in generator.color_palettes['earth']
        
        # Test vibrant palette
        color = generator.generate_random_color('vibrant')
        assert color in generator.color_palettes['vibrant']
    
    def test_stroke_parameters_consistency(self):
        """Test that stroke parameters are consistent across types"""
        generator = StrokeGenerator(400, 300)
        
        # Test line stroke
        line_stroke = generator.generate_line_stroke()
        assert 'type' in line_stroke
        assert 'start_pos' in line_stroke
        assert 'color' in line_stroke
        assert 'thickness' in line_stroke
        
        # Test curve stroke
        curve_stroke = generator.generate_curve_stroke()
        assert 'type' in curve_stroke
        assert 'start_pos' in curve_stroke
        assert 'color' in curve_stroke
        assert 'thickness' in curve_stroke
        
        # Test dot stroke
        dot_stroke = generator.generate_dot_stroke()
        assert 'type' in dot_stroke
        assert 'start_pos' in dot_stroke
        assert 'color' in dot_stroke
        assert 'thickness' in dot_stroke
        
        # Test splash stroke
        splash_stroke = generator.generate_splash_stroke()
        assert 'type' in splash_stroke
        assert 'start_pos' in splash_stroke
        assert 'color' in splash_stroke
        assert 'thickness' in splash_stroke
    
    def test_randomness_in_strokes(self):
        """Test that strokes have some randomness"""
        generator = StrokeGenerator(400, 300)
        
        # Generate multiple strokes and check they're not all identical
        strokes1 = [generator.generate_random_stroke() for _ in range(5)]
        strokes2 = [generator.generate_random_stroke() for _ in range(5)]
        
        # At least some strokes should be different
        assert strokes1 != strokes2
    
    def test_stroke_sequence_diversity(self):
        """Test that stroke sequences have diversity"""
        generator = StrokeGenerator(400, 300)
        
        # Generate a sequence and check for diversity
        strokes = generator.generate_stroke_sequence(10)
        stroke_types = [stroke['type'] for stroke in strokes]
        
        # Should have multiple stroke types (unless very unlucky)
        unique_types = set(stroke_types)
        assert len(unique_types) >= 2  # At least 2 different types


if __name__ == "__main__":
    pytest.main([__file__])
