"""
Tests for PatternSmithAI Canvas Module
Tests all drawing functions, transformations, and color palette functionality.
"""

import pytest
import numpy as np
import os
import tempfile
from PIL import Image
import math

from canvas import PatternCanvas, ColorPalette


class TestPatternCanvas:
    """Test cases for PatternCanvas class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.canvas = PatternCanvas(400, 400, "white")
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up after tests."""
        # Clean up temporary files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
    
    def test_canvas_initialization(self):
        """Test canvas initialization."""
        assert self.canvas.width == 400
        assert self.canvas.height == 400
        assert self.canvas.bg_color == "white"
        assert self.canvas.center_x == 200
        assert self.canvas.center_y == 200
        assert self.canvas.image.size == (400, 400)
    
    def test_canvas_clear(self):
        """Test canvas clearing."""
        # Draw something first
        self.canvas.draw_circle(100, 100, 20, "red")
        
        # Clear canvas
        self.canvas.clear()
        
        # Check that canvas is white
        pixel_data = self.canvas.get_pixel_data()
        assert np.all(pixel_data[0, 0] == [255, 255, 255])  # White pixel
    
    def test_draw_circle(self):
        """Test circle drawing."""
        self.canvas.draw_circle(200, 200, 50, "red", "blue")
        
        # Check that circle was drawn
        pixel_data = self.canvas.get_pixel_data()
        
        # Check center pixel (should be filled)
        center_pixel = pixel_data[200, 200]
        assert not np.array_equal(center_pixel, [255, 255, 255])  # Not white
    
    def test_draw_square(self):
        """Test square drawing."""
        self.canvas.draw_square(200, 200, 40, "green", "yellow")
        
        # Check that square was drawn
        pixel_data = self.canvas.get_pixel_data()
        
        # Check center pixel (should be filled)
        center_pixel = pixel_data[200, 200]
        assert not np.array_equal(center_pixel, [255, 255, 255])  # Not white
    
    def test_draw_polygon(self):
        """Test polygon drawing."""
        points = [(200, 150), (250, 200), (200, 250), (150, 200)]
        self.canvas.draw_polygon(points, "purple", "orange")
        
        # Check that polygon was drawn
        pixel_data = self.canvas.get_pixel_data()
        
        # Check center pixel (should be filled)
        center_pixel = pixel_data[200, 200]
        assert not np.array_equal(center_pixel, [255, 255, 255])  # Not white
    
    def test_draw_star(self):
        """Test star drawing."""
        self.canvas.draw_star(200, 200, 60, 30, 5, "red", "gold")
        
        # Check that star was drawn
        pixel_data = self.canvas.get_pixel_data()
        
        # Check center pixel (should be filled)
        center_pixel = pixel_data[200, 200]
        assert not np.array_equal(center_pixel, [255, 255, 255])  # Not white
    
    def test_draw_curve(self):
        """Test curve drawing."""
        points = [(100, 100), (150, 150), (200, 100), (250, 150)]
        self.canvas.draw_curve(points, "blue", 3)
        
        # Check that curve was drawn
        pixel_data = self.canvas.get_pixel_data()
        
        # Check that some pixels are not white
        non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
        assert np.any(non_white_pixels)
    
    def test_rotation_transformation(self):
        """Test rotation transformation."""
        original_points = [(100, 100), (200, 100), (200, 200), (100, 200)]
        center = (150, 150)
        angle = math.pi / 4  # 45 degrees
        
        rotated_points = self.canvas.apply_rotation(original_points, center, angle)
        
        # Check that points were rotated
        assert len(rotated_points) == len(original_points)
        assert rotated_points != original_points
        
        # Check that center point remains the same
        center_rotated = self.canvas.apply_rotation([center], center, angle)[0]
        assert abs(center_rotated[0] - center[0]) < 0.001
        assert abs(center_rotated[1] - center[1]) < 0.001
    
    def test_reflection_transformation(self):
        """Test reflection transformation."""
        original_points = [(100, 100), (200, 100), (200, 200), (100, 200)]
        line_start = (150, 0)
        line_end = (150, 300)  # Vertical line
        
        reflected_points = self.canvas.apply_reflection(original_points, line_start, line_end)
        
        # Check that points were reflected
        assert len(reflected_points) == len(original_points)
        assert reflected_points != original_points
    
    def test_create_tiling(self):
        """Test tiling pattern creation."""
        def draw_tile(x, y, **kwargs):
            self.canvas.draw_square(x, y, 30, "red", "blue")
        
        self.canvas.create_tiling(draw_tile, tile_size=50)
        
        # Check that tiles were created
        pixel_data = self.canvas.get_pixel_data()
        non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
        assert np.any(non_white_pixels)
    
    def test_create_mandala(self):
        """Test mandala creation."""
        def draw_element(x, y, **kwargs):
            self.canvas.draw_circle(x, y, 10, "purple", "pink")
        
        self.canvas.create_mandala(draw_element, layers=3, rotation_angle=math.pi/4)
        
        # Check that mandala was created
        pixel_data = self.canvas.get_pixel_data()
        non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
        assert np.any(non_white_pixels)
    
    def test_create_fractal(self):
        """Test fractal creation."""
        def draw_shape(x, y, size, **kwargs):
            self.canvas.draw_circle(x, y, size//2, "green", "lightgreen")
        
        self.canvas.create_fractal(draw_shape, depth=3, scale_factor=0.5)
        
        # Check that fractal was created
        pixel_data = self.canvas.get_pixel_data()
        non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
        assert np.any(non_white_pixels)
    
    def test_save_image(self):
        """Test image saving."""
        # Draw something
        self.canvas.draw_circle(200, 200, 50, "red", "blue")
        
        # Save image
        filename = os.path.join(self.temp_dir, "test_image.png")
        self.canvas.save(filename)
        
        # Check that file was created
        assert os.path.exists(filename)
        
        # Check that image can be loaded
        loaded_image = Image.open(filename)
        assert loaded_image.size == (400, 400)
    
    def test_get_image(self):
        """Test getting image copy."""
        # Draw something
        self.canvas.draw_circle(200, 200, 50, "red", "blue")
        
        # Get image copy
        image_copy = self.canvas.get_image()
        
        # Check that it's a PIL Image
        assert isinstance(image_copy, Image.Image)
        assert image_copy.size == (400, 400)
    
    def test_get_pixel_data(self):
        """Test getting pixel data."""
        pixel_data = self.canvas.get_pixel_data()
        
        # Check shape
        assert pixel_data.shape == (400, 400, 3)
        
        # Check data type
        assert pixel_data.dtype == np.uint8
        
        # Check that all pixels are white initially
        assert np.all(pixel_data == [255, 255, 255])


class TestColorPalette:
    """Test cases for ColorPalette class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.palette = ColorPalette()
    
    def test_palette_initialization(self):
        """Test color palette initialization."""
        assert "rainbow" in self.palette.palettes
        assert "pastel" in self.palette.palettes
        assert "monochrome" in self.palette.palettes
        assert "earth" in self.palette.palettes
        assert "ocean" in self.palette.palettes
    
    def test_get_palette(self):
        """Test getting specific palettes."""
        rainbow_palette = self.palette.get_palette("rainbow")
        assert isinstance(rainbow_palette, list)
        assert len(rainbow_palette) > 0
        assert all(color.startswith("#") for color in rainbow_palette)
        
        # Test unknown palette
        unknown_palette = self.palette.get_palette("unknown")
        assert unknown_palette == self.palette.palettes["rainbow"]  # Default
    
    def test_get_random_color(self):
        """Test getting random colors."""
        # Test with known palette
        color = self.palette.get_random_color("rainbow")
        assert color in self.palette.palettes["rainbow"]
        
        # Test with unknown palette
        color = self.palette.get_random_color("unknown")
        assert color in self.palette.palettes["rainbow"]  # Default
    
    def test_generate_harmonic_colors(self):
        """Test harmonic color generation."""
        base_color = "#FF0000"
        harmonic_colors = self.palette.generate_harmonic_colors(base_color, 5)
        
        assert len(harmonic_colors) == 5
        assert harmonic_colors[0] == base_color
        assert all(color.startswith("#") for color in harmonic_colors)


if __name__ == "__main__":
    pytest.main([__file__])
