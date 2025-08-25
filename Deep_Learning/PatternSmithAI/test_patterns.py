"""
Tests for PatternSmithAI Patterns Module
Tests all pattern generators and factory functionality.
"""

import pytest
import numpy as np
import os
import tempfile
from PIL import Image
import math

from canvas import PatternCanvas, ColorPalette
from patterns import (
    PatternGenerator, GeometricPatternGenerator, MandalaGenerator,
    FractalGenerator, TilingGenerator, PatternFactory
)


class TestPatternGenerator:
    """Test cases for base PatternGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.canvas = PatternCanvas(400, 400, "white")
        self.color_palette = ColorPalette()
        self.generator = PatternGenerator(self.canvas, self.color_palette)
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        assert self.generator.canvas == self.canvas
        assert self.generator.color_palette == self.color_palette
        assert hasattr(self.generator, 'rng')
    
    def test_set_seed(self):
        """Test seed setting for reproducibility."""
        self.generator.set_seed(42)
        # Should not raise an exception
        assert True
    
    def test_generate_not_implemented(self):
        """Test that base generate method raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            self.generator.generate()


class TestGeometricPatternGenerator:
    """Test cases for GeometricPatternGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.canvas = PatternCanvas(400, 400, "white")
        self.color_palette = ColorPalette()
        self.generator = GeometricPatternGenerator(self.canvas, self.color_palette)
    
    def test_generator_initialization(self):
        """Test geometric generator initialization."""
        assert isinstance(self.generator, GeometricPatternGenerator)
        assert isinstance(self.generator, PatternGenerator)
    
    def test_generate_circles(self):
        """Test circle pattern generation."""
        result = self.generator.generate(pattern_type="circles", count=5)
        
        # Check that canvas was returned
        assert result == self.canvas
        
        # Check that circles were drawn
        pixel_data = self.canvas.get_pixel_data()
        non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
        assert np.any(non_white_pixels)
    
    def test_generate_squares(self):
        """Test square pattern generation."""
        result = self.generator.generate(pattern_type="squares", count=5)
        
        # Check that canvas was returned
        assert result == self.canvas
        
        # Check that squares were drawn
        pixel_data = self.canvas.get_pixel_data()
        non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
        assert np.any(non_white_pixels)
    
    def test_generate_polygons(self):
        """Test polygon pattern generation."""
        result = self.generator.generate(pattern_type="polygons", count=5)
        
        # Check that canvas was returned
        assert result == self.canvas
        
        # Check that polygons were drawn
        pixel_data = self.canvas.get_pixel_data()
        non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
        assert np.any(non_white_pixels)
    
    def test_generate_stars(self):
        """Test star pattern generation."""
        result = self.generator.generate(pattern_type="stars", count=5)
        
        # Check that canvas was returned
        assert result == self.canvas
        
        # Check that stars were drawn
        pixel_data = self.canvas.get_pixel_data()
        non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
        assert np.any(non_white_pixels)
    
    def test_generate_random(self):
        """Test random pattern generation."""
        result = self.generator.generate(pattern_type="random")
        
        # Check that canvas was returned
        assert result == self.canvas
        
        # Check that something was drawn
        pixel_data = self.canvas.get_pixel_data()
        non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
        assert np.any(non_white_pixels)
    
    def test_generate_with_parameters(self):
        """Test generation with custom parameters."""
        result = self.generator.generate(
            pattern_type="circles",
            count=10,
            min_radius=20,
            max_radius=40,
            palette="rainbow"
        )
        
        # Check that canvas was returned
        assert result == self.canvas
        
        # Check that circles were drawn
        pixel_data = self.canvas.get_pixel_data()
        non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
        assert np.any(non_white_pixels)


class TestMandalaGenerator:
    """Test cases for MandalaGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.canvas = PatternCanvas(400, 400, "white")
        self.color_palette = ColorPalette()
        self.generator = MandalaGenerator(self.canvas, self.color_palette)
    
    def test_generator_initialization(self):
        """Test mandala generator initialization."""
        assert isinstance(self.generator, MandalaGenerator)
        assert isinstance(self.generator, PatternGenerator)
    
    def test_generate_mandala_circles(self):
        """Test mandala generation with circles."""
        result = self.generator.generate(
            layers=5,
            elements_per_layer=8,
            base_shape="circle"
        )
        
        # Check that canvas was returned
        assert result == self.canvas
        
        # Check that mandala was drawn
        pixel_data = self.canvas.get_pixel_data()
        non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
        assert np.any(non_white_pixels)
    
    def test_generate_mandala_squares(self):
        """Test mandala generation with squares."""
        result = self.generator.generate(
            layers=4,
            elements_per_layer=6,
            base_shape="square"
        )
        
        # Check that canvas was returned
        assert result == self.canvas
        
        # Check that mandala was drawn
        pixel_data = self.canvas.get_pixel_data()
        non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
        assert np.any(non_white_pixels)
    
    def test_generate_mandala_stars(self):
        """Test mandala generation with stars."""
        result = self.generator.generate(
            layers=3,
            elements_per_layer=10,
            base_shape="star"
        )
        
        # Check that canvas was returned
        assert result == self.canvas
        
        # Check that mandala was drawn
        pixel_data = self.canvas.get_pixel_data()
        non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
        assert np.any(non_white_pixels)


class TestFractalGenerator:
    """Test cases for FractalGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.canvas = PatternCanvas(400, 400, "white")
        self.color_palette = ColorPalette()
        self.generator = FractalGenerator(self.canvas, self.color_palette)
    
    def test_generator_initialization(self):
        """Test fractal generator initialization."""
        assert isinstance(self.generator, FractalGenerator)
        assert isinstance(self.generator, PatternGenerator)
    
    def test_generate_sierpinski(self):
        """Test Sierpinski triangle generation."""
        result = self.generator.generate(fractal_type="sierpinski", depth=3)
        
        # Check that canvas was returned
        assert result == self.canvas
        
        # Check that fractal was drawn
        pixel_data = self.canvas.get_pixel_data()
        non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
        assert np.any(non_white_pixels)
    
    def test_generate_koch(self):
        """Test Koch snowflake generation."""
        result = self.generator.generate(fractal_type="koch", depth=3)
        
        # Check that canvas was returned
        assert result == self.canvas
        
        # Check that fractal was drawn
        pixel_data = self.canvas.get_pixel_data()
        non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
        assert np.any(non_white_pixels)
    
    def test_generate_tree(self):
        """Test fractal tree generation."""
        result = self.generator.generate(fractal_type="tree", depth=4)
        
        # Check that canvas was returned
        assert result == self.canvas
        
        # Check that fractal was drawn
        pixel_data = self.canvas.get_pixel_data()
        non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
        assert np.any(non_white_pixels)
    
    def test_generate_with_different_depths(self):
        """Test fractal generation with different depths."""
        for depth in [2, 3, 4]:
            self.canvas.clear()
            result = self.generator.generate(fractal_type="sierpinski", depth=depth)
            
            # Check that canvas was returned
            assert result == self.canvas
            
            # Check that fractal was drawn
            pixel_data = self.canvas.get_pixel_data()
            non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
            assert np.any(non_white_pixels)


class TestTilingGenerator:
    """Test cases for TilingGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.canvas = PatternCanvas(400, 400, "white")
        self.color_palette = ColorPalette()
        self.generator = TilingGenerator(self.canvas, self.color_palette)
    
    def test_generator_initialization(self):
        """Test tiling generator initialization."""
        assert isinstance(self.generator, TilingGenerator)
        assert isinstance(self.generator, PatternGenerator)
    
    def test_generate_hexagonal_tiling(self):
        """Test hexagonal tiling generation."""
        result = self.generator.generate(tile_type="hexagonal", tile_size=40)
        
        # Check that canvas was returned
        assert result == self.canvas
        
        # Check that tiling was drawn
        pixel_data = self.canvas.get_pixel_data()
        non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
        assert np.any(non_white_pixels)
    
    def test_generate_square_tiling(self):
        """Test square tiling generation."""
        result = self.generator.generate(tile_type="square", tile_size=50)
        
        # Check that canvas was returned
        assert result == self.canvas
        
        # Check that tiling was drawn
        pixel_data = self.canvas.get_pixel_data()
        non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
        assert np.any(non_white_pixels)
    
    def test_generate_triangular_tiling(self):
        """Test triangular tiling generation."""
        result = self.generator.generate(tile_type="triangular", tile_size=60)
        
        # Check that canvas was returned
        assert result == self.canvas
        
        # Check that tiling was drawn
        pixel_data = self.canvas.get_pixel_data()
        non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
        assert np.any(non_white_pixels)
    
    def test_generate_with_different_tile_sizes(self):
        """Test tiling generation with different tile sizes."""
        for tile_size in [30, 50, 70]:
            self.canvas.clear()
            result = self.generator.generate(tile_type="square", tile_size=tile_size)
            
            # Check that canvas was returned
            assert result == self.canvas
            
            # Check that tiling was drawn
            pixel_data = self.canvas.get_pixel_data()
            non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
            assert np.any(non_white_pixels)


class TestPatternFactory:
    """Test cases for PatternFactory class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.canvas = PatternCanvas(400, 400, "white")
        self.color_palette = ColorPalette()
    
    def test_create_geometric_generator(self):
        """Test creating geometric pattern generator."""
        generator = PatternFactory.create_generator("geometric", self.canvas, self.color_palette)
        assert isinstance(generator, GeometricPatternGenerator)
    
    def test_create_mandala_generator(self):
        """Test creating mandala generator."""
        generator = PatternFactory.create_generator("mandala", self.canvas, self.color_palette)
        assert isinstance(generator, MandalaGenerator)
    
    def test_create_fractal_generator(self):
        """Test creating fractal generator."""
        generator = PatternFactory.create_generator("fractal", self.canvas, self.color_palette)
        assert isinstance(generator, FractalGenerator)
    
    def test_create_tiling_generator(self):
        """Test creating tiling generator."""
        generator = PatternFactory.create_generator("tiling", self.canvas, self.color_palette)
        assert isinstance(generator, TilingGenerator)
    
    def test_create_unknown_generator(self):
        """Test creating unknown generator type."""
        with pytest.raises(ValueError, match="Unknown pattern type"):
            PatternFactory.create_generator("unknown", self.canvas, self.color_palette)
    
    def test_all_generators_work(self):
        """Test that all generators can create patterns."""
        generator_types = ["geometric", "mandala", "fractal", "tiling"]
        
        for gen_type in generator_types:
            # Create generator
            generator = PatternFactory.create_generator(gen_type, self.canvas, self.color_palette)
            
            # Generate pattern
            result = generator.generate()
            
            # Check that canvas was returned
            assert result == self.canvas
            
            # Check that something was drawn
            pixel_data = self.canvas.get_pixel_data()
            non_white_pixels = np.any(pixel_data != [255, 255, 255], axis=2)
            assert np.any(non_white_pixels)
            
            # Clear canvas for next test
            self.canvas.clear()


if __name__ == "__main__":
    pytest.main([__file__])
