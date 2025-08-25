"""
Tests for the canvas module
"""

import pytest
import numpy as np
import tempfile
import os
from PIL import Image

# Import the modules to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from canvas import Canvas


class TestCanvas:
    """Test cases for the Canvas class"""
    
    def test_canvas_initialization(self):
        """Test canvas initialization with default parameters"""
        canvas = Canvas()
        assert canvas.width == 800
        assert canvas.height == 600
        assert canvas.background_color == (255, 255, 255)
        assert len(canvas.stroke_history) == 0
    
    def test_canvas_custom_initialization(self):
        """Test canvas initialization with custom parameters"""
        canvas = Canvas(width=400, height=300, background_color=(0, 0, 0))
        assert canvas.width == 400
        assert canvas.height == 300
        assert canvas.background_color == (0, 0, 0)
    
    def test_canvas_reset(self):
        """Test canvas reset functionality"""
        canvas = Canvas()
        
        # Apply a stroke
        stroke_data = {
            'type': 'line',
            'start_pos': (100, 100),
            'end_pos': (200, 200),
            'color': (255, 0, 0),
            'thickness': 2
        }
        canvas.apply_stroke(stroke_data)
        
        # Verify stroke was applied
        assert len(canvas.stroke_history) == 1
        
        # Reset canvas
        canvas.reset()
        
        # Verify reset
        assert len(canvas.stroke_history) == 0
        assert canvas.get_coverage() == 0.0
    
    def test_get_image(self):
        """Test getting canvas as numpy array"""
        canvas = Canvas(width=100, height=100)
        image = canvas.get_image()
        
        assert isinstance(image, np.ndarray)
        assert image.shape == (100, 100, 3)
        assert image.dtype == np.uint8
    
    def test_get_pil_image(self):
        """Test getting canvas as PIL Image"""
        canvas = Canvas(width=100, height=100)
        pil_image = canvas.get_pil_image()
        
        assert isinstance(pil_image, Image.Image)
        assert pil_image.size == (100, 100)
        assert pil_image.mode == 'RGB'
    
    def test_apply_line_stroke(self):
        """Test applying line stroke"""
        canvas = Canvas(width=200, height=200)
        
        stroke_data = {
            'type': 'line',
            'start_pos': (50, 50),
            'end_pos': (150, 150),
            'color': (255, 0, 0),
            'thickness': 3
        }
        
        success = canvas.apply_stroke(stroke_data)
        assert success == True
        assert len(canvas.stroke_history) == 1
        assert canvas.stroke_history[0]['type'] == 'line'
    
    def test_apply_curve_stroke(self):
        """Test applying curve stroke"""
        canvas = Canvas(width=200, height=200)
        
        stroke_data = {
            'type': 'curve',
            'start_pos': (50, 50),
            'end_pos': (150, 150),
            'control_points': [(100, 100)],
            'color': (0, 255, 0),
            'thickness': 2
        }
        
        success = canvas.apply_stroke(stroke_data)
        assert success == True
        assert len(canvas.stroke_history) == 1
        assert canvas.stroke_history[0]['type'] == 'curve'
    
    def test_apply_dot_stroke(self):
        """Test applying dot stroke"""
        canvas = Canvas(width=200, height=200)
        
        stroke_data = {
            'type': 'dot',
            'start_pos': (100, 100),
            'color': (0, 0, 255),
            'radius': 10,
            'thickness': 1
        }
        
        success = canvas.apply_stroke(stroke_data)
        assert success == True
        assert len(canvas.stroke_history) == 1
        assert canvas.stroke_history[0]['type'] == 'dot'
    
    def test_apply_splash_stroke(self):
        """Test applying splash stroke"""
        canvas = Canvas(width=200, height=200)
        
        stroke_data = {
            'type': 'splash',
            'start_pos': (100, 100),
            'color': (255, 255, 0),
            'radius': 20,
            'thickness': 1
        }
        
        success = canvas.apply_stroke(stroke_data)
        assert success == True
        assert len(canvas.stroke_history) == 1
        assert canvas.stroke_history[0]['type'] == 'splash'
    
    def test_apply_invalid_stroke(self):
        """Test applying invalid stroke type"""
        canvas = Canvas(width=200, height=200)
        
        stroke_data = {
            'type': 'invalid_type',
            'start_pos': (100, 100),
            'color': (255, 0, 0),
            'thickness': 2
        }
        
        success = canvas.apply_stroke(stroke_data)
        assert success == False
        assert len(canvas.stroke_history) == 0
    
    def test_stroke_outside_bounds(self):
        """Test applying stroke outside canvas bounds"""
        canvas = Canvas(width=100, height=100)
        
        stroke_data = {
            'type': 'line',
            'start_pos': (-10, -10),
            'end_pos': (200, 200),
            'color': (255, 0, 0),
            'thickness': 2
        }
        
        success = canvas.apply_stroke(stroke_data)
        assert success == True  # Should still succeed but clip to bounds
    
    def test_get_coverage(self):
        """Test coverage calculation"""
        canvas = Canvas(width=100, height=100)
        
        # Initially no coverage
        assert canvas.get_coverage() == 0.0
        
        # Apply a large dot
        stroke_data = {
            'type': 'dot',
            'start_pos': (50, 50),
            'color': (255, 0, 0),
            'radius': 30,
            'thickness': 1
        }
        canvas.apply_stroke(stroke_data)
        
        # Should have some coverage
        coverage = canvas.get_coverage()
        assert coverage > 0.0
        assert coverage <= 1.0
    
    def test_get_color_diversity(self):
        """Test color diversity calculation"""
        canvas = Canvas(width=100, height=100)
        
        # Initially no diversity (no strokes)
        assert canvas.get_color_diversity() == 0.0
        
        # Apply strokes with different colors
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
        for i, color in enumerate(colors):
            stroke_data = {
                'type': 'dot',
                'start_pos': (30 + i*20, 50),
                'color': color,
                'radius': 5,
                'thickness': 1
            }
            canvas.apply_stroke(stroke_data)
        
        # Should have maximum diversity (all colors different)
        diversity = canvas.get_color_diversity()
        assert diversity == 1.0
    
    def test_get_color_diversity_duplicate_colors(self):
        """Test color diversity with duplicate colors"""
        canvas = Canvas(width=100, height=100)
        
        # Apply strokes with same color
        for i in range(3):
            stroke_data = {
                'type': 'dot',
                'start_pos': (30 + i*20, 50),
                'color': (255, 0, 0),  # Same color
                'radius': 5,
                'thickness': 1
            }
            canvas.apply_stroke(stroke_data)
        
        # Should have minimum diversity (all colors same)
        diversity = canvas.get_color_diversity()
        assert diversity == 1.0 / 3.0  # 1 unique color / 3 total strokes
    
    def test_save_image(self):
        """Test saving canvas to file"""
        canvas = Canvas(width=100, height=100)
        
        # Apply a stroke
        stroke_data = {
            'type': 'line',
            'start_pos': (25, 25),
            'end_pos': (75, 75),
            'color': (255, 0, 0),
            'thickness': 2
        }
        canvas.apply_stroke(stroke_data)
        
        # Save to temporary file
        tmp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp_file.close()  # Close the file handle
        
        try:
            canvas.save_image(tmp_file.name)
            
            # Verify file was created and has content
            assert os.path.exists(tmp_file.name)
            assert os.path.getsize(tmp_file.name) > 0
        finally:
            # Clean up
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass  # File might already be deleted
    
    def test_get_stats(self):
        """Test getting canvas statistics"""
        canvas = Canvas(width=200, height=150)
        
        # Apply some strokes
        stroke_data = {
            'type': 'line',
            'start_pos': (50, 50),
            'end_pos': (150, 100),
            'color': (255, 0, 0),
            'thickness': 2
        }
        canvas.apply_stroke(stroke_data)
        
        stats = canvas.get_stats()
        
        assert stats['width'] == 200
        assert stats['height'] == 150
        assert stats['num_strokes'] == 1
        assert 'coverage' in stats
        assert 'color_diversity' in stats
        assert 'stroke_types' in stats
        assert stats['stroke_types'] == ['line']
    
    def test_multiple_strokes(self):
        """Test applying multiple strokes"""
        canvas = Canvas(width=200, height=200)
        
        strokes = [
            {
                'type': 'line',
                'start_pos': (50, 50),
                'end_pos': (150, 50),
                'color': (255, 0, 0),
                'thickness': 2
            },
            {
                'type': 'dot',
                'start_pos': (100, 100),
                'color': (0, 255, 0),
                'radius': 15,
                'thickness': 1
            },
            {
                'type': 'curve',
                'start_pos': (25, 150),
                'end_pos': (175, 150),
                'control_points': [(100, 125)],
                'color': (0, 0, 255),
                'thickness': 3
            }
        ]
        
        for stroke in strokes:
            success = canvas.apply_stroke(stroke)
            assert success == True
        
        assert len(canvas.stroke_history) == 3
        assert canvas.get_coverage() > 0.0
        assert canvas.get_color_diversity() == 1.0  # All different colors
    
    def test_stroke_history_consistency(self):
        """Test that stroke history is consistent"""
        canvas = Canvas(width=100, height=100)
        
        original_stroke = {
            'type': 'line',
            'start_pos': (25, 25),
            'end_pos': (75, 75),
            'color': (255, 0, 0),
            'thickness': 2,
            'angle': 45
        }
        
        canvas.apply_stroke(original_stroke)
        
        # Check that stroke history contains the exact same data
        stored_stroke = canvas.stroke_history[0]
        assert stored_stroke['type'] == original_stroke['type']
        assert stored_stroke['start_pos'] == original_stroke['start_pos']
        assert stored_stroke['end_pos'] == original_stroke['end_pos']
        assert stored_stroke['color'] == original_stroke['color']
        assert stored_stroke['thickness'] == original_stroke['thickness']
        assert stored_stroke['angle'] == original_stroke['angle']


if __name__ == "__main__":
    pytest.main([__file__])
