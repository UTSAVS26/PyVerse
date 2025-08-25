#!/usr/bin/env python3
"""
Basic functionality test for ArtForgeAI
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        import canvas
        print("✓ canvas module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import canvas: {e}")
        return False
    
    try:
        import strokes
        print("✓ strokes module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import strokes: {e}")
        return False
    
    try:
        import agent
        print("✓ agent module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import agent: {e}")
        return False
    
    try:
        import train
        print("✓ train module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import train: {e}")
        return False
    
    return True

def test_canvas_basic():
    """Test basic canvas functionality"""
    print("\nTesting canvas basic functionality...")
    
    try:
        from canvas import Canvas
        
        # Create canvas
        canvas = Canvas(width=100, height=100)
        print("✓ Canvas created successfully")
        
        # Test basic operations
        img = canvas.get_image()
        assert img.shape == (100, 100, 3), f"Expected shape (100, 100, 3), got {img.shape}"
        print("✓ Canvas image shape correct")
        
        # Test stroke application
        stroke_data = {
            'type': 'line',
            'start_pos': (10, 10),
            'end_pos': (90, 90),
            'color': (255, 0, 0),
            'thickness': 2
        }
        success = canvas.apply_stroke(stroke_data)
        assert success, "Stroke application failed"
        print("✓ Stroke applied successfully")
        
        # Test coverage calculation
        coverage = canvas.get_coverage()
        assert 0 <= coverage <= 1, f"Coverage should be between 0 and 1, got {coverage}"
        print("✓ Coverage calculation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Canvas test failed: {e}")
        return False

def test_strokes_basic():
    """Test basic strokes functionality"""
    print("\nTesting strokes basic functionality...")
    
    try:
        from strokes import StrokeGenerator
        
        # Create generator
        generator = StrokeGenerator(100, 100)
        print("✓ StrokeGenerator created successfully")
        
        # Test color generation
        color = generator.generate_random_color()
        assert len(color) == 3, f"Color should have 3 components, got {len(color)}"
        assert all(0 <= c <= 255 for c in color), f"Color values should be 0-255, got {color}"
        print("✓ Color generation works")
        
        # Test stroke generation
        stroke = generator.generate_line_stroke()
        assert 'type' in stroke, "Stroke should have 'type' field"
        assert 'start_pos' in stroke, "Stroke should have 'start_pos' field"
        assert 'end_pos' in stroke, "Stroke should have 'end_pos' field"
        print("✓ Stroke generation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Strokes test failed: {e}")
        return False

def test_agent_basic():
    """Test basic agent functionality"""
    print("\nTesting agent basic functionality...")
    
    try:
        from agent import PaintingEnvironment, PaintingAgent
        
        # Create environment
        env = PaintingEnvironment(canvas_width=100, canvas_height=100, max_strokes=10)
        print("✓ PaintingEnvironment created successfully")
        
        # Test environment reset
        state = env.reset()
        assert isinstance(state, np.ndarray), f"State should be numpy array, got {type(state)}"
        print("✓ Environment reset works")
        
        # Test action selection
        agent = PaintingAgent(state_dim=state.shape[0], action_dim=8)
        print("✓ PaintingAgent created successfully")
        
        action = agent.select_action(state, add_noise=False)
        assert isinstance(action, np.ndarray), f"Action should be numpy array, got {type(action)}"
        assert action.shape == (8,), f"Action should have shape (8,), got {action.shape}"
        print("✓ Action selection works")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        return False

def test_trainer_basic():
    """Test basic trainer functionality"""
    print("\nTesting trainer basic functionality...")
    
    try:
        from train import ArtForgeTrainer
        
        # Create trainer
        trainer = ArtForgeTrainer(canvas_width=100, canvas_height=100, max_strokes=5)
        print("✓ ArtForgeTrainer created successfully")
        
        # Test episode training
        episode_stats = trainer.train_episode(render=False, save_artwork=False)
        assert isinstance(episode_stats, dict), f"Episode stats should be dict, got {type(episode_stats)}"
        assert 'total_reward' in episode_stats, "Episode stats should contain 'total_reward'"
        print("✓ Episode training works")
        
        return True
        
    except Exception as e:
        print(f"✗ Trainer test failed: {e}")
        return False

def main():
    """Run all basic tests"""
    print("ArtForgeAI Basic Functionality Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_canvas_basic,
        test_strokes_basic,
        test_agent_basic,
        test_trainer_basic
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! ArtForgeAI is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
