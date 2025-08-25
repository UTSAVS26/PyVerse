#!/usr/bin/env python3
"""
Quick test script for ArtForgeAI
Runs basic functionality tests without requiring pytest
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def quick_test():
    """Run a quick test of the basic functionality"""
    print("🧪 ArtForgeAI Quick Test")
    print("=" * 30)
    
    try:
        # Test imports
        print("Testing imports...")
        import canvas
        import strokes
        import agent
        import train
        print("✓ All modules imported successfully")
        
        # Test canvas creation
        print("Testing canvas...")
        from canvas import Canvas
        c = Canvas(100, 100)
        assert c.get_image().shape == (100, 100, 3)
        print("✓ Canvas created successfully")
        
        # Test stroke generation
        print("Testing strokes...")
        from strokes import StrokeGenerator
        sg = StrokeGenerator(100, 100)
        stroke = sg.generate_line_stroke()
        assert 'type' in stroke
        print("✓ Stroke generation works")
        
        # Test environment
        print("Testing environment...")
        from agent import PaintingEnvironment
        env = PaintingEnvironment(100, 100, 10)
        state = env.reset()
        assert isinstance(state, type(c.get_image().flatten()))
        print("✓ Environment works")
        
        # Test agent
        print("Testing agent...")
        from agent import PaintingAgent
        agent = PaintingAgent(state.shape[0], 8)
        action = agent.select_action(state, add_noise=False)
        assert action.shape == (8,)
        print("✓ Agent works")
        
        # Test trainer
        print("Testing trainer...")
        from train import ArtForgeTrainer
        trainer = ArtForgeTrainer(100, 100, 5)
        stats = trainer.train_episode(render=False, save_artwork=False)
        assert 'total_reward' in stats
        print("✓ Trainer works")
        
        print("\n🎉 All quick tests passed! ArtForgeAI is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
