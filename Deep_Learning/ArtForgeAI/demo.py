#!/usr/bin/env python3
"""
ArtForgeAI Demonstration Script
Shows the AI agent learning to paint abstract art
"""

import sys
import os
import numpy as np
from PIL import Image

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demo_basic_painting():
    """Demonstrate basic painting functionality"""
    print("🎨 ArtForgeAI Basic Painting Demo")
    print("=" * 50)
    
    try:
        from canvas import Canvas
        from strokes import StrokeGenerator
        
        # Create canvas and stroke generator
        canvas = Canvas(width=400, height=300, background_color=(240, 240, 240))
        generator = StrokeGenerator(400, 300)
        
        print("✓ Created canvas and stroke generator")
        
        # Generate and apply some random strokes
        strokes = generator.generate_stroke_sequence(num_strokes=20, palette='vibrant')
        
        for i, stroke in enumerate(strokes):
            success = canvas.apply_stroke(stroke)
            if success:
                print(f"✓ Applied stroke {i+1}/20: {stroke['type']}")
            else:
                print(f"✗ Failed to apply stroke {i+1}/20")
        
        # Save the artwork
        canvas.save_image("gallery/artworks/demo_basic_painting.png")
        print("✓ Saved artwork to gallery/artworks/demo_basic_painting.png")
        
        # Show statistics
        stats = canvas.get_stats()
        print(f"✓ Final coverage: {stats['coverage']:.2%}")
        print(f"✓ Color diversity: {stats['color_diversity']:.2f}")
        print(f"✓ Total strokes: {stats['stroke_count']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic painting demo failed: {e}")
        return False

def demo_agent_training():
    """Demonstrate agent training"""
    print("\n🤖 ArtForgeAI Agent Training Demo")
    print("=" * 50)
    
    try:
        from train import ArtForgeTrainer
        
        # Create trainer with smaller canvas for faster demo
        trainer = ArtForgeTrainer(
            canvas_width=200,
            canvas_height=150,
            max_strokes=15,
            save_dir="gallery"
        )
        
        print("✓ Created ArtForgeAI trainer")
        
        # Train a few episodes
        print("Training agent for 5 episodes...")
        for episode in range(5):
            stats = trainer.train_episode(render=False, save_artwork=False)
            print(f"Episode {episode+1}: Reward={stats['total_reward']:.2f}, "
                  f"Coverage={stats['coverage']:.2%}, "
                  f"Strokes={stats['stroke_count']}")
        
        # Generate final artwork
        print("\nGenerating final artwork with trained agent...")
        final_artwork = trainer.generate_artwork(num_strokes=20, render=False)
        
        # Save the final artwork
        final_image = Image.fromarray(final_artwork)
        final_image.save("gallery/artworks/demo_trained_agent.png")
        print("✓ Saved trained agent artwork to gallery/artworks/demo_trained_agent.png")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent training demo failed: {e}")
        return False

def demo_stroke_types():
    """Demonstrate different stroke types"""
    print("\n🖌️ ArtForgeAI Stroke Types Demo")
    print("=" * 50)
    
    try:
        from canvas import Canvas
        from strokes import StrokeGenerator
        
        # Create canvas
        canvas = Canvas(width=300, height=200, background_color=(255, 255, 255))
        generator = StrokeGenerator(300, 200)
        
        # Demonstrate each stroke type
        stroke_types = ['line', 'curve', 'dot', 'splash']
        
        for i, stroke_type in enumerate(stroke_types):
            # Generate stroke of specific type
            if stroke_type == 'line':
                stroke = generator.generate_line_stroke(
                    start_pos=(50 + i*60, 50),
                    end_pos=(50 + i*60, 150),
                    color=(255, 0, 0) if i == 0 else (0, 255, 0) if i == 1 else (0, 0, 255) if i == 2 else (255, 255, 0),
                    thickness=3
                )
            elif stroke_type == 'curve':
                stroke = generator.generate_curve_stroke(
                    start_pos=(50 + i*60, 100),
                    control_pos=(50 + i*60, 50),
                    end_pos=(50 + i*60, 150),
                    color=(255, 0, 0) if i == 0 else (0, 255, 0) if i == 1 else (0, 0, 255) if i == 2 else (255, 255, 0),
                    thickness=3
                )
            elif stroke_type == 'dot':
                stroke = generator.generate_dot_stroke(
                    center=(50 + i*60, 100),
                    color=(255, 0, 0) if i == 0 else (0, 255, 0) if i == 1 else (0, 0, 255) if i == 2 else (255, 255, 0),
                    radius=15
                )
            else:  # splash
                stroke = generator.generate_splash_stroke(
                    center=(50 + i*60, 100),
                    color=(255, 0, 0) if i == 0 else (0, 255, 0) if i == 1 else (0, 0, 255) if i == 2 else (255, 255, 0),
                    radius=20
                )
            
            # Apply stroke
            success = canvas.apply_stroke(stroke)
            if success:
                print(f"✓ Applied {stroke_type} stroke")
            else:
                print(f"✗ Failed to apply {stroke_type} stroke")
        
        # Save the stroke types demo
        canvas.save_image("gallery/artworks/demo_stroke_types.png")
        print("✓ Saved stroke types demo to gallery/artworks/demo_stroke_types.png")
        
        return True
        
    except Exception as e:
        print(f"✗ Stroke types demo failed: {e}")
        return False

def main():
    """Run all demonstrations"""
    print("🎨 ArtForgeAI - Procedural Brushstroke Painter with Reinforcement Learning")
    print("=" * 70)
    print("This demo showcases the AI agent learning to paint abstract art!")
    print()
    
    # Ensure gallery directories exist
    os.makedirs("gallery/artworks", exist_ok=True)
    os.makedirs("gallery/models", exist_ok=True)
    
    demos = [
        ("Basic Painting", demo_basic_painting),
        ("Stroke Types", demo_stroke_types),
        ("Agent Training", demo_agent_training)
    ]
    
    successful_demos = 0
    total_demos = len(demos)
    
    for demo_name, demo_func in demos:
        print(f"\n{'='*20} {demo_name} {'='*20}")
        if demo_func():
            successful_demos += 1
        print()
    
    print("=" * 70)
    print(f"Demo Results: {successful_demos}/{total_demos} demos completed successfully")
    
    if successful_demos == total_demos:
        print("🎉 All demonstrations completed successfully!")
        print("Check the gallery/artworks/ directory for generated artworks.")
        print("\nGenerated files:")
        print("- demo_basic_painting.png: Random stroke painting")
        print("- demo_stroke_types.png: Different stroke types showcase")
        print("- demo_trained_agent.png: AI agent's learned artwork")
    else:
        print("❌ Some demonstrations failed. Check the errors above.")
    
    print("\nThank you for trying ArtForgeAI! 🎨")

if __name__ == "__main__":
    main()
