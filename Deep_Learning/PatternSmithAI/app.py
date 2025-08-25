"""
PatternSmithAI - Streamlit Web Application
Interactive web interface for pattern generation and gallery viewing.
"""

import streamlit as st
import os
import json
import numpy as np
from datetime import datetime
from PIL import Image
import io

from canvas import PatternCanvas, ColorPalette
from patterns import PatternFactory
from agent import PatternAgent, PatternEvaluator
from train import PatternTrainer


def main():
    st.set_page_config(
        page_title="PatternSmithAI",
        page_icon="🧩",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧩 PatternSmithAI")
    st.markdown("### Infinite Procedural Pattern & Mandala Designer")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Canvas settings
    st.sidebar.subheader("Canvas Settings")
    canvas_size = st.sidebar.slider("Canvas Size", 400, 1200, 800, 50)
    
    # Pattern type selection
    st.sidebar.subheader("Pattern Type")
    pattern_type = st.sidebar.selectbox(
        "Select Pattern Type",
        ["geometric", "mandala", "fractal", "tiling", "ai_generated"]
    )
    
    # Color palette selection
    st.sidebar.subheader("Color Palette")
    color_palette_name = st.sidebar.selectbox(
        "Select Color Palette",
        ["rainbow", "pastel", "monochrome", "earth", "ocean"]
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Pattern Generator")
        
        # Pattern parameters based on type
        if pattern_type == "geometric":
            st.write("**Geometric Pattern Parameters**")
            geometric_type = st.selectbox("Shape Type", ["random", "circles", "squares", "polygons", "stars"])
            count = st.slider("Number of Shapes", 5, 50, 20)
            
        elif pattern_type == "mandala":
            st.write("**Mandala Pattern Parameters**")
            base_shape = st.selectbox("Base Shape", ["circle", "square", "star"])
            layers = st.slider("Number of Layers", 3, 15, 8)
            elements_per_layer = st.slider("Elements per Layer", 6, 20, 12)
            
        elif pattern_type == "fractal":
            st.write("**Fractal Pattern Parameters**")
            fractal_type = st.selectbox("Fractal Type", ["sierpinski", "koch", "tree"])
            depth = st.slider("Fractal Depth", 3, 7, 4)
            
        elif pattern_type == "tiling":
            st.write("**Tiling Pattern Parameters**")
            tile_type = st.selectbox("Tile Type", ["hexagonal", "square", "triangular"])
            tile_size = st.slider("Tile Size", 20, 100, 50)
            
        elif pattern_type == "ai_generated":
            st.write("**AI-Generated Pattern Parameters**")
            steps = st.slider("Generation Steps", 5, 30, 15)
            use_trained_model = st.checkbox("Use Trained Model (if available)")
    
    with col2:
        st.subheader("Actions")
        
        if st.button("🎨 Generate Pattern", type="primary"):
            with st.spinner("Generating pattern..."):
                # Create canvas and color palette
                canvas = PatternCanvas(canvas_size, canvas_size)
                color_palette = ColorPalette()
                
                # Generate pattern based on type
                if pattern_type == "ai_generated":
                    agent = PatternAgent(canvas, color_palette)
                    
                    # Try to load trained model
                    if use_trained_model and os.path.exists("gallery/trained_model.pth"):
                        try:
                            agent.load_model("gallery/trained_model.pth")
                            st.success("Loaded trained model!")
                        except:
                            st.warning("Could not load trained model, using untrained agent")
                    
                    canvas = agent.generate_pattern(steps)
                    
                else:
                    # Create generator
                    generator = PatternFactory.create_generator(pattern_type, canvas, color_palette)
                    
                    # Set parameters
                    if pattern_type == "geometric":
                        generator.generate(pattern_type=geometric_type, count=count, palette=color_palette_name)
                    elif pattern_type == "mandala":
                        generator.generate(base_shape=base_shape, layers=layers, 
                                         elements_per_layer=elements_per_layer)
                    elif pattern_type == "fractal":
                        generator.generate(fractal_type=fractal_type, depth=depth)
                    elif pattern_type == "tiling":
                        generator.generate(tile_type=tile_type, tile_size=tile_size)
                
                # Display pattern
                st.image(canvas.get_image(), caption="Generated Pattern", use_column_width=True)
                
                # Save pattern
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"gallery/pattern_{pattern_type}_{timestamp}.png"
                canvas.save(filename)
                
                st.success(f"Pattern saved as {filename}")
        
        if st.button("📊 Evaluate Pattern"):
            if 'canvas' in locals():
                evaluator = PatternEvaluator()
                scores = evaluator.evaluate_pattern(canvas)
                overall_score = evaluator.get_overall_score(scores)
                
                st.write("**Pattern Evaluation**")
                st.write(f"Overall Score: {overall_score:.3f}")
                
                for criterion, score in scores.items():
                    st.write(f"{criterion.title()}: {score:.3f}")
        
        if st.button("🎯 Train AI Agent"):
            st.info("Training AI agent... This may take a while.")
            trainer = PatternTrainer(canvas_size, "gallery")
            trainer.train(episodes=50, steps_per_episode=10)
            st.success("Training completed!")
    
    # Gallery section
    st.subheader("🎨 Pattern Gallery")
    
    # Show existing patterns
    if os.path.exists("gallery"):
        pattern_files = [f for f in os.listdir("gallery") if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if pattern_files:
            # Create columns for gallery
            cols = st.columns(3)
            
            for i, filename in enumerate(pattern_files[:9]):  # Show first 9 patterns
                col_idx = i % 3
                with cols[col_idx]:
                    filepath = os.path.join("gallery", filename)
                    try:
                        image = Image.open(filepath)
                        st.image(image, caption=filename, use_column_width=True)
                        
                        # Add download button
                        with open(filepath, "rb") as file:
                            btn = st.download_button(
                                label=f"Download {filename}",
                                data=file.read(),
                                file_name=filename,
                                mime="image/png"
                            )
                    except Exception as e:
                        st.error(f"Error loading {filename}: {e}")
        else:
            st.info("No patterns found in gallery. Generate some patterns first!")
    
    # Training section
    st.subheader("🤖 AI Training")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("**Training Parameters**")
        episodes = st.slider("Training Episodes", 10, 200, 50)
        steps_per_episode = st.slider("Steps per Episode", 5, 30, 15)
        
        if st.button("🚀 Start Training"):
            with st.spinner("Training AI agent..."):
                trainer = PatternTrainer(canvas_size, "gallery")
                trainer.train(episodes=episodes, steps_per_episode=steps_per_episode)
                st.success("Training completed!")
    
    with col4:
        st.write("**Training Statistics**")
        if os.path.exists("gallery/training_stats.json"):
            try:
                with open("gallery/training_stats.json", "r") as f:
                    stats = json.load(f)
                
                if stats['episodes']:
                    st.write(f"Episodes trained: {len(stats['episodes'])}")
                    st.write(f"Average reward: {np.mean(stats['rewards']):.3f}")
                    st.write(f"Average score: {np.mean(stats['scores']):.3f}")
                    st.write(f"Current epsilon: {stats['epsilon_values'][-1]:.3f}")
            except:
                st.info("No training statistics available")
        else:
            st.info("No training data found")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        **PatternSmithAI** - An AI-powered procedural pattern generator that creates unique geometric designs, 
        mandalas, and fractals using reinforcement learning and mathematical symmetry rules.
        """
    )


if __name__ == "__main__":
    main()
