# 🧩 PatternSmithAI

**Infinite Procedural Pattern & Mandala Designer**

PatternSmithAI is an AI system that procedurally generates geometric patterns, wallpapers, and mandalas using mathematical symmetry rules and reinforcement learning. Instead of relying on existing datasets, the system uses mathematical algorithms and feedback loops to guide the generation process, creating unique and visually appealing patterns.

## 🌟 Features

### Core Functionality
- **Procedural Pattern Generation**: Create geometric patterns, mandalas, fractals, and tiling designs
- **AI-Powered Learning**: Reinforcement learning agent that improves pattern quality over time
- **Mathematical Symmetry**: Built-in symmetry operations, rotations, and transformations
- **Color Harmony**: Multiple color palettes and harmonic color generation
- **Interactive Training**: User feedback integration for personalized pattern styles

### Pattern Types
- **Geometric Patterns**: Circles, squares, polygons, stars with customizable parameters
- **Mandala Designs**: Multi-layered circular patterns with various base shapes
- **Fractal Art**: Sierpinski triangles, Koch snowflakes, and fractal trees
- **Tiling Patterns**: Hexagonal, square, and triangular tessellations
- **AI-Generated**: Machine learning-driven pattern creation

### Technical Features
- **Canvas Engine**: High-performance drawing with PIL/Pillow
- **Neural Network**: PyTorch-based reinforcement learning agent
- **Pattern Evaluation**: Automated quality assessment (symmetry, complexity, balance)
- **Export Options**: PNG/SVG export for wallpapers and design assets
- **Web Interface**: Streamlit-based interactive application

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/PatternSmithAI.git
cd PatternSmithAI
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the web application**:
```bash
streamlit run app.py
```

### Basic Usage

#### Command Line Training
```bash
# Train the AI agent
python train.py --episodes 100 --steps 15

# Interactive training with user feedback
python train.py --interactive --episodes 20 --steps 10
```

#### Python API
```python
from canvas import PatternCanvas, ColorPalette
from patterns import PatternFactory

# Create a canvas and generate patterns
canvas = PatternCanvas(800, 800)
color_palette = ColorPalette()

# Generate a mandala
generator = PatternFactory.create_generator("mandala", canvas, color_palette)
generator.generate(layers=8, elements_per_layer=12, base_shape="circle")

# Save the pattern
canvas.save("my_mandala.png")
```

## 📁 Project Structure

```
PatternSmithAI/
├── canvas.py              # Drawing engine and color palettes
├── patterns.py            # Procedural pattern generators
├── agent.py               # AI agent and neural network
├── train.py               # Training loops and feedback
├── app.py                 # Streamlit web interface
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── gallery/              # Generated patterns and outputs
├── test_canvas.py        # Canvas module tests
├── test_patterns.py      # Pattern generators tests
├── test_agent.py         # AI agent tests
└── test_train.py         # Training module tests
```

## 🎨 Pattern Generation Examples

### Geometric Patterns
```python
from patterns import GeometricPatternGenerator

generator = GeometricPatternGenerator(canvas, color_palette)
generator.generate(
    pattern_type="circles",
    count=25,
    min_radius=10,
    max_radius=50,
    palette="rainbow"
)
```

### Mandala Creation
```python
from patterns import MandalaGenerator

generator = MandalaGenerator(canvas, color_palette)
generator.generate(
    layers=10,
    elements_per_layer=16,
    base_shape="star"
)
```

### Fractal Art
```python
from patterns import FractalGenerator

generator = FractalGenerator(canvas, color_palette)
generator.generate(
    fractal_type="sierpinski",
    depth=5
)
```

### AI-Generated Patterns
```python
from agent import PatternAgent

agent = PatternAgent(canvas, color_palette)
agent.generate_pattern(steps=20)
```

## 🤖 AI Training

### Automatic Training
The AI agent learns to create better patterns through reinforcement learning:

```python
from train import PatternTrainer

trainer = PatternTrainer(canvas_size=800, output_dir="gallery")
trainer.train(episodes=100, steps_per_episode=15)
```

### Interactive Training
Get personalized patterns by providing feedback:

```python
from train import InteractiveTrainer

trainer = InteractiveTrainer(canvas_size=800, output_dir="gallery")
trainer.train_with_feedback(episodes=20, steps_per_episode=10)
```

### Training Parameters
- **Episodes**: Number of training iterations
- **Steps per Episode**: Pattern generation steps per training episode
- **Learning Rate**: Neural network learning rate (default: 0.001)
- **Epsilon**: Exploration rate for action selection
- **Memory Size**: Experience replay buffer size

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run specific test modules
pytest test_canvas.py
pytest test_patterns.py
pytest test_agent.py
pytest test_train.py

# Run with coverage
pytest --cov=. --cov-report=html
```

## 🎯 Pattern Evaluation

The system automatically evaluates pattern quality based on:

- **Symmetry**: Horizontal and vertical symmetry scores
- **Complexity**: Gradient-based complexity measurement
- **Color Harmony**: Color variety and distribution
- **Balance**: Visual balance and center of mass

```python
from agent import PatternEvaluator

evaluator = PatternEvaluator()
scores = evaluator.evaluate_pattern(canvas)
overall_score = evaluator.get_overall_score(scores)
```

## 🌈 Color Palettes

Built-in color palettes:
- **Rainbow**: Vibrant, colorful patterns
- **Pastel**: Soft, gentle colors
- **Monochrome**: Black and white variations
- **Earth**: Natural, earthy tones
- **Ocean**: Blue and teal shades

```python
from canvas import ColorPalette

palette = ColorPalette()
colors = palette.get_palette("rainbow")
random_color = palette.get_random_color("pastel")
```

## 📊 Web Interface

The Streamlit web application provides:

- **Interactive Pattern Generation**: Real-time pattern creation
- **Parameter Controls**: Adjustable pattern parameters
- **Gallery View**: Browse generated patterns
- **Training Dashboard**: Monitor AI training progress
- **Export Options**: Download patterns as PNG files

Access the web interface at `http://localhost:8501` after running `streamlit run app.py`.

## 🔧 Customization

### Adding New Pattern Types
```python
from patterns import PatternGenerator

class CustomPatternGenerator(PatternGenerator):
    def generate(self, **kwargs):
        # Your custom pattern generation logic
        pass
```

### Custom Color Palettes
```python
from canvas import ColorPalette

palette = ColorPalette()
palette.palettes['custom'] = ['#FF0000', '#00FF00', '#0000FF']
```

### Neural Network Architecture
Modify the neural network in `agent.py`:
```python
class CustomNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # Custom architecture
        pass
```

## 📈 Performance

- **Canvas Size**: Supports up to 2000x2000 pixels
- **Training Speed**: ~100 episodes/minute on CPU
- **Memory Usage**: ~500MB for standard training
- **Pattern Generation**: Real-time for simple patterns

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PIL/Pillow**: Image processing and drawing
- **PyTorch**: Neural network framework
- **Streamlit**: Web application framework
- **Matplotlib**: Plotting and visualization
- **NumPy**: Numerical computations

## 📞 Support

- **Issues**: Report bugs and feature requests on GitHub
- **Discussions**: Join community discussions
- **Documentation**: Check the code comments and docstrings

## 🎯 Roadmap

- [ ] **Animation Support**: Animated pattern generation
- [ ] **Style Transfer**: Apply artistic styles to patterns
- [ ] **3D Patterns**: Three-dimensional pattern generation
- [ ] **Mobile App**: iOS/Android application
- [ ] **Cloud Training**: Distributed training capabilities
- [ ] **API Service**: REST API for pattern generation

---

**PatternSmithAI** - Where mathematics meets creativity through artificial intelligence! 🧩✨
