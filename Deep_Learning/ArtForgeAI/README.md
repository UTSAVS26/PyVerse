# 🎨 ArtForgeAI – Procedural Brushstroke Painter with Reinforcement Learning

ArtForgeAI is a creative AI agent that learns to paint abstract art using only brushstroke primitives (lines, curves, dots, splashes). Instead of relying on any dataset, the agent trains through reinforcement learning by experimenting with different brushstrokes and receiving feedback based on coverage, symmetry, and aesthetic scores.

## 🌟 Features

- **Dataset-free creativity** - Art emerges from agent exploration and primitive actions
- **Four brushstroke primitives**: Lines, curves, dots, and splashes
- **Reinforcement learning training** - Agent learns through trial and error
- **Multiple color palettes** - Monochrome, warm, cool, earth, and vibrant
- **Real-time visualization** - Watch the agent paint during training
- **Artwork gallery** - Save and view generated artworks
- **Training progress tracking** - Monitor learning curves and statistics

## 🏗️ Architecture

### Core Components

1. **Canvas (`canvas.py`)** - Digital painting surface with stroke rendering
2. **Strokes (`strokes.py`)** - Brushstroke primitive implementations
3. **Agent (`agent.py`)** - RL painter agent with actor-critic networks
4. **Training (`train.py`)** - Training loop with reward mechanisms

### Brushstroke Types

- **Line**: Straight or angled strokes with configurable thickness
- **Curve**: Curved strokes with control points
- **Dot**: Circular blobs with variable radius
- **Splash**: Random organic shapes

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ArtForgeAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Testing

1. **Quick Test** (recommended first):
```bash
python quick_test.py
```

2. **Basic Functionality Test**:
```bash
python test_basic.py
```

3. **Full Test Suite** (requires pytest):
```bash
python run_tests.py
```

### Basic Usage

1. **Start Training**:
```bash
python train.py
```

2. **Run Demo**:
```bash
python demo.py
```

3. **Custom Training**:
```python
from train import ArtForgeTrainer

# Create trainer with custom parameters
trainer = ArtForgeTrainer(
    canvas_width=800,
    canvas_height=600,
    max_strokes=50,
    save_dir="my_gallery"
)

# Train the agent
trainer.train(
    num_episodes=100,
    render_frequency=10,
    save_frequency=5
)
```

4. **Generate Artwork**:
```python
# Generate artwork using trained agent
artwork = trainer.generate_artwork(num_strokes=40, render=True)
```

## 🎯 Training Process

### Reward System

The agent receives rewards based on:

- **Coverage**: Percentage of canvas covered by strokes
- **Color Diversity**: Variety of colors used
- **Stroke Diversity**: Different types of strokes
- **Spatial Balance**: Distribution of strokes across canvas
- **Successful Application**: Penalty for failed strokes

### Training Parameters

- **State Space**: Flattened canvas image + metadata (coverage, diversity, stroke count)
- **Action Space**: 8-dimensional vector [stroke_type, x, y, angle, color_r, color_g, color_b, thickness]
- **Network Architecture**: Actor-critic with 256 hidden units
- **Learning Rates**: Actor: 1e-4, Critic: 1e-3

## 📁 Project Structure

```
ArtForgeAI/
├── canvas.py              # Canvas and rendering logic
├── strokes.py             # Brushstroke primitives
├── agent.py               # RL painter agent
├── train.py               # Training loop and trainer
├── demo.py                # Demonstration script
├── test_basic.py          # Basic functionality tests
├── quick_test.py          # Quick verification script
├── run_tests.py           # Full test suite runner
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── gallery/               # Generated artworks and models
│   ├── artworks/          # Saved paintings
│   └── models/            # Trained agent checkpoints
└── tests/                 # Unit and integration tests
    ├── test_canvas.py     # Canvas tests
    ├── test_strokes.py    # Stroke tests
    ├── test_agent.py      # Agent tests
    └── test_train.py      # Trainer tests
```

## 🧪 Testing

The project includes comprehensive testing:

### Test Scripts

- **`quick_test.py`**: Fast verification of basic functionality
- **`test_basic.py`**: Detailed basic functionality tests
- **`run_tests.py`**: Full test suite with pytest (85+ tests)

### Running Tests

```bash
# Quick verification
python quick_test.py

# Basic functionality
python test_basic.py

# Full test suite
python run_tests.py

# Individual test modules
python -m pytest tests/test_canvas.py -v
python -m pytest tests/test_strokes.py -v
python -m pytest tests/test_agent.py -v
python -m pytest tests/test_train.py -v
```

### Test Coverage

- **Canvas**: Stroke application, image operations, statistics
- **Strokes**: Color generation, stroke types, bounds checking
- **Agent**: Environment interaction, neural networks, training
- **Trainer**: Episode management, model saving, progress tracking

## 🎨 Generated Artworks

The training process automatically saves:

- **Intermediate artworks** every N episodes
- **Best performing models** based on reward and coverage
- **Training progress plots** showing learning curves
- **Final generated artwork** using the trained agent

## 🔧 Customization

### Color Palettes

```python
from strokes import StrokeGenerator

generator = StrokeGenerator(800, 600)

# Available palettes: 'monochrome', 'warm', 'cool', 'earth', 'vibrant'
stroke = generator.generate_line_stroke(palette='warm')
```

### Custom Reward Functions

Modify the reward calculation in `agent.py`:

```python
def _calculate_reward(self, success: bool) -> float:
    # Add your custom reward logic here
    reward = 1.0 if success else -1.0
    
    # Custom aesthetic scoring
    reward += self._calculate_symmetry_score() * 2.0
    reward += self._calculate_composition_score() * 1.5
    
    return reward
```

### Canvas Configuration

```python
from canvas import Canvas

# Custom canvas with different background
canvas = Canvas(
    width=1024,
    height=768,
    background_color=(240, 240, 240)  # Light gray
)
```

## 📊 Performance Metrics

The training tracks:

- **Episode Rewards**: Total reward per episode
- **Canvas Coverage**: Percentage of canvas covered
- **Color Diversity**: Variety of colors used
- **Training Losses**: Actor and critic network losses
- **Stroke Distribution**: Spatial distribution of strokes

## 🔮 Future Enhancements

### Planned Features

- **Interactive Mode**: User feedback for aesthetic adaptation
- **Style Modes**: Minimalist, chaotic, geometric styles
- **GIF Generation**: Stroke-by-stroke painting progression
- **Evolutionary Art**: Multiple agents competing for "better" art
- **High-Resolution Export**: Export artworks in various formats

### Advanced Features

- **Style Transfer**: Apply learned styles to new canvases
- **Collaborative Painting**: Multiple agents working together
- **User-Guided Training**: Incorporate human aesthetic preferences
- **Real-time Interaction**: Live painting with user input

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by procedural art and generative design
- Built with PyTorch for deep reinforcement learning
- Uses OpenCV and PIL for image processing
- Matplotlib for visualization and plotting

## 📞 Support

For questions, issues, or contributions:

- Open an issue on GitHub
- Check the documentation in the code
- Review the test files for usage examples

---

**ArtForgeAI** - Where creativity meets reinforcement learning! 🎨🤖
