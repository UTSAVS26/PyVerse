# 🧠 MemoAI: Adaptive Memory Card Game Trainer

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-45%20passed-brightgreen.svg)](tests/)

> An intelligent memory card-matching game that uses **machine learning** to adapt difficulty based on player behavior, reaction times, and performance patterns.

## 🎯 Overview

**MemoAI** is an interactive memory card game that combines traditional gameplay with cutting-edge AI technology. The game tracks your behavior—reaction times, mistakes, and patterns—to train a lightweight ML model that intelligently adjusts difficulty in real-time.

### ✨ Key Features

- 🎮 **Dynamic Memory Game**: 3x3 to 6x6 grid sizes with emoji-based cards
- ⏱️ **Real-time Analytics**: Tracks reaction time, mistakes, and completion speed
- 🧠 **Adaptive AI**: ML model learns from your gameplay and adjusts difficulty
- 📊 **Performance Insights**: Detailed analytics and personalized feedback
- 🎯 **Progressive Learning**: Tracks improvement over multiple sessions
- 🌐 **Beautiful Web Interface**: Modern Streamlit-based UI with custom styling

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SK8-infi/MemoAI.git
   cd MemoAI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run ui/streamlit_app.py
   ```

4. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Start playing and watch the AI learn!

## 🎮 How to Play

1. **Select Difficulty**: Choose from Easy (3x3) to Expert (6x6)
2. **Click Cards**: Reveal cards by clicking on them
3. **Find Matches**: Match pairs of identical cards to clear them
4. **Complete the Board**: Clear all pairs to finish the game
5. **Learn from AI**: Get personalized feedback and difficulty recommendations

### Game Mechanics

- **Reaction Time Tracking**: Every click is timed for analysis
- **Mistake Detection**: Wrong matches are counted and analyzed
- **Performance Scoring**: Overall score based on speed, accuracy, and efficiency
- **Adaptive Difficulty**: AI suggests optimal difficulty for next round

## 🧠 AI Components

### Pattern Learner
- **Random Forest Models**: Predicts optimal difficulty and performance
- **Feature Extraction**: Analyzes reaction time, mistakes, completion time
- **Training Data**: Learns from 5+ completed game sessions
- **Real-time Updates**: Model improves with each new session

### Adaptive Difficulty System
- **Dynamic Adjustment**: Changes grid size, time limits, and complexity
- **Performance Analysis**: Evaluates strengths and weaknesses
- **Personalized Recommendations**: Provides specific improvement tips
- **Confidence Scoring**: Indicates prediction reliability

### Analytics Dashboard
- **Performance Trends**: Visualizes improvement over time
- **Behavioral Insights**: Identifies patterns in gameplay
- **Predictive Analytics**: Forecasts expected performance
- **Motivational Feedback**: Encourages continued improvement

## 📊 Performance Metrics

The AI tracks and analyzes:

| Metric | Description | Impact |
|--------|-------------|---------|
| **Reaction Time** | Time between card clicks | Speed assessment |
| **Mistake Rate** | Wrong matches per game | Accuracy measurement |
| **Completion Time** | Total time to finish | Efficiency evaluation |
| **Grid Performance** | Success rate by grid size | Difficulty optimization |
| **Improvement Trend** | Progress over sessions | Learning curve analysis |

## 🏗️ Architecture

```
MemoAI/
├── game/
│   └── memory_game.py          # Core game logic and state management
├── ml/
│   ├── pattern_learner.py      # ML models for behavior analysis
│   └── adaptive_difficulty.py  # Difficulty adjustment algorithms
├── ui/
│   └── streamlit_app.py        # Web interface and visualizations
├── utils/
│   └── timer.py                # Timing and session management
├── data/
│   └── user_sessions.json      # Persistent session storage
├── tests/
│   └── test_game_logic.py      # Comprehensive test suite
└── assets/
    └── card_images/            # Game assets and images
```

### Technology Stack

- **Backend**: Python 3.8+
- **Web Framework**: Streamlit 1.28+
- **Machine Learning**: Scikit-learn, NumPy, Pandas
- **Data Visualization**: Plotly, Matplotlib
- **Testing**: Pytest
- **Data Storage**: JSON

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/test_game_logic.py -v

# Run specific test categories
python -m pytest tests/test_game_logic.py::TestMemoryGame -v
python -m pytest tests/test_game_logic.py::TestPatternLearner -v
```

### Test Coverage

- ✅ **45 test cases** covering all components
- ✅ **Game Logic**: Card matching, state management, statistics
- ✅ **ML Models**: Training, prediction, feature extraction
- ✅ **Timer System**: Session tracking, reaction time analysis
- ✅ **Integration**: Complete game flow testing

## 📈 Usage Examples

### Basic Game Session
```python
from game.memory_game import MemoryGame
from utils.timer import SessionTimer

# Initialize game
game = MemoryGame(4)  # 4x4 grid
timer = SessionTimer()

# Start session
game.start_game()
timer.start_session(4)

# Play game
result = game.click_card(0, 0)
timer.record_move()

# Get analytics
stats = game.get_game_stats()
print(f"Completion time: {stats['total_time']:.1f}s")
```

### ML Model Training
```python
from ml.pattern_learner import PatternLearner

# Initialize learner
learner = PatternLearner()

# Add training data
learner.add_session(session_data)

# Train model
result = learner.train_models()
print(f"Model accuracy: {result['difficulty_accuracy']:.2f}")
```

## 🔧 Configuration

### Difficulty Settings

| Level | Grid Size | Time Limit | Max Mistakes | Complexity |
|-------|-----------|------------|--------------|------------|
| Easy | 3x3 | 5 min | 10 | Simple |
| Medium | 4x4 | 4 min | 8 | Simple |
| Hard | 5x5 | 3 min | 6 | Medium |
| Expert | 6x6 | 2 min | 4 | Complex |

### ML Model Parameters

- **Training Threshold**: 5 completed sessions minimum
- **Feature Count**: 7 behavioral metrics
- **Model Type**: Random Forest (Classification & Regression)
- **Update Frequency**: After each completed session

## 🚀 Deployment

### Local Development
```bash
# Install in development mode
pip install -e .

# Run with hot reload
streamlit run ui/streamlit_app.py --server.runOnSave true
```

### Production Deployment
```bash
# Install production dependencies
pip install -r requirements.txt

# Run with production settings
streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests** for new functionality
5. **Run the test suite**
   ```bash
   python -m pytest tests/ -v
   ```
6. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
7. **Push to your branch**
   ```bash
   git push origin feature/amazing-feature
   ```
8. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Write tests for new features
- Update documentation as needed

## 🐛 Troubleshooting

### Common Issues

**Q: Streamlit app won't start**
```bash
# Check Python version
python --version  # Should be 3.8+

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Q: ML model not training**
```bash
# Ensure you have 5+ completed games
# Check data file permissions
ls -la data/user_sessions.json
```

**Q: Tests failing**
```bash
# Clear pytest cache
python -m pytest --cache-clear

# Run with verbose output
python -m pytest tests/ -v -s
```

## 📚 API Reference

### MemoryGame Class
```python
class MemoryGame:
    def __init__(self, grid_size: int = 4)
    def start_game(self) -> None
    def click_card(self, row: int, col: int) -> Dict[str, Any]
    def get_game_stats(self) -> Dict[str, Any]
    def is_game_complete(self) -> bool
```

### PatternLearner Class
```python
class PatternLearner:
    def train_models(self) -> Dict[str, Any]
    def predict_difficulty(self, session_data: Dict) -> Dict[str, Any]
    def get_player_insights(self, sessions: List[Dict]) -> Dict[str, Any]
```

### AdaptiveDifficulty Class
```python
class AdaptiveDifficulty:
    def suggest_next_difficulty(self, session_data: Dict) -> Dict[str, Any]
    def get_personalized_feedback(self, session_data: Dict) -> Dict[str, Any]
    def get_adaptive_hints(self, session_data: Dict) -> List[str]
```

## 🔮 Roadmap

### Planned Features
- [ ] **Multiplayer Mode**: Collaborative learning with friends
- [ ] **Advanced ML Models**: RNNs and attention-based models
- [ ] **Voice Guidance**: Audio feedback and instructions
- [ ] **Leaderboards**: Global and local performance rankings
- [ ] **Custom Themes**: Different card sets and visual styles
- [ ] **Export Analytics**: Download performance reports
- [ ] **Mobile Support**: Responsive design for mobile devices

### Research Applications
- [ ] **Cognitive Assessment**: Memory and attention testing
- [ ] **Learning Analytics**: Educational performance tracking
- [ ] **Behavioral Research**: User interaction pattern analysis
- [ ] **Accessibility Features**: Support for users with disabilities

---

## 👨‍💻 Author

**@SK8-infi** - Creator of MemoAI

---

**Happy Gaming and Learning! 🎮🧠**

---

*Made with ❤️ for the AI and gaming communities* 