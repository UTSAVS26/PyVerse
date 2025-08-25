# ⌨️ KeyMentor – Self-Improving Typing Speed & Accuracy Coach

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-87%20passed-brightgreen.svg)](tests/)

**KeyMentor** is an intelligent typing coach that learns from your typing patterns to identify weak areas and generates personalized exercises for improvement. Unlike traditional typing tutors that use pre-existing datasets, KeyMentor continuously adapts to your unique typing behavior.

## 🎯 Key Features

### 📊 **Real-Time Data Capture**
- Records individual keypress timings and accuracy
- Tracks typing speed (WPM) and error patterns
- Monitors reaction times and finger movements
- Stores comprehensive session data in SQLite database

### 🧠 **Intelligent Analysis**
- Identifies weak characters, bigrams, and trigrams
- Analyzes common typing mistakes and patterns
- Maps finger weaknesses based on QWERTY layout
- Generates personalized typing profiles

### 🎯 **Personalized Exercises**
- Character-focused drills for problematic letters
- Bigram/trigram practice for challenging combinations
- Progressive difficulty scaling based on improvement
- Mixed exercises targeting multiple weak areas

### 📈 **Progress Tracking**
- 7-day progress reports with trends
- WPM and accuracy improvement tracking
- Session history and performance analytics
- Export capabilities for data analysis

### 🖥️ **Dual Interface**
- **GUI Mode**: Full-featured Tkinter interface with tabs
- **CLI Mode**: Command-line interface for quick practice

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/keymentor.git
   cd keymentor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   # GUI Mode (default)
   python ui.py
   
   # CLI Mode
   python ui.py --cli
   ```

### Basic Usage

#### GUI Mode
1. Launch the application: `python ui.py`
2. Navigate through the tabs:
   - **Dashboard**: Overview of your typing stats
   - **Typing Practice**: Real-time typing sessions
   - **Analysis**: View your weak spots and patterns
   - **Exercises**: Practice personalized drills
   - **Progress**: Track your improvement over time

#### CLI Mode
```bash
python ui.py --cli
```

Choose from the menu:
- **1. Start Typing Practice**: Type custom or sample text
- **2. View Progress**: See your typing analysis
- **3. Generate Exercises**: Get personalized practice drills
- **4. Export Data**: Download your progress data
- **5. Exit**: Close the application

## 📁 Project Structure

```
KeyMentor/
├── tracker.py              # Real-time typing data capture
├── analyzer.py             # Weak spot analysis engine
├── exercise_generator.py   # Personalized exercise creation
├── ui.py                   # GUI and CLI interfaces
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── tests/                 # Comprehensive test suite
│   ├── test_tracker.py
│   ├── test_analyzer.py
│   ├── test_exercise_generator.py
│   └── test_ui.py
└── typing_data.db         # SQLite database (created automatically)
```

## 🏗️ Architecture

### Core Components

#### 1. **TypingTracker** (`tracker.py`)
- Manages typing sessions and real-time data capture
- Records individual keypress events with timestamps
- Calculates WPM, accuracy, and session metrics
- Persists data to SQLite database

#### 2. **TypingAnalyzer** (`analyzer.py`)
- Analyzes typing patterns to identify weak spots
- Processes character, bigram, and trigram statistics
- Maps finger weaknesses using QWERTY layout
- Generates comprehensive typing profiles

#### 3. **ExerciseGenerator** (`exercise_generator.py`)
- Creates personalized practice exercises
- Targets specific weak areas (characters, combinations)
- Implements progressive difficulty scaling
- Generates mixed exercises for comprehensive practice

#### 4. **User Interface** (`ui.py`)
- Tkinter-based GUI with tabbed interface
- Command-line interface for quick access
- Real-time typing practice environment
- Progress visualization and reporting

### Data Models

#### TypingEvent
```python
@dataclass
class TypingEvent:
    timestamp: float
    key: str
    expected_key: str
    is_correct: bool
    reaction_time: float
    session_id: str
```

#### TypingSession
```python
@dataclass
class TypingSession:
    session_id: str
    start_time: float
    end_time: float
    total_keys: int
    correct_keys: int
    wpm: float
    accuracy: float
    text_content: str
    mistakes: List[Tuple[str, str]]
```

#### WeakSpot
```python
@dataclass
class WeakSpot:
    pattern: str
    error_rate: float
    avg_reaction_time: float
    frequency: int
    difficulty_score: float
    pattern_type: str
```

## 🧪 Testing

The project includes comprehensive test coverage with 87 tests across all modules:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_tracker.py -v
python -m pytest tests/test_analyzer.py -v
python -m pytest tests/test_exercise_generator.py -v
python -m pytest tests/test_ui.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Test Categories
- **Unit Tests**: Individual component functionality
- **Integration Tests**: Component interaction testing
- **Database Tests**: SQLite operations and data persistence
- **UI Tests**: GUI and CLI interface testing
- **Edge Cases**: Error handling and boundary conditions

## 📊 Database Schema

### typing_sessions
| Column | Type | Description |
|--------|------|-------------|
| session_id | TEXT PRIMARY KEY | Unique session identifier |
| start_time | REAL | Session start timestamp |
| end_time | REAL | Session end timestamp |
| total_keys | INTEGER | Total keypresses recorded |
| correct_keys | INTEGER | Correct keypresses |
| wpm | REAL | Words per minute |
| accuracy | REAL | Accuracy percentage |
| text_content | TEXT | Typed text content |

### typing_events
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Auto-incrementing ID |
| session_id | TEXT | Foreign key to sessions |
| timestamp | REAL | Event timestamp |
| key | TEXT | Actual key pressed |
| expected_key | TEXT | Expected key |
| is_correct | BOOLEAN | Whether key was correct |
| reaction_time | REAL | Time since last keypress |

## 🔧 Configuration

### Environment Variables
- `KEYMENTOR_DB_PATH`: Custom database file path (default: `typing_data.db`)
- `KEYMENTOR_LOG_LEVEL`: Logging level (default: `INFO`)

### Customization
- **Exercise Difficulty**: Adjust minimum sample sizes in `analyzer.py`
- **Finger Mapping**: Modify QWERTY layout mapping in `analyzer.py`
- **UI Themes**: Customize Tkinter appearance in `ui.py`
- **Database**: Switch to PostgreSQL/MySQL by modifying connection logic

## 📈 Performance Metrics

KeyMentor tracks and analyzes:

### Speed Metrics
- **WPM (Words Per Minute)**: Standard typing speed measurement
- **Reaction Time**: Time between consecutive keypresses
- **Flow Rate**: Consistency of typing rhythm

### Accuracy Metrics
- **Overall Accuracy**: Percentage of correct keypresses
- **Character Accuracy**: Per-character error rates
- **Pattern Accuracy**: Bigram/trigram success rates

### Weakness Analysis
- **Character Weak Spots**: Problematic individual letters
- **Bigram Weak Spots**: Challenging two-letter combinations
- **Trigram Weak Spots**: Difficult three-letter patterns
- **Finger Weaknesses**: Error rates by finger position

## 🎮 Advanced Features

### Progressive Difficulty
- Exercises automatically scale based on improvement
- Difficulty multipliers: Easy (0.5x), Medium (1.0x), Hard (1.5x), Expert (2.0x)
- Dynamic adjustment based on recent performance

### Smart Exercise Generation
- **Character Exercises**: Focus on specific problematic letters
- **Bigram Exercises**: Practice challenging letter combinations
- **Trigram Exercises**: Master three-letter patterns
- **Mixed Exercises**: Comprehensive practice targeting multiple areas

### Data Export
- CSV export for external analysis
- JSON format for API integration
- Progress reports with trend analysis
- Session history with detailed metrics

## 🔮 Future Enhancements

### Planned Features
- [ ] **Multi-language Support**: International keyboard layouts
- [ ] **Advanced Analytics**: Machine learning-based pattern recognition
- [ ] **Gamification**: Achievements, streaks, and leaderboards
- [ ] **Cloud Sync**: Cross-device progress synchronization
- [ ] **API Integration**: RESTful API for third-party applications
- [ ] **Mobile App**: React Native or Flutter implementation

### Technical Improvements
- [ ] **Performance Optimization**: Caching and query optimization
- [ ] **Real-time Collaboration**: Multi-user typing competitions
- [ ] **Advanced UI**: Modern web interface with React/Vue.js
- [ ] **Machine Learning**: Predictive exercise recommendations
- [ ] **Accessibility**: Screen reader support and keyboard navigation

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone and setup
git clone https://github.com/yourusername/keymentor.git
cd keymentor
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/ -v

# Run linting
flake8 .
black .
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints throughout
- Write comprehensive docstrings
- Maintain test coverage above 90%

---

**Happy Typing!** 🚀

*KeyMentor - Your Personal Typing Coach*
