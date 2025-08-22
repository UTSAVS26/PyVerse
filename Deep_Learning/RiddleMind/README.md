# 🧠 RiddleMind – Logic Puzzle Solver Bot

A powerful logic puzzle solver that uses natural language processing and symbolic reasoning to solve complex riddles and logic puzzles.

## 🚀 Features

- **Natural Language Processing**: Parses logic puzzles written in plain English
- **Symbolic Reasoning**: Uses rule-based logic to deduce conclusions
- **Constraint Solving**: Handles complex relationships and comparisons
- **Step-by-step Solutions**: Provides detailed reasoning traces
- **Web Interface**: Beautiful Streamlit UI for easy interaction

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RiddleMind
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## 🎯 Usage

### Command Line Interface
```bash
python riddlemind/cli.py
```

### Web Interface
```bash
streamlit run riddlemind/web_app.py
```

### Python API
```python
from riddlemind.solver import RiddleMind

solver = RiddleMind()
result = solver.solve("If Alice is older than Bob and Charlie is younger than Alice, who is the oldest?")
print(result)
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v --cov=riddlemind
```

## 📁 Project Structure

```
RiddleMind/
├── riddlemind/
│   ├── __init__.py
│   ├── parser.py          # Natural language parsing
│   ├── solver.py          # Logic reasoning engine
│   ├── constraints.py     # Constraint representation
│   ├── cli.py            # Command line interface
│   └── web_app.py        # Streamlit web interface
├── tests/
│   ├── __init__.py
│   ├── test_parser.py
│   ├── test_solver.py
│   └── test_constraints.py
├── requirements.txt
└── README.md
```

## 🔧 How It Works

1. **Parsing**: Converts natural language into structured logical constraints
2. **Constraint Building**: Represents facts as Prolog-like predicates
3. **Reasoning**: Uses symbolic logic to derive new conclusions
4. **Output**: Provides step-by-step reasoning traces

## 📝 Example

**Input:**
```
If Alice is older than Bob and Charlie is younger than Alice, who is the oldest?
```

**Output:**
```
Parsed Constraints:
- older(Alice, Bob)
- younger(Charlie, Alice)

Reasoning Steps:
1. From 'Charlie is younger than Alice': Alice is older than Charlie
2. Alice is older than both Bob and Charlie
3. Therefore, Alice is the oldest

Conclusion: Alice is the oldest.
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.
