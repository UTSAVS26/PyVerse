# 🧮 MathWhiz: AI-Powered Expression Generator & Step-by-Step Solver

## 📋 Overview

This PR introduces **MathWhiz**, a complete AI-powered mathematical expression generator and step-by-step solver built using Sequence-to-Sequence (Seq2Seq) models. The project demonstrates how to create synthetic training data and train neural networks to solve mathematical problems with human-readable explanations.

## ✨ Key Features

### 🎯 Core Functionality
- **100% Synthetic Dataset Generation** - No external data required, generates unlimited training data
- **Multiple Mathematical Operations** - Linear equations, polynomial simplification, arithmetic expressions
- **Step-by-Step Solutions** - Human-readable explanations with clear reasoning steps
- **Dual Model Architecture** - Both LSTM-based Seq2Seq and Transformer models
- **Comprehensive Evaluation** - Accuracy, BLEU score, and step-by-step correctness metrics

### 🧮 Supported Mathematical Operations

| Operation Type | Example Input | Example Output |
|----------------|---------------|----------------|
| **Linear Equations** | `2x + 3 = 9` | `Step 1: Subtract 3 from both sides → 2x = 6`<br>`Step 2: Divide both sides by 2 → x = 3` |
| **Polynomial Simplification** | `x² + 5x + 6` | `Step 1: Factor → (x + 2)(x + 3)` |
| **Polynomial Expansion** | `(2x + 1)(3x + 2)` | `Step 1: Use FOIL method → 6x² + 7x + 2` |
| **Arithmetic Simplification** | `2x + 3y - 4x + 5y` | `Step 1: Combine like terms → -2x + 8y` |

## 🏗️ Architecture

### System Components
- **Data Generation** (`data/generator.py`) - Synthetic expression generation
- **Math Solver** (`utils/math_solver.py`) - Ground truth solution generation
- **Tokenizer** (`model/tokenizer.py`) - Custom tokenizer for math expressions
- **Neural Models** - LSTM Seq2Seq and Transformer architectures
- **Training Pipeline** (`training/train_seq2seq.py`) - Complete training script
- **Evaluation** (`evaluation/evaluate.py`) - Multiple evaluation metrics

### Model Architecture Details
- **LSTM Seq2Seq**: Bidirectional LSTM with attention mechanism
- **Transformer**: Multi-head self-attention with positional encoding
- **Custom Tokenizer**: Handles mathematical symbols and operators
- **Teacher Forcing**: Configurable ratio for training stability

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample data
python data/generator.py

# Run tests
python tests/test_generator.py

# Quick demo
python demo.py

# Quick training
python quick_train.py
```

## 📊 Testing Results

All tests pass successfully:
```
Tests run: 21
Failures: 0
Errors: 0
```

## 📁 Project Structure

```
MathWhiz/
├── 📊 data/
│   └── generator.py              # Synthetic data generation
├── 🤖 model/
│   ├── tokenizer.py              # Custom math tokenizer
│   ├── seq2seq_lstm.py           # LSTM-based Seq2Seq model
│   └── transformer.py            # Transformer model
├── 🎯 training/
│   └── train_seq2seq.py          # Training script with CLI
├── 📈 evaluation/
│   └── evaluate.py               # Evaluation and metrics
├── 🧮 utils/
│   └── math_solver.py            # Math solving utilities
├── 🧪 tests/
│   └── test_generator.py         # Comprehensive test suite
├── 📖 examples/
│   └── demo_notebook.ipynb       # Jupyter notebook demo
├── 🚀 demo.py                    # Quick demo script
├── ⚡ quick_train.py             # Quick training demo
├── 📋 requirements.txt            # Python dependencies
├── 📄 README.md                  # Comprehensive documentation
├── 📄 LICENSE                    # MIT License
└── 📄 PROJECT_SUMMARY.md         # Detailed project overview
```

## 🔧 Technical Implementation

### Data Generation
- **Synthetic Expression Generator**: Creates linear equations, polynomials, and arithmetic expressions
- **Step-by-Step Solution Generation**: Auto-generates human-readable solution steps
- **Solution Verification**: Mathematical correctness validation using SymPy

### Neural Models
- **LSTM Seq2Seq**: Bidirectional encoder with attention-based decoder
- **Transformer**: Multi-head attention with positional encoding
- **Custom Tokenizer**: Vocabulary building from mathematical expressions
- **Teacher Forcing**: Configurable during training for stability

### Training Pipeline
- **Data Loading**: PyTorch DataLoader with custom collation
- **Training Loop**: Teacher forcing with gradient clipping
- **Model Checkpointing**: Save/load trained models
- **Training Visualization**: Loss curves and metrics plotting

### Evaluation Metrics
- **Expression-level Accuracy**: Exact match between predicted and target solutions
- **Token-level BLEU Score**: N-gram precision for text generation quality
- **Step-by-Step Correctness**: Individual step accuracy analysis
- **Solution Verification**: Mathematical correctness validation

## 🎓 Educational Value

This project demonstrates:
1. **Synthetic Data Generation** for ML training without human annotation
2. **Sequence-to-Sequence Models** for translation tasks
3. **Mathematical Reasoning** with neural networks
4. **Complete ML Pipeline** with testing and evaluation
5. **Research Applications** for educational AI systems

## 🚀 Usage Examples

### Data Generation
```python
from data.generator import MathExpressionGenerator

generator = MathExpressionGenerator()
expressions, solutions = generator.generate_dataset(1000)
generator.save_dataset(expressions, solutions, "dataset.csv")
```

### Math Solving
```python
from utils.math_solver import MathSolver

solver = MathSolver()
solution = solver.solve_step_by_step("2x + 3 = 9")
# Output: "Step 1: Subtract 3 from both sides → 2x = 6 Step 2: Divide both sides by 2 → x = 3"
```

### Model Training
```bash
python training/train_seq2seq.py \
    --epochs 50 \
    --batch_size 32 \
    --hidden_size 256 \
    --learning_rate 0.001 \
    --data_size 10000
```

### Model Evaluation
```bash
python evaluation/evaluate.py \
    --model_path models/best_model.pth \
    --model_type lstm \
    --test_size 100
```

## 🔧 Configuration Options

### Training Parameters
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Training batch size (default: 32)
- `--hidden_size`: LSTM hidden layer size (default: 256)
- `--num_layers`: Number of LSTM layers (default: 2)
- `--learning_rate`: Adam optimizer learning rate (default: 0.001)
- `--dropout`: Dropout rate for regularization (default: 0.1)
- `--vocab_size`: Maximum vocabulary size (default: 1000)
- `--data_size`: Number of training samples (default: 10000)

## 📈 Performance

### Model Specifications
- **Parameters**: ~1M (configurable)
- **Training Time**: ~10-30 minutes on CPU
- **Memory Usage**: ~2GB for training
- **Inference Time**: <1 second per expression

### Evaluation Results
- **Accuracy**: Expression-level exact match accuracy
- **BLEU Score**: Token-level generation quality
- **Step Correctness**: Individual step accuracy
- **Solution Verification**: Mathematical correctness validation

## 🧪 Testing

### Comprehensive Test Suite
- **Data Generation Tests**: Expression and solution generation
- **Math Solver Tests**: Step-by-step solving and verification
- **Tokenizer Tests**: Encoding, decoding, and vocabulary building
- **Model Architecture Tests**: Forward pass and prediction
- **File Operation Tests**: Save/load functionality

### Test Coverage
```
Tests run: 21
Failures: 0
Errors: 0
```

## 🚀 Future Enhancements

### Possible Extensions
- [ ] Support for more complex math (calculus, linear algebra)
- [ ] Web interface for interactive solving
- [ ] Voice-based input/output
- [ ] Multi-language support
- [ ] Real-world dataset integration
- [ ] Model distillation for deployment

### Research Applications
- Educational AI tutors
- Automated homework assistance
- Mathematical reasoning research
- Symbolic computation with neural networks

## 📄 Documentation

### Complete Documentation
- **README.md**: Comprehensive project documentation
- **PROJECT_SUMMARY.md**: Detailed technical overview
- **API Reference**: Complete class and method documentation
- **Usage Examples**: Code snippets and tutorials
- **Installation Guide**: Step-by-step setup instructions

### Code Quality
- **Type Hints**: Full type annotation for all functions
- **Docstrings**: Comprehensive documentation for all classes and methods
- **PEP 8 Compliance**: Clean, readable code style
- **Modular Design**: Focused, reusable components

## 🤝 Contributing

### Development Guidelines
- Follow PEP 8 style guidelines
- Use type hints for function parameters
- Add docstrings for all public functions
- Keep functions focused and modular
- Add tests for new functionality

### Setup Instructions
```bash
git clone <repository-url>
cd MathWhiz
python -m venv dev_env
source dev_env/bin/activate  # On Windows: dev_env\Scripts\activate
pip install -r requirements.txt
pip install pytest pytest-cov  # For testing
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch** for the deep learning framework
- **SymPy** for symbolic mathematical computations
- **NLTK** for natural language processing utilities
- **Research community** for inspiration on symbolic reasoning in NLP

---

**MathWhiz** - Making mathematical reasoning accessible through AI! 🧮✨

---

**@SK8-infi** 