# 🧮 MathWhiz: AI-Powered Expression Generator & Step-by-Step Solver

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](tests/)

**MathWhiz** is a complete Sequence-to-Sequence (Seq2Seq) NLP model that generates random algebraic math problems and learns to solve them step-by-step using synthetic training data. It uses an encoder-decoder architecture (LSTM/Transformer) to convert math questions into human-readable solution steps.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

MathWhiz demonstrates how to build an AI system that can:
- **Generate synthetic mathematical expressions** automatically
- **Solve equations step-by-step** with human-readable explanations
- **Learn from synthetic data** without requiring human annotations
- **Handle multiple mathematical operations** including linear equations, polynomials, and arithmetic

### Use Cases

- 📚 **Math tutoring bots** and educational AI systems
- 🤖 **Intelligent education platforms** with automated problem generation
- 🧪 **Research on symbolic reasoning** using NLP techniques
- 🎓 **Educational tools** for learning mathematical concepts

## ✨ Features

### ✅ Core Functionality
- **100% Synthetic Dataset**: No external data required - generates unlimited training data
- **Multiple Math Operations**: Linear equations, polynomial simplification, arithmetic expressions
- **Step-by-Step Solutions**: Human-readable explanations with clear reasoning steps
- **Dual Model Support**: Both LSTM-based Seq2Seq and Transformer architectures
- **Comprehensive Evaluation**: Accuracy, BLEU score, and step-by-step correctness metrics

### 🧮 Supported Mathematical Operations

| Operation Type | Example Input | Example Output |
|----------------|---------------|----------------|
| **Linear Equations** | `2x + 3 = 9` | `Step 1: Subtract 3 from both sides → 2x = 6`<br>`Step 2: Divide both sides by 2 → x = 3` |
| **Polynomial Simplification** | `x² + 5x + 6` | `Step 1: Factor → (x + 2)(x + 3)` |
| **Polynomial Expansion** | `(2x + 1)(3x + 2)` | `Step 1: Use FOIL method → 6x² + 7x + 2` |
| **Arithmetic Simplification** | `2x + 3y - 4x + 5y` | `Step 1: Combine like terms → -2x + 8y` |

### 📊 Evaluation Metrics
- **Expression-level Accuracy**: Exact match between predicted and target solutions
- **Token-level BLEU Score**: N-gram precision for text generation quality
- **Step-by-Step Correctness**: Individual step accuracy analysis
- **Solution Verification**: Mathematical correctness validation

## 🏗️ Architecture

### System Components

```
MathWhiz/
├── 📊 Data Generation
│   ├── MathExpressionGenerator    # Synthetic expression generation
│   └── Dataset creation          # CSV format with expression-solution pairs
├── 🧮 Math Solver
│   ├── MathSolver               # Ground truth solution generation
│   └── Solution verification     # Mathematical correctness checking
├── 🔤 Tokenizer
│   ├── MathTokenizer            # Custom tokenizer for math expressions
│   └── Vocabulary building      # Dynamic vocabulary from training data
├── 🤖 Neural Models
│   ├── Seq2SeqLSTM             # LSTM-based encoder-decoder with attention
│   └── TransformerModel        # Transformer-based alternative
├── 🎯 Training Pipeline
│   ├── Data loading             # PyTorch DataLoader with custom collation
│   ├── Training loop            # Teacher forcing with gradient clipping
│   └── Model checkpointing     # Save/load trained models
└── 📈 Evaluation
    ├── Multiple metrics         # Accuracy, BLEU, step correctness
    └── Results analysis         # Detailed performance reporting
```

### Model Architecture Details

#### LSTM Seq2Seq Model
- **Encoder**: Bidirectional LSTM with dropout
- **Decoder**: LSTM with attention mechanism
- **Attention**: Bahdanau-style attention over encoder outputs
- **Teacher Forcing**: Configurable ratio for training stability

#### Transformer Model
- **Encoder**: Multi-head self-attention layers
- **Decoder**: Causal attention with cross-attention to encoder
- **Positional Encoding**: Sinusoidal positional embeddings
- **Layer Normalization**: Applied after each sublayer

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9 or higher
- CUDA (optional, for GPU acceleration)

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MathWhiz
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv mathwhiz_env
   source mathwhiz_env/bin/activate  # On Windows: mathwhiz_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (required for tokenization)
   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

5. **Verify installation**
   ```bash
   python tests/test_generator.py
   ```

### Dependencies

The project requires the following key dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥1.9.0 | Deep learning framework |
| `sympy` | ≥1.9 | Symbolic mathematics |
| `pandas` | ≥1.3.0 | Data manipulation |
| `numpy` | ≥1.21.0 | Numerical computing |
| `nltk` | ≥3.6 | Natural language processing |
| `tqdm` | ≥4.62.0 | Progress bars |
| `matplotlib` | ≥3.4.0 | Plotting and visualization |

## 🎯 Quick Start

### 1. Generate Sample Data

```python
from data.generator import MathExpressionGenerator

# Create generator
generator = MathExpressionGenerator()

# Generate 1000 expression-solution pairs
expressions, solutions = generator.generate_dataset(1000)

# Save to CSV
generator.save_dataset(expressions, solutions, "my_dataset.csv")
```

### 2. Solve Math Problems

```python
from utils.math_solver import MathSolver

# Create solver
solver = MathSolver()

# Solve a linear equation
expression = "2x + 3 = 9"
solution = solver.solve_step_by_step(expression)
print(solution)
# Output: "Step 1: Subtract 3 from both sides → 2x = 6 Step 2: Divide both sides by 2 → x = 3"

# Verify solution
is_correct = solver.verify_solution(expression, solution)
print(f"Correct: {is_correct}")  # True
```

### 3. Train a Model

```python
from model.tokenizer import create_tokenizer_from_data
from model.seq2seq_lstm import Seq2SeqLSTM
from training.train_seq2seq import MathDataset, collate_fn

# Create tokenizer
tokenizer = create_tokenizer_from_data(expressions, solutions, vocab_size=1000)

# Create dataset
dataset = MathDataset(expressions, solutions, tokenizer, max_length=50)

# Create model
model = Seq2SeqLSTM(
    vocab_size=tokenizer.get_vocab_size(),
    hidden_size=256,
    num_layers=2,
    dropout=0.1
)

# Train (see training section for complete example)
```

### 4. Run Demo

```bash
# Quick demo
python demo.py

# Quick training demo
python quick_train.py
```

## 📖 Usage Examples

### Data Generation

```python
from data.generator import MathExpressionGenerator

generator = MathExpressionGenerator()

# Generate different types of expressions
linear_expr, linear_sol = generator.generate_linear_equation()
poly_expr, poly_sol = generator.generate_polynomial_expression()
arith_expr, arith_sol = generator.generate_arithmetic_expression()

print(f"Linear: {linear_expr}")
print(f"Solution: {linear_sol}")
```

### Custom Tokenizer

```python
from model.tokenizer import MathTokenizer

# Create tokenizer
tokenizer = MathTokenizer(vocab_size=500)

# Fit on data
texts = ["2x + 3 = 9", "Step 1: Subtract 3 → 2x = 6"]
tokenizer.fit(texts)

# Encode/Decode
encoded = tokenizer.encode("2x + 3 = 9")
decoded = tokenizer.decode(encoded)

print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
```

### Model Inference

```python
import torch
from model.seq2seq_lstm import Seq2SeqLSTM

# Load trained model
model = Seq2SeqLSTM(vocab_size=1000, hidden_size=256)
# model.load_state_dict(torch.load('model.pth'))

# Prepare input
expression = "2x + 3 = 9"
expr_tokens = tokenizer.encode(expression)
expr_tensor = torch.tensor([expr_tokens], dtype=torch.long)

# Generate prediction
model.eval()
with torch.no_grad():
    pred_tokens = model.predict(expr_tensor, [len(expr_tokens)])
    pred_text = tokenizer.decode(pred_tokens[0].cpu().numpy())
    
print(f"Input: {expression}")
print(f"Prediction: {pred_text}")
```

## 🎯 Training

### Command Line Training

```bash
# Train LSTM model
python training/train_seq2seq.py \
    --epochs 50 \
    --batch_size 32 \
    --hidden_size 256 \
    --learning_rate 0.001 \
    --data_size 10000 \
    --save_dir models/

# Train with custom parameters
python training/train_seq2seq.py \
    --epochs 100 \
    --batch_size 64 \
    --hidden_size 512 \
    --num_layers 3 \
    --dropout 0.2 \
    --learning_rate 0.0005
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 32 | Training batch size |
| `--hidden_size` | 256 | LSTM hidden layer size |
| `--num_layers` | 2 | Number of LSTM layers |
| `--learning_rate` | 0.001 | Adam optimizer learning rate |
| `--dropout` | 0.1 | Dropout rate for regularization |
| `--vocab_size` | 1000 | Maximum vocabulary size |
| `--max_length` | 50 | Maximum sequence length |
| `--data_size` | 10000 | Number of training samples |
| `--save_dir` | models/ | Directory to save models |

### Training Monitoring

The training script provides:
- **Progress bars** with real-time loss updates
- **Validation metrics** after each epoch
- **Model checkpointing** (best model + periodic saves)
- **Training plots** (loss curves saved as PNG)

### Training Output

```
Epoch 1/50
Training: 100%|██████████| 250/250 [00:45<00:00, Loss: 4.2341]
Validation: 100%|██████████| 63/63 [00:12<00:00, Loss: 3.9876]
Train Loss: 4.2341, Val Loss: 3.9876

Model saved to models/best_model.pth
```

## 📊 Evaluation

### Command Line Evaluation

```bash
# Evaluate trained model
python evaluation/evaluate.py \
    --model_path models/best_model.pth \
    --model_type lstm \
    --test_size 100 \
    --output_file results.csv
```

### Evaluation Metrics

The evaluation script provides comprehensive metrics:

```python
# Example evaluation results
{
    'accuracy': 0.85,           # Exact match accuracy
    'bleu_score': 0.72,         # Token-level BLEU score
    'step_accuracy': 0.91,      # Step-by-step correctness
    'total_steps': 342,         # Total steps in test set
    'correct_steps': 311        # Correctly predicted steps
}
```

### Evaluation Output

```
==================================================
EVALUATION RESULTS
==================================================
Accuracy: 0.8500 (85.00%)
BLEU Score: 0.7200
Step Accuracy: 0.9100 (91.00%)
Correct Steps: 311/342
==================================================
```

## 🔧 API Reference

### MathExpressionGenerator

```python
class MathExpressionGenerator:
    def generate_linear_equation() -> Tuple[str, str]
    def generate_polynomial_expression() -> Tuple[str, str]
    def generate_arithmetic_expression() -> Tuple[str, str]
    def generate_dataset(num_samples: int) -> Tuple[List[str], List[str]]
    def save_dataset(expressions, solutions, filename: str)
    def load_dataset(filename: str) -> Tuple[List[str], List[str]]
```

### MathSolver

```python
class MathSolver:
    def solve_step_by_step(expression: str) -> str
    def verify_solution(expression: str, solution: str) -> bool
    def get_solution_steps(expression: str) -> List[str]
    def evaluate_expression(expression: str, x_value: float) -> float
```

### MathTokenizer

```python
class MathTokenizer:
    def fit(texts: List[str])
    def encode(text: str) -> List[int]
    def decode(indices: List[int]) -> str
    def encode_batch(texts: List[str]) -> List[List[int]]
    def decode_batch(batch_indices: List[List[int]]) -> List[str]
    def save(filepath: str)
    def load(filepath: str)
```

### Seq2SeqLSTM

```python
class Seq2SeqLSTM:
    def __init__(vocab_size, hidden_size, num_layers, dropout)
    def forward(src, src_lengths, tgt, teacher_forcing_ratio)
    def predict(src, src_lengths, max_length, sos_token, eos_token)
```

## 🧪 Testing

### Run All Tests

```bash
# Run comprehensive test suite
python tests/test_generator.py

# Expected output
Tests run: 21
Failures: 0
Errors: 0
```

### Test Coverage

The test suite covers:
- ✅ **Data Generation**: Expression and solution generation
- ✅ **Math Solver**: Step-by-step solving and verification
- ✅ **Tokenizer**: Encoding, decoding, and vocabulary building
- ✅ **Model Architecture**: Forward pass and prediction
- ✅ **File Operations**: Save/load functionality

### Individual Test Categories

```bash
# Test specific components
python -c "
from tests.test_generator import *
# Test data generation
test_generator = TestMathExpressionGenerator()
test_generator.test_generate_linear_equation()
"
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
├── 📄 README.md                  # This file
├── 📄 LICENSE                    # MIT License
└── 📄 PROJECT_SUMMARY.md         # Detailed project overview
```

## 🔧 Configuration

### Environment Variables

```bash
# Optional: Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Optional: Set random seed for reproducibility
export PYTHONHASHSEED=42
```

### Model Configuration

```python
# LSTM Model Configuration
lstm_config = {
    'vocab_size': 1000,
    'hidden_size': 256,
    'num_layers': 2,
    'dropout': 0.1,
    'max_length': 50
}

# Transformer Model Configuration
transformer_config = {
    'vocab_size': 1000,
    'd_model': 512,
    'nhead': 8,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'dim_feedforward': 2048,
    'dropout': 0.1
}
```

## 🚀 Deployment

### Production Deployment

1. **Train the model**
   ```bash
   python training/train_seq2seq.py --epochs 100 --save_dir production_models/
   ```

2. **Create inference script**
   ```python
   import torch
   from model.seq2seq_lstm import Seq2SeqLSTM
   from model.tokenizer import MathTokenizer
   
   # Load model and tokenizer
   checkpoint = torch.load('production_models/best_model.pth')
   model = Seq2SeqLSTM(vocab_size=1000, hidden_size=256)
   model.load_state_dict(checkpoint['model_state_dict'])
   tokenizer = checkpoint['tokenizer']
   
   # Inference function
   def solve_math_problem(expression):
       model.eval()
       with torch.no_grad():
           expr_tokens = tokenizer.encode(expression)
           expr_tensor = torch.tensor([expr_tokens], dtype=torch.long)
           pred_tokens = model.predict(expr_tensor, [len(expr_tokens)])
           return tokenizer.decode(pred_tokens[0].cpu().numpy())
   ```

3. **API Integration**
   ```python
   from flask import Flask, request, jsonify
   
   app = Flask(__name__)
   
   @app.route('/solve', methods=['POST'])
   def solve():
       data = request.json
       expression = data['expression']
       solution = solve_math_problem(expression)
       return jsonify({'expression': expression, 'solution': solution})
   ```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests for new functionality**
   ```bash
   python tests/test_generator.py
   ```
5. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
6. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Development Setup

```bash
# Clone and setup development environment
git clone <repository-url>
cd MathWhiz
python -m venv dev_env
source dev_env/bin/activate  # On Windows: dev_env\Scripts\activate
pip install -r requirements.txt
pip install pytest pytest-cov  # For testing
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for function parameters
- Add docstrings for all public functions
- Keep functions focused and modular

---

**MathWhiz** - Making mathematical reasoning accessible through AI! 🧮✨

---

**@SK8-infi** 