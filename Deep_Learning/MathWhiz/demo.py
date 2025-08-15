#!/usr/bin/env python3
"""
MathWhiz Demo Script
Showcases the key features of the MathWhiz project.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.generator import MathExpressionGenerator
from model.tokenizer import MathTokenizer, create_tokenizer_from_data
from model.seq2seq_lstm import Seq2SeqLSTM
from utils.math_solver import MathSolver

def main():
    print("🧮 MathWhiz: AI-Powered Expression Generator & Step-by-Step Solver")
    print("=" * 70)
    print()
    
    # 1. Data Generation Demo
    print("1. 📊 Data Generation")
    print("-" * 30)
    generator = MathExpressionGenerator()
    expressions, solutions = generator.generate_dataset(10)
    print(f"Generated {len(expressions)} expression-solution pairs")
    print()
    
    # Show some examples
    print("Sample Expressions:")
    for i, (expr, sol) in enumerate(zip(expressions[:3], solutions[:3])):
        print(f"  {i+1}. Expression: {expr}")
        print(f"     Solution: {sol}")
        print()
    
    # 2. Math Solver Demo
    print("2. 🧮 Math Solver")
    print("-" * 30)
    solver = MathSolver()
    
    test_cases = [
        "2x + 3 = 9",
        "x^2 + 5x + 6",
        "(2x + 1)(3x + 2)",
        "2x + 3y - 4x + 5y"
    ]
    
    for expr in test_cases:
        solution = solver.solve_step_by_step(expr)
        is_correct = solver.verify_solution(expr, solution)
        print(f"Expression: {expr}")
        print(f"Solution: {solution}")
        print(f"Correct: {is_correct}")
        print()
    
    # 3. Tokenizer Demo
    print("3. 🔤 Tokenizer")
    print("-" * 30)
    tokenizer = create_tokenizer_from_data(expressions, solutions, vocab_size=200)
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    test_text = "2x + 3 = 9"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print()
    
    # 4. Model Demo
    print("4. 🤖 Model Architecture")
    print("-" * 30)
    vocab_size = tokenizer.get_vocab_size()
    model = Seq2SeqLSTM(
        vocab_size=vocab_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.1
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Vocabulary size: {vocab_size}")
    print()
    
    # 5. Project Summary
    print("5. 📋 Project Summary")
    print("-" * 30)
    print("✅ Complete MathWhiz project with:")
    print("   • Synthetic data generation")
    print("   • Step-by-step math solving")
    print("   • Custom tokenizer for math expressions")
    print("   • LSTM-based Seq2Seq model")
    print("   • Transformer model alternative")
    print("   • Comprehensive test suite")
    print("   • Training and evaluation scripts")
    print()
    print("🚀 Ready for training and deployment!")
    print("=" * 70)

if __name__ == "__main__":
    main() 