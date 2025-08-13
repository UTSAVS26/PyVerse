import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import os
import sys
from typing import List, Tuple, Dict
import re
from collections import Counter

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generator import MathExpressionGenerator
from model.tokenizer import MathTokenizer
from model.seq2seq_lstm import Seq2SeqLSTM
from model.transformer import TransformerModel
from utils.math_solver import MathSolver

def calculate_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Calculate exact match accuracy."""
    correct = 0
    total = len(predictions)
    
    for pred, target in zip(predictions, targets):
        if pred.strip() == target.strip():
            correct += 1
    
    return correct / total if total > 0 else 0.0

def calculate_bleu_score(predictions: List[str], targets: List[str]) -> float:
    """Calculate BLEU score for token-level evaluation."""
    def tokenize(text: str) -> List[str]:
        return text.lower().split()
    
    total_bleu = 0.0
    total_samples = len(predictions)
    
    for pred, target in zip(predictions, targets):
        pred_tokens = tokenize(pred)
        target_tokens = tokenize(target)
        
        # Calculate n-gram precision
        n_grams = 4
        bleu_score = 0.0
        
        for n in range(1, n_grams + 1):
            pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)]
            target_ngrams = [tuple(target_tokens[i:i+n]) for i in range(len(target_tokens) - n + 1)]
            
            if not pred_ngrams:
                continue
                
            # Count matches
            pred_counter = Counter(pred_ngrams)
            target_counter = Counter(target_ngrams)
            
            matches = sum(min(pred_counter[ngram], target_counter[ngram]) for ngram in pred_ngrams)
            precision = matches / len(pred_ngrams) if pred_ngrams else 0.0
            
            bleu_score += precision
        
        # Calculate brevity penalty
        if len(pred_tokens) < len(target_tokens):
            bp = np.exp(1 - len(target_tokens) / len(pred_tokens)) if pred_tokens else 0.0
        else:
            bp = 1.0
        
        bleu_score = bp * (bleu_score / n_grams)
        total_bleu += bleu_score
    
    return total_bleu / total_samples if total_samples > 0 else 0.0

def calculate_step_correctness(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """Calculate step-by-step correctness metrics."""
    def extract_steps(solution: str) -> List[str]:
        """Extract individual steps from solution string."""
        steps = []
        for line in solution.split('Step'):
            if line.strip():
                steps.append(line.strip())
        return steps
    
    total_steps = 0
    correct_steps = 0
    step_accuracy = 0.0
    
    for pred, target in zip(predictions, targets):
        pred_steps = extract_steps(pred)
        target_steps = extract_steps(target)
        
        total_steps += len(target_steps)
        
        # Count correct steps
        for pred_step, target_step in zip(pred_steps, target_steps):
            if pred_step.strip() == target_step.strip():
                correct_steps += 1
    
    step_accuracy = correct_steps / total_steps if total_steps > 0 else 0.0
    
    return {
        'step_accuracy': step_accuracy,
        'total_steps': total_steps,
        'correct_steps': correct_steps
    }

def evaluate_model(model: nn.Module, tokenizer: MathTokenizer, test_expressions: List[str], 
                  test_solutions: List[str], device: torch.device) -> Dict[str, float]:
    """Evaluate model performance."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for expression in test_expressions:
            # Tokenize expression
            expr_tokens = tokenizer.encode(expression)
            expr_tensor = torch.tensor([expr_tokens], dtype=torch.long).to(device)
            expr_lengths = [len(expr_tokens)]
            
            # Generate prediction
            if isinstance(model, Seq2SeqLSTM):
                pred_tokens = model.predict(expr_tensor, expr_lengths)
            else:  # Transformer
                pred_tokens = model.predict(expr_tensor)
            
            # Decode prediction
            pred_sequence = pred_tokens[0].cpu().numpy()
            pred_text = tokenizer.decode(pred_sequence)
            
            # Remove padding and special tokens
            pred_text = clean_prediction(pred_text)
            predictions.append(pred_text)
    
    # Calculate metrics
    accuracy = calculate_accuracy(predictions, test_solutions)
    bleu_score = calculate_bleu_score(predictions, test_solutions)
    step_metrics = calculate_step_correctness(predictions, test_solutions)
    
    return {
        'accuracy': accuracy,
        'bleu_score': bleu_score,
        'step_accuracy': step_metrics['step_accuracy'],
        'total_steps': step_metrics['total_steps'],
        'correct_steps': step_metrics['correct_steps']
    }

def clean_prediction(prediction: str) -> str:
    """Clean prediction by removing padding and special tokens."""
    # Remove special tokens
    prediction = re.sub(r'<PAD>|<UNK>|<SOS>|<EOS>', '', prediction)
    
    # Remove extra whitespace
    prediction = ' '.join(prediction.split())
    
    return prediction

def load_model_and_tokenizer(model_path: str, model_type: str = 'lstm') -> Tuple[nn.Module, MathTokenizer]:
    """Load trained model and tokenizer."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Load tokenizer
    tokenizer = checkpoint['tokenizer']
    
    # Create model
    if model_type == 'lstm':
        model = Seq2SeqLSTM(
            vocab_size=tokenizer.get_vocab_size(),
            hidden_size=256,
            num_layers=2,
            dropout=0.1
        )
    else:  # transformer
        model = TransformerModel(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, tokenizer

def generate_test_samples(num_samples: int = 100) -> Tuple[List[str], List[str]]:
    """Generate test samples for evaluation."""
    generator = MathExpressionGenerator()
    expressions, solutions = generator.generate_dataset(num_samples)
    return expressions, solutions

def print_evaluation_results(results: Dict[str, float]):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"BLEU Score: {results['bleu_score']:.4f}")
    print(f"Step Accuracy: {results['step_accuracy']:.4f} ({results['step_accuracy']*100:.2f}%)")
    print(f"Correct Steps: {results['correct_steps']}/{results['total_steps']}")
    print("="*50)

def save_evaluation_results(results: Dict[str, float], predictions: List[str], 
                          targets: List[str], expressions: List[str], filepath: str):
    """Save evaluation results to CSV file."""
    df = pd.DataFrame({
        'expression': expressions,
        'target': targets,
        'prediction': predictions,
        'correct': [pred.strip() == target.strip() for pred, target in zip(predictions, targets)]
    })
    
    df.to_csv(filepath, index=False)
    print(f"Evaluation results saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'transformer'], 
                       help='Type of model to evaluate')
    parser.add_argument('--test_size', type=int, default=100, help='Number of test samples')
    parser.add_argument('--output_file', type=str, default='evaluation_results.csv', 
                       help='Output file for detailed results')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.model_type)
    model.to(device)
    model.eval()
    
    # Generate test data
    print(f"Generating {args.test_size} test samples...")
    test_expressions, test_solutions = generate_test_samples(args.test_size)
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, tokenizer, test_expressions, test_solutions, device)
    
    # Print results
    print_evaluation_results(results)
    
    # Save detailed results
    predictions = []
    model.eval()
    with torch.no_grad():
        for expression in test_expressions:
            expr_tokens = tokenizer.encode(expression)
            expr_tensor = torch.tensor([expr_tokens], dtype=torch.long).to(device)
            expr_lengths = [len(expr_tokens)]
            
            if isinstance(model, Seq2SeqLSTM):
                pred_tokens = model.predict(expr_tensor, expr_lengths)
            else:
                pred_tokens = model.predict(expr_tensor)
            
            pred_sequence = pred_tokens[0].cpu().numpy()
            pred_text = tokenizer.decode(pred_sequence)
            pred_text = clean_prediction(pred_text)
            predictions.append(pred_text)
    
    save_evaluation_results(results, predictions, test_solutions, test_expressions, args.output_file)
    
    # Print some examples
    print("\nExample Predictions:")
    print("-" * 50)
    for i in range(min(5, len(test_expressions))):
        print(f"Expression: {test_expressions[i]}")
        print(f"Target: {test_solutions[i]}")
        print(f"Prediction: {predictions[i]}")
        print(f"Correct: {predictions[i].strip() == test_solutions[i].strip()}")
        print("-" * 50)

if __name__ == "__main__":
    main() 