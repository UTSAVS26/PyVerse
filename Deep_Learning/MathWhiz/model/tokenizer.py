import re
from typing import List, Dict, Tuple
from collections import Counter
import pickle
import os

class MathTokenizer:
    """
    Custom tokenizer for mathematical expressions and solutions.
    Handles mathematical symbols, operators, and step-by-step solutions.
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,
            '<EOS>': 3
        }
        self.is_fitted = False
    
    def fit(self, texts: List[str]):
        """Build vocabulary from a list of texts."""
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = self._tokenize(text)
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Build vocabulary (special tokens + most frequent tokens)
        vocab = list(self.special_tokens.keys())
        vocab.extend([token for token, count in token_counts.most_common(self.vocab_size - len(self.special_tokens))])
        
        # Create mappings
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        self.is_fitted = True
        print(f"Vocabulary built with {len(self.word2idx)} tokens")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a single text string."""
        # Handle mathematical expressions and step-by-step solutions
        tokens = []
        
        # Split by spaces first
        parts = text.split()
        
        for part in parts:
            # Handle mathematical expressions
            if any(op in part for op in ['+', '-', '*', '/', '=', '^', '(', ')', 'x', 'y']):
                # Split mathematical expressions
                math_tokens = self._tokenize_math_expression(part)
                tokens.extend(math_tokens)
            else:
                # Handle regular words and numbers
                word_tokens = self._tokenize_word(part)
                tokens.extend(word_tokens)
        
        return tokens
    
    def _tokenize_math_expression(self, expr: str) -> List[str]:
        """Tokenize mathematical expressions."""
        tokens = []
        
        # Pattern for mathematical tokens
        pattern = r'(\d+|[+\-*/=^()]|[xy]|\^|\d+\.\d+)'
        matches = re.findall(pattern, expr)
        
        for match in matches:
            if match in ['+', '-', '*', '/', '=', '^', '(', ')', 'x', 'y']:
                tokens.append(match)
            elif match.isdigit() or '.' in match:
                tokens.append(match)
            else:
                # Handle special cases like x², y³, etc.
                if '^' in match:
                    base, exp = match.split('^')
                    tokens.extend([base, '^', exp])
                else:
                    tokens.append(match)
        
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize regular words and numbers."""
        tokens = []
        
        # Handle step numbers like "Step 1:", "Step 2:"
        step_match = re.match(r'Step\s+(\d+):', word)
        if step_match:
            tokens.extend(['Step', step_match.group(1), ':'])
            return tokens
        
        # Handle arrows and special symbols
        if '→' in word:
            parts = word.split('→')
            for part in parts:
                if part.strip():
                    tokens.extend(self._tokenize_word(part.strip()))
            tokens.append('→')
            return tokens
        
        # Handle regular words
        tokens.append(word.lower())
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token indices."""
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before encoding")
        
        tokens = self._tokenize(text)
        indices = []
        
        for token in tokens:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                indices.append(self.word2idx['<UNK>'])
        
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """Convert token indices back to text."""
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before decoding")
        
        tokens = []
        for idx in indices:
            if idx in self.idx2word:
                tokens.append(self.idx2word[idx])
            else:
                tokens.append('<UNK>')
        
        return ' '.join(tokens)
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of texts."""
        return [self.encode(text) for text in texts]
    
    def decode_batch(self, batch_indices: List[List[int]]) -> List[str]:
        """Decode a batch of token indices."""
        return [self.decode(indices) for indices in batch_indices]
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return len(self.word2idx)
    
    def save(self, filepath: str):
        """Save the tokenizer to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'vocab_size': self.vocab_size,
                'is_fitted': self.is_fitted
            }, f)
        print(f"Tokenizer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the tokenizer from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.vocab_size = data['vocab_size']
        self.is_fitted = data['is_fitted']
        print(f"Tokenizer loaded from {filepath}")

def create_tokenizer_from_data(expressions: List[str], solutions: List[str], vocab_size: int = 1000) -> MathTokenizer:
    """Create and fit a tokenizer from expression and solution data."""
    tokenizer = MathTokenizer(vocab_size=vocab_size)
    
    # Combine all texts for vocabulary building
    all_texts = expressions + solutions
    tokenizer.fit(all_texts)
    
    return tokenizer

if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = MathTokenizer(vocab_size=100)
    
    # Test texts
    test_texts = [
        "2x + 3 = 9",
        "Step 1: Subtract 3 from both sides → 2x = 6",
        "Step 2: Divide both sides by 2 → x = 3",
        "(2x + 1)(3x + 2)",
        "x^2 + 5x + 6"
    ]
    
    # Fit the tokenizer
    tokenizer.fit(test_texts)
    
    # Test encoding and decoding
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"Original: {text}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
        print("-" * 50) 