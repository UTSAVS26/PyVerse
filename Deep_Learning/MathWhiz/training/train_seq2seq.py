import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import argparse
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generator import MathExpressionGenerator
from model.tokenizer import MathTokenizer, create_tokenizer_from_data
from model.seq2seq_lstm import Seq2SeqLSTM
from utils.math_solver import MathSolver

class MathDataset(Dataset):
    """Dataset class for math expressions and solutions."""
    
    def __init__(self, expressions: list, solutions: list, tokenizer: MathTokenizer, max_length: int = 50):
        self.expressions = expressions
        self.solutions = solutions
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.expressions)
    
    def __getitem__(self, idx):
        expression = self.expressions[idx]
        solution = self.solutions[idx]
        
        # Tokenize
        expr_tokens = self.tokenizer.encode(expression)
        sol_tokens = self.tokenizer.encode(solution)
        
        # Pad sequences
        expr_tokens = self._pad_sequence(expr_tokens, self.max_length)
        sol_tokens = self._pad_sequence(sol_tokens, self.max_length)
        
        return {
            'expression': torch.tensor(expr_tokens, dtype=torch.long),
            'solution': torch.tensor(sol_tokens, dtype=torch.long),
            'expr_length': len(self.tokenizer.encode(expression)),
            'sol_length': len(self.tokenizer.encode(solution))
        }
    
    def _pad_sequence(self, sequence: list, max_length: int) -> list:
        """Pad sequence to max_length."""
        if len(sequence) >= max_length:
            return sequence[:max_length]
        else:
            return sequence + [0] * (max_length - len(sequence))

def collate_fn(batch):
    """Custom collate function for DataLoader."""
    expressions = torch.stack([item['expression'] for item in batch])
    solutions = torch.stack([item['solution'] for item in batch])
    expr_lengths = [item['expr_length'] for item in batch]
    sol_lengths = [item['sol_length'] for item in batch]
    
    return {
        'expressions': expressions,
        'solutions': solutions,
        'expr_lengths': expr_lengths,
        'sol_lengths': sol_lengths
    }

def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device, teacher_forcing_ratio: float = 0.5):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        expressions = batch['expressions'].to(device)
        solutions = batch['solutions'].to(device)
        expr_lengths = batch['expr_lengths']
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(expressions, expr_lengths, solutions, teacher_forcing_ratio)
        
        # Calculate loss
        batch_size, seq_len, vocab_size = outputs.shape
        outputs = outputs.view(-1, vocab_size)
        targets = solutions.view(-1)
        
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches

def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                  device: torch.device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for batch in progress_bar:
            expressions = batch['expressions'].to(device)
            solutions = batch['solutions'].to(device)
            expr_lengths = batch['expr_lengths']
            
            # Forward pass
            outputs = model(expressions, expr_lengths, solutions, teacher_forcing_ratio=0.0)
            
            # Calculate loss
            batch_size, seq_len, vocab_size = outputs.shape
            outputs = outputs.view(-1, vocab_size)
            targets = solutions.view(-1)
            
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches

def save_model(model: nn.Module, tokenizer: MathTokenizer, optimizer: optim.Optimizer, 
               epoch: int, loss: float, filepath: str):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'tokenizer': tokenizer
    }, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath: str, model: nn.Module, optimizer: optim.Optimizer):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    tokenizer = checkpoint['tokenizer']
    return model, optimizer, epoch, loss, tokenizer

def plot_training_history(train_losses: list, val_losses: list, save_path: str = None):
    """Plot training and validation loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training plot saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train Seq2Seq LSTM model for math expressions')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size of LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--vocab_size', type=int, default=1000, help='Vocabulary size')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('--data_size', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to checkpoint to load')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate or load data
    print("Generating/loading dataset...")
    generator = MathExpressionGenerator()
    
    # Check if dataset exists
    dataset_path = "data/dataset.csv"
    if os.path.exists(dataset_path):
        expressions, solutions = generator.load_dataset()
        print(f"Loaded {len(expressions)} samples from existing dataset")
    else:
        expressions, solutions = generator.generate_dataset(args.data_size)
        generator.save_dataset(expressions, solutions)
        print(f"Generated {len(expressions)} new samples")
    
    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = create_tokenizer_from_data(expressions, solutions, args.vocab_size)
    
    # Split data
    train_size = int(0.8 * len(expressions))
    val_size = len(expressions) - train_size
    
    train_expressions = expressions[:train_size]
    train_solutions = solutions[:train_size]
    val_expressions = expressions[train_size:]
    val_solutions = solutions[train_size:]
    
    # Create datasets
    train_dataset = MathDataset(train_expressions, train_solutions, tokenizer, args.max_length)
    val_dataset = MathDataset(val_expressions, val_solutions, tokenizer, args.max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    model = Seq2SeqLSTM(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # Create optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        model, optimizer, start_epoch, best_loss, tokenizer = load_model(args.load_checkpoint, model, optimizer)
        print(f"Loaded checkpoint from epoch {start_epoch} with loss {best_loss}")
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, tokenizer, optimizer, epoch, val_loss, 
                      os.path.join(args.save_dir, 'best_model.pth'))
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_model(model, tokenizer, optimizer, epoch, val_loss,
                      os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pth'))
    
    # Save final model
    save_model(model, tokenizer, optimizer, args.epochs, val_loss,
              os.path.join(args.save_dir, 'final_model.pth'))
    
    # Plot training history
    plot_training_history(train_losses, val_losses, 
                         os.path.join(args.save_dir, 'training_history.png'))
    
    print("Training completed!")

if __name__ == "__main__":
    main() 