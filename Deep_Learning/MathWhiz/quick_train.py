#!/usr/bin/env python3
"""
Quick Training Demo for MathWhiz
Trains the model on a small dataset to demonstrate the training process.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from data.generator import MathExpressionGenerator
from model.tokenizer import create_tokenizer_from_data
from model.seq2seq_lstm import Seq2SeqLSTM
from training.train_seq2seq import MathDataset, collate_fn

def quick_train():
    print("🚀 Quick Training Demo for MathWhiz")
    print("=" * 50)
    
    # Generate small dataset
    print("1. Generating dataset...")
    generator = MathExpressionGenerator()
    expressions, solutions = generator.generate_dataset(100)
    
    # Create tokenizer
    print("2. Creating tokenizer...")
    tokenizer = create_tokenizer_from_data(expressions, solutions, vocab_size=200)
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Create dataset and dataloader
    print("3. Preparing data...")
    dataset = MathDataset(expressions, solutions, tokenizer, max_length=30)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    # Create model
    print("4. Creating model...")
    model = Seq2SeqLSTM(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_size=64,  # Small for quick demo
        num_layers=1,
        dropout=0.1
    )
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    print(f"5. Training on {device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Quick training loop (just a few epochs)
    model.train()
    for epoch in range(3):
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/3")
        
        for batch in progress_bar:
            expressions = batch['expressions'].to(device)
            solutions = batch['solutions'].to(device)
            expr_lengths = batch['expr_lengths']
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(expressions, expr_lengths, solutions, teacher_forcing_ratio=0.5)
            
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
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    
    print("\n✅ Quick training completed!")
    print("The model is now ready for inference.")
    
    # Test the model
    print("\n6. Testing model...")
    model.eval()
    with torch.no_grad():
        # Test with a simple expression
        test_expr = "2x + 3 = 9"
        expr_tokens = tokenizer.encode(test_expr)
        expr_tensor = torch.tensor([expr_tokens], dtype=torch.long).to(device)
        expr_lengths = [len(expr_tokens)]
        
        # Generate prediction
        pred_tokens = model.predict(expr_tensor, expr_lengths)
        pred_sequence = pred_tokens[0].cpu().numpy()
        pred_text = tokenizer.decode(pred_sequence)
        
        print(f"Input: {test_expr}")
        print(f"Prediction: {pred_text}")
    
    print("\n🎉 Demo completed successfully!")

if __name__ == "__main__":
    quick_train() 