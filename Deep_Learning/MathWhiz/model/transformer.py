import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    """Transformer model for math expression to solution translation."""
    
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, 
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6, 
                 dim_feedforward: int = 2048, dropout: float = 0.1, max_len: int = 100):
        super(TransformerModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a square mask for the sequence."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                src_padding_mask: Optional[torch.Tensor] = None,
                tgt_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the transformer model.
        
        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
            src_padding_mask: Source padding mask
            tgt_padding_mask: Target padding mask
            
        Returns:
            output: Transformer output [batch_size, tgt_len, vocab_size]
        """
        # Embeddings
        src_embedded = self.embedding(src) * math.sqrt(self.d_model)
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.d_model)
        
        # Positional encoding
        src_embedded = self.pos_encoder(src_embedded.transpose(0, 1)).transpose(0, 1)
        tgt_embedded = self.pos_encoder(tgt_embedded.transpose(0, 1)).transpose(0, 1)
        
        # Create target mask for causal attention
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Transformer forward pass
        output = self.transformer(
            src_embedded, 
            tgt_embedded,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
            tgt_mask=tgt_mask
        )
        
        # Output layer
        output = self.output_layer(output)
        
        return output
    
    def predict(self, src: torch.Tensor, src_padding_mask: Optional[torch.Tensor] = None,
                max_length: int = 50, sos_token: int = 2, eos_token: int = 3) -> torch.Tensor:
        """
        Generate predictions for inference.
        
        Args:
            src: Source sequence [batch_size, src_len]
            src_padding_mask: Source padding mask
            max_length: Maximum length of generated sequence
            sos_token: Start of sequence token
            eos_token: End of sequence token
            
        Returns:
            predictions: Generated sequences [batch_size, max_length]
        """
        batch_size = src.size(0)
        device = src.device
        
        # Initialize predictions
        predictions = torch.full((batch_size, max_length), eos_token, dtype=torch.long, device=device)
        predictions[:, 0] = sos_token
        
        # Source embeddings
        src_embedded = self.embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.pos_encoder(src_embedded.transpose(0, 1)).transpose(0, 1)
        
        # Generate sequence
        for t in range(max_length - 1):
            # Get current target sequence
            tgt = predictions[:, :t + 1]
            
            # Target embeddings
            tgt_embedded = self.embedding(tgt) * math.sqrt(self.d_model)
            tgt_embedded = self.pos_encoder(tgt_embedded.transpose(0, 1)).transpose(0, 1)
            
            # Create target mask
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            # Transformer forward pass
            with torch.no_grad():
                output = self.transformer(
                    src_embedded,
                    tgt_embedded,
                    src_key_padding_mask=src_padding_mask,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_padding_mask
                )
                
                # Get next token
                next_token_logits = self.output_layer(output[:, -1, :])
                next_token = next_token_logits.argmax(dim=-1)
                predictions[:, t + 1] = next_token
            
            # Stop if all sequences have EOS token
            if (predictions[:, t + 1] == eos_token).all():
                break
        
        return predictions

def create_transformer_model(vocab_size: int, d_model: int = 512, nhead: int = 8,
                           num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                           dim_feedforward: int = 2048, dropout: float = 0.1) -> TransformerModel:
    """Create a Transformer model."""
    return TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )

if __name__ == "__main__":
    # Test the transformer model
    vocab_size = 1000
    d_model = 512
    batch_size = 4
    src_len = 10
    tgt_len = 15
    
    # Create model
    model = TransformerModel(vocab_size, d_model)
    
    # Create dummy data
    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))
    
    # Forward pass
    outputs = model(src, tgt)
    print(f"Output shape: {outputs.shape}")
    
    # Test prediction
    predictions = model.predict(src)
    print(f"Prediction shape: {predictions.shape}")
    
    print("Transformer model test completed successfully!") 