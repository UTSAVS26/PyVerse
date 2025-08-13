import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import random

class Encoder(nn.Module):
    """LSTM encoder for mathematical expressions."""
    
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_seq: torch.Tensor, input_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the encoder.
        
        Args:
            input_seq: Input sequence tensor [batch_size, seq_len]
            input_lengths: List of sequence lengths
            
        Returns:
            outputs: Encoder outputs [batch_size, seq_len, hidden_size * 2]
            (hidden, cell): Final hidden and cell states
        """
        # Convert input lengths to tensor
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)
        
        # Embedding
        embedded = self.dropout(self.embedding(input_seq))  # [batch_size, seq_len, hidden_size]
        
        # Pack sequence
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        
        # LSTM forward pass
        outputs, (hidden, cell) = self.lstm(packed)
        
        # Unpack sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # Sum bidirectional outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        
        return outputs, (hidden, cell)

class Attention(nn.Module):
    """Attention mechanism for the decoder."""
    
    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of attention mechanism.
        
        Args:
            hidden: Decoder hidden state [batch_size, hidden_size]
            encoder_outputs: Encoder outputs [batch_size, seq_len, hidden_size]
            
        Returns:
            attention_weights: Attention weights [batch_size, seq_len]
            context: Context vector [batch_size, hidden_size]
        """
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # Repeat hidden state seq_len times
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Calculate attention scores
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention_weights = F.softmax(self.v(energy).squeeze(2), dim=1)
        
        # Calculate context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return attention_weights, context

class Decoder(nn.Module):
    """LSTM decoder with attention for generating solutions."""
    
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(
            hidden_size + hidden_size,  # input_size = embedding_size + context_size
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.out = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_seq: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor], 
                encoder_outputs: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Forward pass of the decoder.
        
        Args:
            input_seq: Target sequence [batch_size, seq_len]
            hidden: Initial hidden state from encoder
            encoder_outputs: Encoder outputs
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            outputs: Decoder outputs [batch_size, seq_len, vocab_size]
        """
        batch_size = input_seq.size(0)
        seq_len = input_seq.size(1)
        
        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(input_seq.device)
        
        # Get first input (SOS token)
        input_token = input_seq[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        for t in range(seq_len):
            # Embedding
            embedded = self.dropout(self.embedding(input_token))  # [batch_size, 1, hidden_size]
            
            # Attention
            attention_weights, context = self.attention(hidden[0][-1], encoder_outputs)
            
            # Combine embedded input and context
            lstm_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
            
            # LSTM forward pass
            output, hidden = self.lstm(lstm_input, hidden)
            
            # Output layer
            output = self.out(torch.cat((output, context.unsqueeze(1)), dim=2))
            outputs[:, t, :] = output.squeeze(1)
            
            # Teacher forcing
            if random.random() < teacher_forcing_ratio and t < seq_len - 1:
                input_token = input_seq[:, t + 1].unsqueeze(1)
            else:
                # Use predicted token
                input_token = output.argmax(2)
        
        return outputs

class Seq2SeqLSTM(nn.Module):
    """Complete Seq2Seq model with LSTM encoder-decoder and attention."""
    
    def __init__(self, vocab_size: int, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super(Seq2SeqLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.encoder = Encoder(vocab_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(vocab_size, hidden_size, num_layers, dropout)
        
    def forward(self, src: torch.Tensor, src_lengths: List[int], tgt: torch.Tensor, 
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Forward pass of the complete Seq2Seq model.
        
        Args:
            src: Source sequence [batch_size, src_len]
            src_lengths: Source sequence lengths
            tgt: Target sequence [batch_size, tgt_len]
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            outputs: Decoder outputs [batch_size, tgt_len, vocab_size]
        """
        # Encoder forward pass
        encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)
        
        # Prepare decoder hidden state
        decoder_hidden = (
            encoder_hidden[0][:self.num_layers],  # Take first num_layers from bidirectional
            encoder_hidden[1][:self.num_layers]
        )
        
        # Decoder forward pass
        outputs = self.decoder(tgt, decoder_hidden, encoder_outputs, teacher_forcing_ratio)
        
        return outputs
    
    def predict(self, src: torch.Tensor, src_lengths: List[int], max_length: int = 50, 
                sos_token: int = 2, eos_token: int = 3) -> torch.Tensor:
        """
        Generate predictions for inference.
        
        Args:
            src: Source sequence [batch_size, src_len]
            src_lengths: Source sequence lengths
            max_length: Maximum length of generated sequence
            sos_token: Start of sequence token
            eos_token: End of sequence token
            
        Returns:
            predictions: Generated sequences [batch_size, max_length]
        """
        batch_size = src.size(0)
        
        # Encoder forward pass
        encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)
        
        # Prepare decoder hidden state
        decoder_hidden = (
            encoder_hidden[0][:self.num_layers],
            encoder_hidden[1][:self.num_layers]
        )
        
        # Initialize predictions
        predictions = torch.full((batch_size, max_length), eos_token, dtype=torch.long, device=src.device)
        predictions[:, 0] = sos_token
        
        # Generate sequence
        for t in range(max_length - 1):
            # Get current input
            input_token = predictions[:, t].unsqueeze(1)
            
            # Embedding
            embedded = self.decoder.embedding(input_token)
            
            # Attention
            attention_weights, context = self.decoder.attention(decoder_hidden[0][-1], encoder_outputs)
            
            # Combine embedded input and context
            lstm_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
            
            # LSTM forward pass
            output, decoder_hidden = self.decoder.lstm(lstm_input, decoder_hidden)
            
            # Output layer
            output = self.decoder.out(torch.cat((output, context.unsqueeze(1)), dim=2))
            
            # Get next token
            next_token = output.argmax(2)
            predictions[:, t + 1] = next_token.squeeze(1)
            
            # Stop if all sequences have EOS token
            if (predictions[:, t + 1] == eos_token).all():
                break
        
        return predictions

def create_model(vocab_size: int, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.1) -> Seq2SeqLSTM:
    """Create a Seq2Seq LSTM model."""
    return Seq2SeqLSTM(vocab_size, hidden_size, num_layers, dropout)

if __name__ == "__main__":
    # Test the model
    vocab_size = 1000
    hidden_size = 256
    batch_size = 4
    src_len = 10
    tgt_len = 15
    
    # Create model
    model = Seq2SeqLSTM(vocab_size, hidden_size)
    
    # Create dummy data
    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))
    src_lengths = [src_len] * batch_size
    
    # Forward pass
    outputs = model(src, src_lengths, tgt)
    print(f"Output shape: {outputs.shape}")
    
    # Test prediction
    predictions = model.predict(src, src_lengths)
    print(f"Prediction shape: {predictions.shape}")
    
    print("Model test completed successfully!") 