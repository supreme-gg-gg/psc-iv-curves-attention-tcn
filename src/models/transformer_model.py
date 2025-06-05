import torch
import torch.nn as nn
import numpy as np
import math
from src.models.seq_model_base import SeqModelBase

class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings.
    This version is adapted to work with batch_first=True inputs.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]

class TransformerIVModel(SeqModelBase):
    def __init__(self, physical_dim, d_model, nhead, num_decoder_layers, dropout=0.1, max_sequence_length=100, **kwargs):
        super(TransformerIVModel, self).__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length

        # Embeddings and encodings
        self.physical_embedding = nn.Sequential(
            nn.Linear(physical_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.value_embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_sequence_length)
        self.dropout = nn.Dropout(dropout)
        self.decoder_norm = nn.LayerNorm(d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=d_model * 2, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.output_proj = nn.Linear(d_model, 1)
        self.eos_head = nn.Linear(d_model, 1)  # EOS prediction head
        self.start_token = nn.Parameter(torch.zeros(1))

    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for decoder's self-attention."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, physical, target_seq=None, lengths=None, teacher_forcing_ratio=None):
        """
        Forward pass that handles both training and inference modes.
        During training: Process entire sequence in parallel using teacher forcing.
        During inference: Generate autoregressively until EOS token or max length.
        NOTE: Teacher forcing ratio is not used here because we only use ground truth sequences.
        """
        batch_size = physical.size(0)
        device = physical.device
        memory = self.physical_embedding(physical).unsqueeze(1)

        if target_seq is not None:
            # Training mode - process entire sequence in parallel
            # Prepare decoder input sequence (shift target right by 1 and prepend start token)
            start_tokens = self.start_token.expand(batch_size, 1)
            decoder_input = torch.cat([start_tokens, target_seq[:, :-1]], dim=1)
            
            # Embed and add positional encoding
            decoder_input = self.value_embedding(decoder_input.unsqueeze(-1))
            decoder_input = self.pos_encoder(decoder_input)
            decoder_input = self.dropout(decoder_input)
            
            # Create causal attention mask
            tgt_mask = self.generate_square_subsequent_mask(decoder_input.size(1)).to(device)
            
            # Forward pass through transformer
            decoder_output = self.transformer_decoder(
                tgt=decoder_input,
                memory=memory,
                tgt_mask=tgt_mask
            )
            decoder_output = self.decoder_norm(decoder_output)
            
            # Project to output space
            outputs = self.output_proj(decoder_output).squeeze(-1)
            # Get EOS logits
            eos_logits = self.eos_head(decoder_output).squeeze(-1)

            return outputs, eos_logits
        
        else:
            # Inference mode - generate autoregressively with EOS prediction
            max_len = self.max_sequence_length if lengths is None else max(lengths)
            device = physical.device
            
            # Start with start token
            # Initialize with start token [batch, 1, 1]
            current_token = self.start_token.expand(batch_size, 1, 1).to(device)
            value_outputs = []
            eos_outputs = []
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            for t in range(max_len):
                # current_token already [batch, seq, 1]
                decoder_input = self.value_embedding(current_token)
                decoder_input = self.pos_encoder(decoder_input)
                decoder_input = self.dropout(decoder_input)
                
                # Generate next token
                decoder_output = self.transformer_decoder(
                    tgt=decoder_input,
                    memory=memory,
                    tgt_mask=None  # No mask needed during inference as we only attend to past tokens
                )
                decoder_output = self.decoder_norm(decoder_output)
                
                # Get prediction and EOS probability
                next_token = self.output_proj(decoder_output[:, -1:]).squeeze(-1)
                next_eos = self.eos_head(decoder_output[:, -1:]).squeeze(-1)
                
                value_outputs.append(next_token)
                eos_outputs.append(next_eos)
                
                # Update finished flags based on EOS prediction
                eos_prob = torch.sigmoid(next_eos)
                finished = finished | (eos_prob > 0.5)
                # note that we require ALL of the sequences to finish before we stop
                # so that it works properly with batch generation
                if finished.all():
                    break
                
                # Update input sequence for next iteration
                # next_token shape: [batch, 1] -> [batch, 1, 1]
                current_token = torch.cat([current_token, next_token.unsqueeze(-1)], dim=1)
            
            # Stack outputs
            seq_preds = torch.stack(value_outputs, dim=1)
            eos_logits = torch.stack(eos_outputs, dim=1)

            return seq_preds, eos_logits

    def generate_curve_batch(self, physical_input, scalers, device):
        """
        Generate IV curves in batch using auto-regressive generation.
        """
        self.eval()
        _, output_scaler = scalers
        batch_size = physical_input.size(0)
        
        with torch.no_grad():
            # Generate sequence and get EOS predictions
            seq_outputs, eos_logits = self.forward(physical_input, target_seq=None)
            eos_probs = torch.sigmoid(eos_logits)
            
            # Process outputs and determine lengths using EOS
            outputs_cpu = seq_outputs.detach().cpu().numpy()
            eos_cpu = eos_probs.detach().cpu().numpy()
            gen_curves = []
            lengths = []
            
            # this loop will determine the actual length and crop each sequence in the batch
            # because during generation we generated until the longest sequence finishes
            for i in range(batch_size):
                seq = outputs_cpu[i]
                eos = eos_cpu[i]
                # Find first position where EOS probability > 0.5
                eos_positions = np.where(eos > 0.5)[0]
                length = int(eos_positions[0]) + 1 if len(eos_positions) > 0 else len(seq)
                lengths.append(length)
                
                # Convert to actual values
                unscaled = output_scaler.inverse_transform(seq[:length].reshape(-1, 1)).flatten()
                gen_curves.append(unscaled)
            
            return gen_curves, lengths

    def save_model(self, save_path, scalers, params):
        """Save model state, scalers, and parameters."""
        save_dict = {
            "model_state_dict": self.state_dict(),
            "scalers": scalers,
            "params": params
        }
        if self.max_sequence_length is not None:
            save_dict["max_sequence_length"] = self.max_sequence_length
        torch.save(save_dict, save_path)

    @classmethod
    def load_model(cls, model_path, device):
        """Load model from file."""
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        params = checkpoint["params"]
        max_sequence_length = checkpoint.get("max_sequence_length", None)
        model = cls(
            physical_dim=params["physical_dim"],
            d_model=params["d_model"],
            nhead=params["nhead"],
            num_decoder_layers=params["num_decoder_layers"],
            dropout=params["dropout"],
            max_sequence_length=max_sequence_length if max_sequence_length is not None else 100,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, checkpoint["scalers"], max_sequence_length
