import torch
import torch.nn as nn
import math

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
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model) for batch_first
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        # Add positional encoding to the input.
        # Ensure pe is broadcastable or matches seq_len.
        return x + self.pe[:, :x.size(1), :]

class TransformerIVModel(nn.Module):
    def __init__(self, physical_dim, d_model, nhead, num_decoder_layers, dropout=0.1, max_seq_len=100):
        super(TransformerIVModel, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # 1. Embed physical parameters to d_model dimension. This acts as the 'memory' for the decoder.
        self.physical_embedding = nn.Linear(physical_dim, d_model)

        # 2. Embedding for the input sequence values (current values)
        # We assume input values are scalars, so we embed them to d_model
        self.value_embedding = nn.Linear(1, d_model)

        # 3. Positional Encoding for the generated sequence
        # Max_len for PE should be at least max_seq_len from data + 1 (for EOS)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)

        # 4. Transformer Decoder Layer
        # Set batch_first=True here, and ensure inputs to TransformerDecoder are also batch_first
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # 5. Output projection: predicts value and EOS logit
        self.output_proj = nn.Linear(d_model, 2) # [value, EOS_logit]

        # 6. Learnable start token for sequence generation
        # This token will be embedded to d_model. It's a scalar value that gets embedded.
        self.start_token = nn.Parameter(torch.zeros(1))

    def generate_square_subsequent_mask(self, sz):
        """
        Generates a square mask for the sequence. Used to prevent attention to future tokens.
        This mask should be (seq_len, seq_len).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, physical, target_seq=None, lengths=None, eos_targets=None, max_gen_length=None):
        """
        Generates IV curve sequences.
        Args:
            physical (Tensor): Batch of physical parameters (batch_size, physical_dim)
            target_seq (Tensor, optional): True target sequence for teacher forcing (batch_size, max_len)
            lengths (Tensor, optional): True lengths of sequences (batch_size)
            eos_targets (Tensor, optional): Binary EOS targets (batch_size, max_len+1)
            max_gen_length (int, optional): Max length for auto-regressive generation (inference)
        Returns:
            Tuple: (value_outputs, eos_logits)
        """
        batch_size = physical.size(0)
        device = physical.device

        # Prepare memory (context) from physical parameters
        # Shape: (batch_size, physical_dim) -> (batch_size, 1, d_model)
        memory = self.physical_embedding(physical).unsqueeze(1)

        # If target_seq is provided, we use parallel computation (teacher-forced or scheduled sampling)
        if target_seq is not None:
            max_len = target_seq.size(1) # Actual max length of target data

            # Embed the start token for the batch
            start_token_embedded = self.value_embedding(self.start_token.expand(batch_size, 1)).unsqueeze(1)

            # Embed the target sequence values
            target_seq_embedded = self.value_embedding(target_seq.unsqueeze(-1))

            # Concatenate to form the full decoder input sequence
            decoder_input_seq = torch.cat([start_token_embedded, target_seq_embedded], dim=1)

            # Apply positional encoding
            decoder_input_seq_with_pos = self.pos_encoder(decoder_input_seq)

            # Create attention mask
            tgt_mask = self.generate_square_subsequent_mask(max_len + 1).to(device)

            decoder_output = self.transformer_decoder(
                tgt=decoder_input_seq_with_pos,
                memory=memory,
                tgt_mask=tgt_mask
            )

            combined_preds = self.output_proj(decoder_output)

            value_outputs = combined_preds[:, :-1, 0]
            eos_logits = combined_preds[:, :, 1]

            return value_outputs, eos_logits

        else:
            # Inference mode (auto-regressive generation)
            if max_gen_length is None:
                max_gen_length = self.max_seq_len

            value_outputs_list = []
            eos_logits_list = []

            current_decoder_input_seq = self.value_embedding(self.start_token.expand(batch_size, 1)).unsqueeze(1)

            for t in range(max_gen_length):
                tgt_mask = self.generate_square_subsequent_mask(t + 1).to(device)
                decoder_input_with_pos = self.pos_encoder(current_decoder_input_seq)

                decoder_output = self.transformer_decoder(
                    tgt=decoder_input_with_pos,
                    memory=memory,
                    tgt_mask=tgt_mask
                )

                last_token_output = decoder_output[:, -1, :]
                combined_preds = self.output_proj(last_token_output)

                value_pred = combined_preds[:, 0].unsqueeze(1)
                eos_logit = combined_preds[:, 1].unsqueeze(1)

                value_outputs_list.append(value_pred)
                eos_logits_list.append(eos_logit)

                if batch_size == 1 and torch.sigmoid(eos_logit[0]).item() > 0.5:
                    break

                next_input_embedding = self.value_embedding(value_pred).unsqueeze(1)
                current_decoder_input_seq = torch.cat([current_decoder_input_seq, next_input_embedding], dim=1)

            value_outputs = torch.cat(value_outputs_list, dim=1)
            eos_logits = torch.cat(eos_logits_list, dim=1)

            return value_outputs, eos_logits
        
def load_trained_transformer_model(model_path, device):
    """
    Load a trained Transformer model with its scalers and parameters.

    Args:
        model_path: Path to the saved model file
        device: Device to load the model onto

    Returns:
        model: Loaded model
        scalers: Tuple of (input_scaler, output_scaler)
        max_sequence_length: Max sequence length used during training/generation
    """
    checkpoint = torch.load(model_path, map_location=device)

    # Create model with saved parameters
    params = checkpoint['params']
    model = TransformerIVModel(
        physical_dim=params['physical_dim'],
        d_model=params['d_model'],
        nhead=params['nhead'],
        num_decoder_layers=params['num_decoder_layers'],
        dropout=params['dropout'],
        max_seq_len=checkpoint['max_sequence_length'] # Pass max_seq_len
    ).to(device)

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint['scalers'], checkpoint['max_sequence_length'] # Return max_sequence_length