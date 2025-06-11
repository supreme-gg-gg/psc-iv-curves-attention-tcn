import torch
import torch.nn as nn
import numpy as np
import math
from src.utils.iv_model_base import IVModelBase


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
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, : x.size(1), :]


class TransformerIVModel(IVModelBase):
    """
    Transformer-based model for IV curve generation.
    This model uses a Transformer decoder to generate IV curves from physical parameters.
    The common practice is training transformer is with full teacher forcing with no scheduled sampling so it is implemented here.
    During inference, it generates sequences autoregressively until an EOS token is predicted or max length is reached.

    NOTE: You should set decoder_mask_ratio to zero (default) unless you have a particular reason to not do so. This is an experimental feature.

    Args:
        physical_dim (int): Dimension of the physical input features.
        d_model (int): Dimension of the model (embedding size).
        nhead (int): Number of attention heads in the transformer decoder.
        num_decoder_layers (int): Number of decoder layers in the transformer.
        dropout (float): Dropout rate for regularization.
        max_sequence_length (int): Maximum length of the output sequence.
        decoder_mask_ratio (float): Fraction of decoder inputs to randomly mask during training (for denoising).
        eos_threshold (float): Threshold for EOS token prediction during inference.
        eos_temperature (float): Temperature for scaling EOS logits during inference.
    """

    def __init__(
        self,
        physical_dim,
        d_model,
        nhead,
        num_decoder_layers,
        dropout=0.1,
        max_sequence_length=50,
        decoder_mask_ratio=0.0,
        eos_threshold=0.5,
        eos_temperature=1.0,
        pad_value=-1.0,
        **kwargs,
    ):
        super(TransformerIVModel, self).__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.decoder_mask_ratio = decoder_mask_ratio
        self.eos_threshold = eos_threshold
        self.eos_temperature = eos_temperature
        self.pad_value = pad_value

        # Embeddings and encodings
        self.physical_embedding = nn.Sequential(
            nn.Linear(physical_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.value_embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_sequence_length)
        self.dropout = nn.Dropout(dropout)
        self.decoder_norm = nn.LayerNorm(d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_decoder_layers
        )
        self.output_proj = nn.Linear(d_model, 1)
        self.eos_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.start_token = nn.Parameter(torch.zeros(1))

    def generate_square_subsequent_mask(self, sz):
        """
        Generate a boolean causal mask for Transformer (True = masked).
        Shape: (sz, sz)
        """
        return torch.triu(
            torch.ones(sz, sz, dtype=torch.bool, device=self.start_token.device),
            diagonal=1,
        )

    def forward(
        self, physical, target_seq=None, lengths=None, teacher_forcing_ratio=None
    ):
        """
        Forward pass that handles both training and inference modes.
        During training: Process entire sequence in parallel using teacher forcing.
        During inference: Generate autoregressively until EOS token or max length.
        NOTE: Teacher forcing ratio is not used here because we only use ground truth sequences.
        """
        batch_size = physical.size(0)
        device = physical.device
        seq_len = (
            target_seq.size(1) if target_seq is not None else self.max_sequence_length
        )
        # embed physical parameters and tile across time
        memory = (
            self.physical_embedding(physical)
            .unsqueeze(1)
            .expand(batch_size, seq_len, self.d_model)
        )

        if target_seq is not None:
            # Training mode - process entire sequence in parallel
            # Prepare decoder input sequence (shift target right by 1 and prepend start token)
            start_tokens = self.start_token.expand(batch_size, 1)
            decoder_input = torch.cat(
                [start_tokens, target_seq[:, :-1]], dim=1
            )  # # Shape: (batch, seq_len)

            # Create padding mask for decoder input, True where we should ignore attention (at padded pos)
            # NOTE: THIS MUST BE DONE BEFORE EMBEDDING!
            pad_mask = decoder_input.eq(self.pad_value)  # Shape: (batch, seq_len)

            # Embed and add positional encoding
            decoder_input = self.value_embedding(decoder_input.unsqueeze(-1))

            # # apply random masking for denoising (BART-style)
            if self.training and self.decoder_mask_ratio > 0.0:
                # mask per timestep with probability decoder_mask_ratio
                mask_shape = decoder_input.shape[:2]  # (batch, seq)
                noise_mask = (
                    torch.rand(mask_shape, device=device) < self.decoder_mask_ratio
                )
                decoder_input = decoder_input.masked_fill(noise_mask.unsqueeze(-1), 0.0)

            # positional encoding and dropout
            decoder_input = self.pos_encoder(decoder_input)
            decoder_input = self.dropout(decoder_input)

            # Create causal attention mask
            tgt_mask = self.generate_square_subsequent_mask(decoder_input.size(1)).to(
                device
            )

            # Forward pass through transformer, both tgt_mask and pad_mask are boolean masks
            # tgt_mask is causal mask, pad_mask is padding mask
            # When we set memory to physical embedding, we perform cross attention using tgt as query and memory as both key and value
            # this is similar to how encoder-decoder attention works in transformers
            decoder_output = self.transformer_decoder(
                tgt=decoder_input,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=pad_mask,
            )
            decoder_output = self.decoder_norm(decoder_output)

            # Project to output space
            outputs = self.output_proj(decoder_output).squeeze(-1)
            # Get EOS logits
            eos_logits = self.eos_head(decoder_output).squeeze(-1)

            return outputs, eos_logits

        else:  # Inference mode
            # Start with the start token, embedded
            generated_seq_embedded = self.value_embedding(
                self.start_token.expand(batch_size, 1, 1).to(device)
            )

            value_outputs = []
            eos_outputs = []
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for t in range(self.max_sequence_length):
                # Add positional encoding to the current sequence
                pos_encoded_input = self.pos_encoder(generated_seq_embedded)

                # Create causal mask for the current sequence
                current_seq_len = pos_encoded_input.size(1)
                tgt_mask = self.generate_square_subsequent_mask(current_seq_len).to(
                    device
                )
                decoder_output = self.transformer_decoder(
                    tgt=pos_encoded_input,
                    memory=memory,
                    tgt_mask=tgt_mask,
                )
                decoder_output = self.decoder_norm(decoder_output)

                # Prediction using the last token's hidden state
                last_token_hidden_state = decoder_output[
                    :, -1:
                ]  # Shape: (batch, 1, d_model)
                next_token_val = self.output_proj(last_token_hidden_state).squeeze(
                    -1
                )  # Shape: (batch, 1)
                next_eos_logit = self.eos_head(last_token_hidden_state).squeeze(
                    -1
                )  # Shape: (batch, 1)

                # record outputs
                value_outputs.append(next_token_val)
                eos_outputs.append(next_eos_logit)

                # check EOS condition
                eos_prob = torch.sigmoid(next_eos_logit / self.eos_temperature).squeeze(
                    1
                )
                finished = finished | (eos_prob > self.eos_threshold)

                if finished.all():
                    break

                # Prepare input for the next step by appending the new prediction
                next_token_embedded = self.value_embedding(next_token_val.unsqueeze(-1))
                generated_seq_embedded = torch.cat(
                    [generated_seq_embedded, next_token_embedded], dim=1
                )

            # Stack outputs
            seq_preds = torch.stack(value_outputs, dim=1)
            eos_logits = torch.stack(eos_outputs, dim=1)

            return seq_preds, eos_logits

    def generate_curve_batch(self, physical_input, scalers, device):
        """
        Generate IV curves in batch using auto-regressive generation.
        Uses EOS token predictions only to determine sequence lengths.
        The physics-informed penalties (e.g. monotonicity, curvature, J_sc, V_oc)
        are then applied in the loss function.
        """
        self.eval()
        _, output_scaler = scalers
        batch_size = physical_input.size(0)

        with torch.no_grad():
            # Generate sequence and EOS logits (with temperature scaling)
            seq_outputs, eos_logits = self.forward(physical_input, target_seq=None)
            eos_probs = torch.sigmoid(eos_logits / self.eos_temperature)

            outputs_cpu = seq_outputs.detach().cpu().numpy()
            eos_cpu = eos_probs.detach().cpu().numpy()
            gen_curves = []
            lengths = []

            for i in range(batch_size):
                seq = outputs_cpu[i]
                eos = eos_cpu[i]

                # Determine EOS positions using the EOS threshold only
                eos_positions = np.where(eos > self.eos_threshold)[0]

                if eos_positions.size > 0:
                    length = int(eos_positions[0]) + 1
                else:
                    length = int(np.argmax(eos)) + 1

                lengths.append(length)
                unscaled = output_scaler.inverse_transform(
                    seq[:length].reshape(-1, 1)
                ).flatten()
                gen_curves.append(unscaled)

            return gen_curves, lengths

    def save_model(self, save_path, scalers, params):
        """Save model state, scalers, and parameters."""
        save_dict = {
            "model_state_dict": self.state_dict(),
            "scalers": scalers,
            "params": params,
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
            dropout=params.get("dropout", 0.1),
            max_sequence_length=(
                max_sequence_length
                if max_sequence_length is not None
                else params.get("max_sequence_length", 50)
            ),
            decoder_mask_ratio=params.get("decoder_mask_ratio", 0.0),
            eos_threshold=params.get("eos_threshold", 0.5),
            eos_temperature=params.get("eos_temperature", 1.0),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, checkpoint["scalers"], max_sequence_length
