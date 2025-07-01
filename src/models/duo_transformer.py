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


class TransformerDualOutputIVModel(IVModelBase):
    """
    Transformer-based model for generating (current, voltage) pairs of IV curves.
    It takes physical parameters and generates sequences autoregressively.
    This version assumes a fixed output length determined by max_sequence_length
    and does not use an explicit EOS token for simplicity.
    """

    def __init__(
        self,
        physical_dim,
        d_model,
        nhead,
        num_decoder_layers,
        max_sequence_length,  # This will be num_pre + num_post - 1 from preprocessing
        dropout=0.1,
        pad_value=-1.0,  # Used for padding mask in training if needed
        **kwargs,
    ):
        super(TransformerDualOutputIVModel, self).__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.pad_value = pad_value  # For padding mask creation if necessary

        # Embeddings and encodings
        self.physical_embedding = nn.Sequential(
            nn.Linear(physical_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        # Input to value_embedding is a [current, voltage] pair
        self.pair_embedding = nn.Linear(2, d_model)
        self.pos_encoder = PositionalEncoding(
            d_model, max_len=max_sequence_length + 1
        )  # +1 for start token
        self.dropout = nn.Dropout(dropout)
        self.decoder_norm = nn.LayerNorm(d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward=d_model * 2,  # Common choice
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_decoder_layers
        )
        # Output projection to [current, voltage]
        self.output_proj = nn.Linear(d_model, 2)

        # Learnable start token representing an initial [current, voltage] pair
        # Initialized with small random values, model will learn appropriate start
        self.start_pair_values = nn.Parameter(
            torch.randn(1, 1, 2) * 0.01
        )  # Shape: (1, 1, 2)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(
            torch.ones(sz, sz, dtype=torch.bool, device=self.start_pair_values.device),
            diagonal=1,
        )

    def forward(self, physical, target_pairs=None, target_mask=None):
        """
        Forward pass for training and inference.
        Args:
            physical (Tensor): Physical input parameters, shape (batch_size, physical_dim).
            target_pairs (Tensor, optional): Target (current, voltage) pairs for training.
                                            Shape (batch_size, seq_len, 2).
            target_mask (Tensor, optional): Boolean mask for target_pairs during training.
                                           True for valid positions. Shape (batch_size, seq_len).
                                           If None, assumes all target_pairs are valid.
        """
        batch_size = physical.size(0)
        device = physical.device

        # Embed physical parameters and tile across time
        # The sequence length for memory will match max_sequence_length
        memory = (
            self.physical_embedding(physical)
            .unsqueeze(1)
            .expand(batch_size, self.max_sequence_length, self.d_model)
        )

        if target_pairs is not None:
            # --- Training Mode (Teacher Forcing) ---
            seq_len = target_pairs.size(1)
            if seq_len != self.max_sequence_length:
                # This should ideally not happen if data is preprocessed to fixed length
                # For safety, one might truncate or pad, but better to ensure data matches
                raise ValueError(
                    f"Training target_pairs seq_len {seq_len} "
                    f"does not match model.max_sequence_length {self.max_sequence_length}"
                )

            # Prepend start token values to target pairs for decoder input
            # start_pair_values is (1,1,2), expand to (batch_size, 1, 2)
            start_values_expanded = self.start_pair_values.expand(batch_size, -1, -1)
            # Decoder input is shifted target: start_token + target_pairs[:, :-1]
            decoder_input_pairs = torch.cat(
                [start_values_expanded, target_pairs[:, :-1, :]], dim=1
            )  # Shape: (batch_size, seq_len, 2)

            # Embed the [current, voltage] pairs
            decoder_input_embedded = self.pair_embedding(decoder_input_pairs)
            decoder_input_embedded = self.pos_encoder(decoder_input_embedded)
            decoder_input_embedded = self.dropout(decoder_input_embedded)

            # Create causal attention mask
            # Size is seq_len because decoder_input_embedded has that length
            tgt_mask_causal = self.generate_square_subsequent_mask(seq_len).to(device)

            # Create padding mask for decoder input if target_mask is provided
            # The padding mask for TransformerDecoder needs to be True for padded positions
            # target_mask is True for valid positions. So we need to invert it.
            # Also, the start token is always valid.
            if target_mask is not None:
                # Shift target_mask for decoder input (like target_pairs)
                # start token position is always unmasked (False)
                start_token_mask = torch.zeros(
                    batch_size, 1, dtype=torch.bool, device=device
                )
                # Invert target_mask (True for pad) and take up to seq_len-1
                shifted_target_key_padding_mask = ~target_mask[:, :-1]
                tgt_key_padding_mask = torch.cat(
                    [start_token_mask, shifted_target_key_padding_mask], dim=1
                )
            else:
                tgt_key_padding_mask = None  # No padding if no mask provided

            decoder_output = self.transformer_decoder(
                tgt=decoder_input_embedded,
                memory=memory,  # Memory spans max_sequence_length
                tgt_mask=tgt_mask_causal,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=None,  # Assuming physical params are always valid
            )
            decoder_output = self.decoder_norm(decoder_output)

            # Project to output [current, voltage] pairs
            # This output corresponds to predictions for target_pairs (original, not shifted)
            output_pairs = self.output_proj(
                decoder_output
            )  # Shape: (batch_size, seq_len, 2)
            return output_pairs

        else:
            # --- Inference Mode (Autoregressive Generation) ---
            # Start with the learnable start_pair_values, embedded
            current_input_embedded = self.pair_embedding(
                self.start_pair_values.expand(
                    batch_size, -1, -1
                )  # (B, 1, 2) -> (B, 1, D_MODEL)
            )
            generated_sequence_embedded = (
                current_input_embedded  # Shape: (B, 1, D_MODEL)
            )

            predicted_pairs_list = []

            for t in range(self.max_sequence_length):
                pos_encoded_input = self.pos_encoder(generated_sequence_embedded)

                # Causal mask for the current generated sequence length
                current_seq_len = pos_encoded_input.size(1)
                tgt_mask_causal = self.generate_square_subsequent_mask(
                    current_seq_len
                ).to(device)

                # Memory for this step should be sliced if its length matters,
                # but TransformerDecoder handles variable memory length.
                # Here, memory is already expanded to max_sequence_length.
                decoder_output_step = self.transformer_decoder(
                    tgt=pos_encoded_input,
                    memory=memory,  # Full memory used for cross-attention
                    tgt_mask=tgt_mask_causal,
                )
                decoder_output_step = self.decoder_norm(decoder_output_step)

                # Prediction using the last token's hidden state
                last_token_hidden_state = decoder_output_step[
                    :, -1:, :
                ]  # Shape: (B, 1, D_MODEL)
                next_pair_pred = self.output_proj(
                    last_token_hidden_state
                )  # Shape: (B, 1, 2)
                predicted_pairs_list.append(next_pair_pred.squeeze(1))  # Store (B, 2)

                if t < self.max_sequence_length - 1:
                    # Prepare input for the next step
                    next_input_embedded = self.pair_embedding(
                        next_pair_pred
                    )  # (B, 1, 2) -> (B, 1, D_MODEL)
                    generated_sequence_embedded = torch.cat(
                        [generated_sequence_embedded, next_input_embedded], dim=1
                    )

            # Stack all predicted pairs
            output_pairs = torch.stack(
                predicted_pairs_list, dim=1
            )  # Shape: (B, max_sequence_length, 2)
            return output_pairs

    def generate_curve_batch(self, physical_input, scalers, device):
        """
        Generate (current, voltage) IV curves in batch.
        Scalers should be a tuple (current_scaler, voltage_scaler, input_scaler)
        or similar, where current_scaler and voltage_scaler are for output.
        """
        self.eval()
        if len(scalers) < 2:
            raise ValueError(
                "generate_curve_batch expects at least current_scaler and voltage_scaler in scalers"
            )
        current_scaler, voltage_scaler = (
            scalers[0],
            scalers[2],
        )  # Assuming order from preprocess_fixed_length_dual_output

        with torch.no_grad():
            # Generate sequence of [current, voltage] pairs
            # Model forward in inference mode (target_pairs=None)
            predicted_pairs_scaled = self.forward(
                physical_input.to(device), target_pairs=None
            )
            # predicted_pairs_scaled shape: (batch_size, max_sequence_length, 2)

            pred_currents_scaled = predicted_pairs_scaled[..., 0].cpu().numpy()
            pred_voltages_scaled = predicted_pairs_scaled[..., 1].cpu().numpy()

            batch_size = physical_input.size(0)
            gen_current_curves = []
            gen_voltage_curves = []

            for i in range(batch_size):
                # Inverse transform current
                # Scaler expects (n_samples, n_features)
                unscaled_current = current_scaler.inverse_transform(
                    pred_currents_scaled[i]
                ).flatten()
                gen_current_curves.append(unscaled_current)

                # Inverse transform voltage
                unscaled_voltage = voltage_scaler.inverse_transform(
                    pred_voltages_scaled[i]
                ).flatten()
                gen_voltage_curves.append(unscaled_voltage)

        # Returns lists of numpy arrays
        return gen_current_curves, gen_voltage_curves

    # save_model and load_model would need to be adapted if parameters like
    # self.voltage_points are removed or start_pair_values is added.
    # For now, keeping them structurally similar to the original.
    def save_model(self, save_path, scalers, params):
        save_dict = {
            "model_state_dict": self.state_dict(),
            "scalers": scalers,  # (current_scaler, input_scaler, voltage_scaler)
            "params": params,  # Should include physical_dim, d_model, etc.
            # and max_sequence_length for this model.
        }
        torch.save(save_dict, save_path)

    @classmethod
    def load_model(cls, model_path, device):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        params = checkpoint["params"]
        # Ensure max_sequence_length is present in saved params for this model
        if "max_sequence_length" not in params:
            # Try to infer from a common output_dim if saved that way, or error
            if "output_dim" in params:  # output_dim from MLP was num_pre + num_post - 1
                params["max_sequence_length"] = params["output_dim"]
            else:
                raise KeyError(
                    "max_sequence_length or output_dim not found in saved model parameters."
                )

        model = cls(
            physical_dim=params["physical_dim"],
            d_model=params["d_model"],
            nhead=params["nhead"],
            num_decoder_layers=params["num_decoder_layers"],
            max_sequence_length=params["max_sequence_length"],
            dropout=params.get("dropout", 0.1),
            pad_value=params.get("pad_value", -1.0),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        # Scalers now would be (current_scaler, input_scaler, voltage_scaler)
        return model, checkpoint["scalers"], params["max_sequence_length"]
