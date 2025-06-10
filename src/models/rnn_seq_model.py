import torch
import torch.nn as nn
from src.utils.iv_model_base import IVModelBase 
import random
import numpy as np

class RNNIVModel(IVModelBase):
    """
    RNN-based IV model for sequence prediction with enhanced physical feature encoding.
    Implements a bidirectional LSTM with layer normalization and residual connections.
    Supports both training with teacher forcing and inference with auto-regressive generation.
    Uses a learned start token for initialization and predicts end-of-sequence (EOS) tokens.
    Supports batch generation of IV curves with EOS detection.

    Args:
        physical_dim (int): Dimension of physical input features.
        hidden_dim (int): Hidden dimension for LSTM and MLP layers.
        num_layers (int): Number of LSTM layers.
        dropout (float): Dropout rate for regularization.
        max_sequence_length (int): Maximum length of generated sequences.
        eos_threshold (float): Threshold for end-of-sequence (EOS) prediction.
        **kwargs: Additional keyword arguments for future extensibility.
    """

    def __init__(self, physical_dim, 
                 hidden_dim, 
                 voltage_points=None,
                 num_layers=2, 
                 dropout=0.2, 
                 max_sequence_length=100, 
                 eos_threshold=0.5, 
                 **kwargs):
        super(RNNIVModel, self).__init__()
        self.physical_dim = physical_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length
        # threshold for EOS prediction
        self.eos_threshold = eos_threshold

        # voltage points for IV curve generation
        self.voltage_points = voltage_points if voltage_points is not None else np.concatenate((np.arange(0, 0.41, 0.1), np.arange(0.425, 1.401, 0.025)))

        # Encoder: enhanced physical features encoder with layer norm
        self.physical_enc = nn.Sequential(
            nn.Linear(physical_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # LSTM: Bidirectional LSTM with residual connections
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # used during training
        )

        # Layer normalization for LSTM outputs
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)  # *2 for bidirectionality

        # Prediction head: Project and predict the output value with a residual pathway
        self.current_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.current_head = nn.Linear(hidden_dim, 1, bias=True)  # Predict single value per timestep
        # EOS head for end-of-sequence prediction (auxiliary)
        self.eos_head = nn.Linear(hidden_dim, 1, bias=True)

        # Learnable start token initialization
        self.init_input = nn.Parameter(torch.randn(1) * 0.02)

    def init_hidden(self, physical):
        # Initialize hidden state using the physical encoder, accounting for bidirectionality.
        h0 = torch.tanh(self.physical_enc(physical))  # shape: (batch, hidden_dim)
        num_directions = 2 if self.lstm.bidirectional else 1
        # Repeat across layers and directions: (num_layers * num_directions, batch, hidden_dim)
        return h0.unsqueeze(0).repeat(self.num_layers * num_directions, 1, 1)

    def init_cell(self, physical):
        # Initialize cell state with zeros, accounting for bidirectionality.
        num_directions = 2 if self.lstm.bidirectional else 1
        return torch.zeros(self.num_layers * num_directions, physical.size(0), self.hidden_dim, device=physical.device)

    def _rnn_step(self, input_token, hidden, cell):
        """
        Single LSTM step: returns prediction, EOS logit, and updated hidden/cell.
        """
        out, (hidden, cell) = self.lstm(input_token, (hidden, cell))
        out_norm = self.layer_norm(out.squeeze(1))
        proj = self.current_projection(out_norm)
        pred = self.current_head(proj)
        eos = self.eos_head(proj)
        return pred, eos, hidden, cell

    def forward(self, physical, target_seq=None, lengths=None, teacher_forcing_ratio=0.5):
        """
        Forward pass for training (teacher forcing) and inference.
        """
        batch_size = physical.size(0)
        device = physical.device
        # expand voltage_points to shape [batch_size, T, 1]
        volt_seq = self.voltage_points.to(device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)

        # Initialize hidden & cell states
        hidden = self.init_hidden(physical.to(device))
        cell = self.init_cell(physical.to(device))

        # cache start token
        init_token = self.init_input.expand(batch_size, 1, 1).to(device)

        # Training mode: if full teacher forcing, use parallel LSTM; else scheduled sampling loop
        if target_seq is not None:
            max_len = target_seq.size(1)

            if teacher_forcing_ratio == 1.0:
                # fast parallel teacher forcing
                x = torch.cat([
                    target_seq.to(device).unsqueeze(-1),      # [B,T,1]
                    volt_seq                     # [B,T,1]
                ], dim=-1)                       # → [B,T,2]
                outputs, _ = self.lstm(x, (hidden, cell))
                outputs = self.layer_norm(outputs)
                projected = self.current_projection(outputs)
                preds = self.current_head(projected).squeeze(-1)
                # EOS logits per timestep
                eos_logits = self.eos_head(projected).squeeze(-1)
                return preds, eos_logits

            # scheduled sampling: mix ground truth and model predictions
            sam_outputs = []
            sam_eos = []
            # vectorized decisions per time-step
            teacher_mask = torch.rand(batch_size, max_len, device=device) < teacher_forcing_ratio

            # Initialize input token with learned start token and first voltage
            input_token = torch.cat([init_token, volt_seq[:, 0:1]], dim=-1)
            
            for t in range(max_len):
                # run one step
                pred, eos, hidden, cell = self._rnn_step(input_token, hidden, cell)
                sam_outputs.append(pred)
                sam_eos.append(eos)

                # prepare next current: either ground truth or model prediction
                if teacher_mask[:, t].all():
                    next_curr = target_seq[:, t].view(batch_size, 1, 1)
                else:
                    gt = target_seq[:, t].view(batch_size, 1, 1)
                    next_curr = torch.where(teacher_mask[:, t].view(batch_size, 1, 1), gt, pred.unsqueeze(-1))

                # attach next voltage (or zero padding)
                v_next = volt_seq[:, t+1:t+2] if t+1 < max_len else torch.zeros_like(init_token)
                input_token = torch.cat([next_curr, v_next], dim=-1)

            seq_preds = torch.cat(sam_outputs, dim=1)
            # compile EOS logits
            eos_logits = torch.cat(sam_eos, dim=1).squeeze(-1)
            return seq_preds, eos_logits

        # Inference (auto-regressive)
        if lengths is not None:
            max_len = max(lengths)
        else:
            max_len = self.max_sequence_length

        # Start with the cached init_token
        input_token = init_token
        value_outputs = []
        eos_outputs = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Inference (auto-regressive) loop
        for t in range(max_len):
            # run one step
            next_token, next_eos, hidden, cell = self._rnn_step(input_token, hidden, cell)
             
            value_outputs.append(next_token)
            eos_outputs.append(next_eos)
             
            # Update finished flags based on EOS prediction
            eos_prob = torch.sigmoid(next_eos)
            finished = finished | (eos_prob > self.eos_threshold)
            if finished.all():
                break
             
            # next_token is [batch,1] -> [batch,1,1]
            curr = next_token.unsqueeze(-1)

            # attach next voltage (or zero)
            v_next = volt_seq[:, t+1:t+2] if t+1<max_len else torch.zeros_like(init_token)
            input_token = torch.cat([curr, v_next], dim=-1)

        # Stack outputs
        seq_preds = torch.stack(value_outputs, dim=1).squeeze(-1)
        eos_logits = torch.stack(eos_outputs, dim=1).squeeze(-1)
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
            
            threshold = self.eos_threshold
            for i in range(batch_size):
                seq = outputs_cpu[i]
                eos = eos_cpu[i]
                # Find first position where EOS probability crosses threshold
                eos_positions = np.where(eos > threshold)[0]
                if len(eos_positions) > 0:
                    length = int(eos_positions[0]) + 1
                else:
                    # fallback to most likely EOS position
                    length = int(np.argmax(eos)) + 1
                lengths.append(length)
                # Convert to actual values
                unscaled = output_scaler.inverse_transform(seq[:length].reshape(-1, 1)).flatten()
                gen_curves.append(unscaled)
            
            return gen_curves, lengths

    def save_model(self, save_path, scalers, params):
        """
        Save model state, scalers, and parameters.
        """
        save_dict = {
            "model_state_dict": self.state_dict(),
            "scalers": scalers,
            "params": params
        }
        torch.save(save_dict, save_path)
        print(f"\nModel saved to {save_path}")

    @classmethod
    def load_model(cls, model_path, device):
        """
        Load a SeqIVModel from file.
        Returns:
            Tuple: (loaded model, scalers, max_sequence_length if available)
        """
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        params = checkpoint["params"]
        max_sequence_length = checkpoint.get("max_sequence_length", None)
        model = cls(
            physical_dim=params["physical_dim"],
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            max_sequence_length=max_sequence_length
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, checkpoint["scalers"], max_sequence_length
