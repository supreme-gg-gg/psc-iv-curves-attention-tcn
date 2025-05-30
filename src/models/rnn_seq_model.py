import torch
import torch.nn as nn
from src.models.seq_model_base import SeqModelBase
import random

class SeqIVModel(SeqModelBase):
    def __init__(self, physical_dim, hidden_dim, num_layers=2, dropout=0.2, scaled_zero_threshold=0.0, max_sequence_length=100):
        super(SeqIVModel, self).__init__()
        self.physical_dim = physical_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.scaled_zero_threshold = scaled_zero_threshold
        self.max_sequence_length = max_sequence_length

        # Encoder: enhanced physical features encoder with layer norm
        self.physical_enc = nn.Sequential(
            nn.Linear(physical_dim, hidden_dim * 2, bias=True),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim)
        )

        # LSTM: Bidirectional LSTM with residual connections
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # used during training
        )

        # Layer normalization for LSTM outputs
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)  # *2 for bidirectionality

        # Prediction head: Project and predict the output value with a residual pathway
        self.current_projection = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        self.current_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=True)
        )

        # Learnable start token initialization
        self.init_input = nn.Parameter(torch.randn(1) * 0.02)

    def init_hidden(self, physical):
        # Initialize hidden state using the physical encoder.
        h0 = torch.tanh(self.physical_enc(physical))
        return h0.unsqueeze(0).repeat(self.num_layers, 1, 1)

    def init_cell(self, physical):
        # Initialize cell state with zeros.
        return torch.zeros(self.num_layers, physical.size(0), self.hidden_dim, device=physical.device)

    def forward(self, physical, target_seq=None, lengths=None, teacher_forcing_ratio=0.5):
        """
        Forward pass for training (teacher forcing) and inference.
        Args:
            physical (Tensor): Input physical parameters (batch, physical_dim).
            target_seq (Tensor, optional): Target sequence for teacher forcing.
            lengths (Tensor, optional): Lengths of target sequences.
            teacher_forcing_ratio (float): Probability to use teacher forcing.
        Returns:
            Tensor: Generated or predicted sequence values.
        """
        batch_size = physical.size(0)
        # Determine maximum sequence length for generation.
        if target_seq is not None:
            max_len = target_seq.size(1)
        elif lengths is not None:
            max_len = max(lengths)
        else:
            max_len = 100

        device = physical.device
        hidden = self.init_hidden(physical)
        cell = self.init_cell(physical)
        # Start with the learned start token.
        input_token = self.init_input.expand(batch_size, 1).unsqueeze(1)
        current_outputs = []

        # Training and teacher forcing loop
        for t in range(max_len):
            out, (hidden, cell) = self.lstm(input_token, (hidden, cell))
            out = self.layer_norm(out.squeeze(1))
            projected = self.current_projection(out)
            current_pred = self.current_head(projected)
            if t > 0:
                # Apply a weighted residual connection.
                prev_pred = current_outputs[-1]
                current_pred = 0.8 * current_pred + 0.2 * prev_pred
            current_outputs.append(current_pred)

            # In inference mode, check for threshold termination.
            if target_seq is None:
                finished = (current_pred < self.scaled_zero_threshold).squeeze()
                if finished.all():
                    break
            
            # Decide on teacher forcing.
            use_teacher = (target_seq is not None) and (random.random() < teacher_forcing_ratio)
            next_input = target_seq[:, t].unsqueeze(1) if use_teacher else current_pred
            input_token = next_input.unsqueeze(1)
        return torch.cat(current_outputs, dim=1)

    def generate_curve_batch(self, physical_input, scalers, device):
        """
        Generate IV curves in batch using auto-regressive generation.
        """
        self.eval()
        _, output_scaler = scalers
        batch_size = physical_input.size(0)
        with torch.no_grad():
            # Move inputs to device
            phys = physical_input.to(device)
            hidden = self.init_hidden(phys)
            cell = self.init_cell(phys)
            input_token = self.init_input.expand(batch_size, 1).unsqueeze(1).to(device)
            # Pre-allocate output tensor and tracking
            max_len = self.max_sequence_length if self.max_sequence_length is not None else 100
            outputs_tensor = torch.zeros(batch_size, max_len, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
            scaled_zero = output_scaler.transform([[0]])[0][0]
            # Auto-regressive generation
            for t in range(max_len):
                out, (hidden, cell) = self.lstm(input_token, (hidden, cell))
                out = self.layer_norm(out.squeeze(1))
                projected = self.current_projection(out)
                current_pred = self.current_head(projected)
                # save predictions
                pred_flat = current_pred.squeeze(1)
                outputs_tensor[:, t] = pred_flat
                # update finished flags
                finished = finished | (pred_flat < scaled_zero)
                # record first finish time
                new_done = (lengths == 0) & finished
                lengths[new_done] = t + 1
                if finished.all():
                    break
                # next input
                input_token = current_pred.unsqueeze(1)
            # for any never-finished sequences
            if (lengths == 0).any():
                lengths[lengths == 0] = (t + 1)
            # Collect and inverse transform
            outputs_cpu = outputs_tensor.detach().cpu().numpy()
            gen_curves = [
                output_scaler.inverse_transform(
                    outputs_cpu[i, :lengths[i]].reshape(-1, 1)
                ).flatten()
                for i in range(batch_size)
            ]
            return gen_curves, lengths.cpu().tolist()

    def save_model(self, save_path, scalers, params):
        """
        Save model state, scalers, and parameters.
        """
        save_dict = {
            "model_state_dict": self.state_dict(),
            "scalers": scalers,
            "params": params
        }
        if self.max_sequence_length is not None:
            save_dict["max_sequence_length"] = self.max_sequence_length
        if self.scaled_zero_threshold is not None:
            save_dict["scaled_zero_threshold"] = self.scaled_zero_threshold
        torch.save(save_dict, save_path)
        print(f"\nModel saved to {save_path}")

    @classmethod
    def load_model(cls, model_path, device):
        """
        Load a SeqIVModel from file.
        Returns:
            Tuple: (loaded model, scalers, max_sequence_length if available)
        """
        checkpoint = torch.load(model_path, map_location=device)
        params = checkpoint["params"]
        max_sequence_length = checkpoint.get("max_sequence_length", None)
        model = cls(
            physical_dim=params["physical_dim"],
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            scaled_zero_threshold=params.get("scaled_zero_threshold", 0.0)
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, checkpoint["scalers"], max_sequence_length
