import torch
import torch.nn as nn
from src.models.iv_model_base import IVModelBase 
import random
import numpy as np

class SeqIVModel(IVModelBase):
    def __init__(self, physical_dim, hidden_dim, num_layers=2, dropout=0.2, max_sequence_length=100, eos_threshold=0.5, **kwargs):
        super(SeqIVModel, self).__init__()
        self.physical_dim = physical_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length
        # threshold for EOS prediction
        self.eos_threshold = eos_threshold

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

    def forward(self, physical, target_seq=None, lengths=None, teacher_forcing_ratio=0.5):
        """
        Forward pass for training (teacher forcing) and inference.
        """
        batch_size = physical.size(0)
        device = physical.device

        # Initialize hidden & cell states
        hidden = self.init_hidden(physical.to(device))
        cell = self.init_cell(physical.to(device))

        # Training mode: if full teacher forcing, use parallel LSTM; else scheduled sampling loop
        if target_seq is not None:
            max_len = target_seq.size(1)

            if teacher_forcing_ratio == 1.0:
                # fast parallel teacher forcing
                x = target_seq.unsqueeze(-1).to(device)
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
            
            # Initialize input token with learned start token [batch, 1, 1]
            input_token = self.init_input.expand(batch_size, 1, 1).to(device)
            
            for t in range(max_len):
                out, (hidden, cell) = self.lstm(input_token, (hidden, cell))
                out = self.layer_norm(out.squeeze(1))
                projected = self.current_projection(out)
                pred = self.current_head(projected)  # shape (batch,1)
                sam_outputs.append(pred)
                # per-step EOS logit
                sam_eos.append(self.eos_head(projected))
                # decide next input
                use_teacher = (random.random() < teacher_forcing_ratio)
                if use_teacher:
                    # [batch] -> [batch, 1, 1]
                    input_token = target_seq[:, t].unsqueeze(1).unsqueeze(-1).to(device)
                else:
                    # pred is [batch, 1] -> [batch, 1, 1]
                    input_token = pred.unsqueeze(-1)
            
            seq_preds = torch.cat(sam_outputs, dim=1)
            # compile EOS logits
            eos_logits = torch.cat(sam_eos, dim=1).squeeze(-1)
            return seq_preds, eos_logits

        # Inference (auto-regressive)
        if lengths is not None:
            max_len = max(lengths)
        else:
            max_len = self.max_sequence_length

        # Start with the learned start token [batch, 1, 1]
        input_token = self.init_input.expand(batch_size, 1, 1).to(device)
        value_outputs = []
        eos_outputs = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(max_len):
            out, (hidden, cell) = self.lstm(input_token, (hidden, cell))
            out = self.layer_norm(out.squeeze(1))
            projected = self.current_projection(out)
            
            # Get value and EOS predictions
            next_token = self.current_head(projected)
            next_eos = self.eos_head(projected)
            
            value_outputs.append(next_token)
            eos_outputs.append(next_eos)
            
            # Update finished flags based on EOS prediction
            eos_prob = torch.sigmoid(next_eos)
            finished = finished | (eos_prob > self.eos_threshold)
            if finished.all():
                break
                
            # next_token is [batch, 1] -> [batch, 1, 1]
            input_token = next_token.unsqueeze(-1)
        
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
