"""
Mamba is no longer being actively developed for this research project.
"""

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import InferenceParams
import torch
import torch.nn as nn

class MambaIVModel(nn.Module):
    def __init__(self, physical_dim, d_model, num_mamba_layers, dropout=0.1,
                 max_seq_len=100, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super(MambaIVModel, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len # Used for default max_gen_length

        # 1. Embed physical parameters to d_model dimension.
        self.physical_embedding = nn.Linear(physical_dim, d_model)

        # 2. Embedding for the input sequence values (current values)
        self.value_embedding = nn.Linear(1, d_model) # Input values are scalars

        # 3. Stack of Mamba layers
        self.mamba_layers = nn.ModuleList()
        for i in range(num_mamba_layers):
            layer = Mamba(
                d_model=d_model,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand
            )
            # Crucial for Mamba's state management during batched inference with InferenceParams
            layer.layer_idx = i
            self.mamba_layers.append(layer)

        # 4. Dropout layer (optional, apply as needed)
        self.dropout_layer = nn.Dropout(dropout)

        # 5. Output projection: predicts value and EOS logit
        self.output_proj = nn.Linear(d_model, 2) # [value, EOS_logit]

        # 6. Learnable start token for sequence generation
        self.start_token = nn.Parameter(torch.zeros(1)) # Scalar value that gets embedded

    def forward(self, physical, target_seq=None, lengths=None, eos_targets=None, max_gen_length=None):
        """
        Generates IV curve sequences using Mamba.
        Args:
            physical (Tensor): Batch of physical parameters (batch_size, physical_dim)
            target_seq (Tensor, optional): True target sequence for teacher forcing (batch_size, max_len)
            lengths (Tensor, optional): True lengths of sequences (batch_size) - used in loss
            eos_targets (Tensor, optional): Binary EOS targets (batch_size, max_len+1) - used in loss
            max_gen_length (int, optional): Max length for auto-regressive generation (inference)
        Returns:
            Tuple: (value_outputs, eos_logits)
        """
        batch_size = physical.size(0)
        device = physical.device

        # Prepare conditioning from physical parameters
        # Shape: (batch_size, physical_dim) -> (batch_size, d_model)
        cond_embedding = self.physical_embedding(physical)

        if target_seq is not None:
            # --- Training or Evaluation with Teacher Forcing ---
            max_len = target_seq.size(1) # Actual max length of target data in this batch

            # Embed the start token for the batch
            # Shape: (batch_size, 1) -> (batch_size, 1, d_model)
            start_token_embedded = self.value_embedding(self.start_token.expand(batch_size, 1).unsqueeze(-1))

            # Embed the target sequence values
            # Shape: (batch_size, max_len) -> (batch_size, max_len, d_model)
            target_seq_embedded = self.value_embedding(target_seq.unsqueeze(-1))

            # Concatenate to form the full Mamba input sequence for teacher forcing
            # Input: [start_token, true_val_0, ..., true_val_{N-1}]
            mamba_input_seq = torch.cat([start_token_embedded, target_seq_embedded], dim=1)
            # Shape: (batch_size, max_len + 1, d_model)

            # ***IMPORTANT: Add conditioning to each step of the sequence (broadcast cond_embedding)***
            mamba_input_seq = mamba_input_seq + cond_embedding.unsqueeze(1)
            mamba_input_seq = self.dropout_layer(mamba_input_seq) # Apply dropout

            # Pass through Mamba layers
            hidden_states = mamba_input_seq
            for layer in self.mamba_layers:
                # During training/teacher-forcing, inference_params is None
                hidden_states = layer(hidden_states, inference_params=None)
            mamba_output = hidden_states # Shape: (batch_size, max_len + 1, d_model)

            # Project Mamba output to predictions [value, EOS_logit]
            combined_preds = self.output_proj(mamba_output) # Shape: (batch_size, max_len + 1, 2)

            # value_outputs correspond to predictions for target_seq
            value_outputs = combined_preds[:, :-1, 0] # Shape: (batch_size, max_len)
            # eos_logits are for all positions, including the one after the last target_seq item
            eos_logits = combined_preds[:, :, 1]      # Shape: (batch_size, max_len + 1)

            return value_outputs, eos_logits

        else:
            # --- Inference mode (auto-regressive generation) ---
            if max_gen_length is None:
                max_gen_length = self.max_seq_len

            value_outputs_list = []
            eos_logits_list = []

            # Initialize InferenceParams for Mamba state management
            # This object is stateful and gets updated by Mamba modules.
            inference_params = InferenceParams(
                max_seqlen=max_gen_length,
                max_batch_size=batch_size
            )

            # Initial input value for generation is the start_token
            current_val_for_embedding = self.start_token.expand(batch_size, 1) # Shape: (batch_size, 1)

            for t in range(max_gen_length):
                # Embed current value: (batch_size, 1) -> (batch_size, 1, d_model)
                current_input_embedded = self.value_embedding(current_val_for_embedding)

                # Add conditioning: (batch_size, 1, d_model) + (batch_size, 1, d_model)
                current_input_for_mamba = current_input_embedded + cond_embedding.unsqueeze(1)

                # Set the current sequence offset for Mamba's stateful processing
                inference_params.seqlen_offset = t

                # Pass the single time step input through Mamba layers
                hidden_states_step = current_input_for_mamba # Shape: (batch_size, 1, d_model)
                for layer in self.mamba_layers:
                    hidden_states_step = layer(hidden_states_step, inference_params=inference_params)
                mamba_output_token = hidden_states_step # Shape: (batch_size, 1, d_model)

                # Project the output of the last Mamba layer for the current token
                last_token_hidden_state = mamba_output_token[:, -1, :] # Shape: (batch_size, d_model)
                combined_preds_step = self.output_proj(last_token_hidden_state) # Shape: (batch_size, 2)

                value_pred_step = combined_preds_step[:, 0].unsqueeze(1) # Shape: (batch_size, 1)
                eos_logit_step = combined_preds_step[:, 1].unsqueeze(1)  # Shape: (batch_size, 1)

                value_outputs_list.append(value_pred_step)
                eos_logits_list.append(eos_logit_step)

                # Prepare the predicted value as the next input
                current_val_for_embedding = value_pred_step

                # This only works for batch_size = 1 because otherwise different samples in the batch have different EOS positions
                if batch_size == 1 and torch.sigmoid(eos_logit_step[0]).item() > 0.5:
                    break

            value_outputs = torch.cat(value_outputs_list, dim=1) # Shape: (batch_size, gen_len)
            eos_logits = torch.cat(eos_logits_list, dim=1)      # Shape: (batch_size, gen_len)

            return value_outputs, eos_logits

def load_trained_mamba_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    params = checkpoint['params']
    model = MambaIVModel(
        physical_dim=params['physical_dim'],
        d_model=params['d_model'],
        num_mamba_layers=params['num_mamba_layers'],
        dropout=params['dropout'],
        max_seq_len=checkpoint['max_sequence_length'],
        mamba_d_state=params['mamba_d_state'],
        mamba_d_conv=params['mamba_d_conv'],
        mamba_expand=params['mamba_expand']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['scalers'], checkpoint['max_sequence_length']