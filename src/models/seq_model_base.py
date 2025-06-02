import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class SeqModelBase(nn.Module, ABC):
    """
    Base class for sequence models to enforce a common interface.
    Both RNN and Transformer models should implement this interface.
    """

    @abstractmethod
    def forward(self, physical, target_seq=None, lengths=None, teacher_forcing_ratio=0.5):
        """
        Forward pass for the sequence model.
        
        Args:
            physical (Tensor): Input physical parameters (batch_size, physical_dim)
            target_seq (Tensor, optional): Target sequence for teacher forcing (batch_size, seq_len)
            lengths (list[int], optional): Actual lengths of each sequence in batch
            teacher_forcing_ratio (float, optional): Probability of using teacher forcing (0 to 1)
            
        Returns:
            Training mode (when target_seq is provided):
                tuple: (sequence_predictions, eos_logits)
                - sequence_predictions: Tensor of shape (batch_size, seq_len)
                - eos_logits: Tensor of shape (batch_size, seq_len), raw logits for EOS prediction
                
            Inference mode (when target_seq is None):
                tuple: (sequence_predictions, eos_logits)
                Generated sequence and corresponding EOS logits
        """
        pass

    @abstractmethod
    def generate_curve_batch(self, physical_input, scalers, device):
        """
        Generate IV curves in batch using auto-regressive generation.
        
        Args:
            physical_input (Tensor): Input physical parameters (batch_size, physical_dim)
            scalers (tuple): Tuple containing (input_scaler, output_scaler) for unscale operations
            device (torch.device): Device for inference
            
        Returns:
            tuple: (generated_curves, lengths)
            - generated_curves: List of unscaled curves, each having variable length based on EOS prediction
            - lengths: List of actual sequence lengths determined by EOS prediction
            
        Note:
            This method should:
            1. Use forward() in inference mode to get sequences and EOS predictions
            2. Determine sequence lengths using EOS probabilities (sigmoid(eos_logits) > 0.5)
            3. Apply output_scaler to convert predictions back to original scale
        """
        pass

    @abstractmethod
    def save_model(self, save_path, scalers, params):
        """
        Save the model state along with scalers and parameters.
        """
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, model_path, device):
        """
        Load a model from file.
        Returns:
            Tuple: (loaded model, scalers, [max_sequence_length] if applicable)
        """
        pass