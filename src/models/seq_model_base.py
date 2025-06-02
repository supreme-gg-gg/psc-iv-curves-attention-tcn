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
        """
        pass

    @abstractmethod
    def generate_curve_batch(self, physical_input, scalers, device):
        """
        Generate IV curves in batch using auto-regressive generation.
        Args:
            physical_input (Tensor): Input physical parameters.
            scalers (tuple): Tuple containing (input_scaler, output_scaler) for unscale operations.
            device (torch.device): Device for inference.
            max_length_limit (int): Maximum generation length.
        Returns:
            tuple: (list of unscaled curves, list of generation lengths)
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