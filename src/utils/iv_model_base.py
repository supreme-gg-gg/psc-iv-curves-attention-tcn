import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Union, Any, Dict
import numpy as np

class IVModelBase(nn.Module, ABC):
    """
    Enhanced base class for all IV models (sequence and generative).
    Provides unified interface for training, inference, and evaluation.
    """

    @abstractmethod
    def forward(self, physical: torch.Tensor, target_seq: Optional[torch.Tensor] = None,
                lengths: Optional[List[int]] = None, teacher_forcing_ratio: Optional[float] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass for training (and inference for sequence models).
        
        Args:
            physical: Input physical parameters (batch_size, physical_dim)
            target_seq: Target sequence for training
                       - Required for CVAE models (always in training mode)
                       - Optional for sequence models (None = inference mode)
            lengths: Actual lengths of sequences in batch (sequence models only)
            teacher_forcing_ratio: Probability of using teacher forcing (sequence models only)
            
        Returns:
            Model outputs for loss computation:
                - For sequence models: (sequence_predictions, eos_logits)
                - For CVAE: (reconstructed_curves, eos_logits, mu, logvar)
        """
        pass

    @abstractmethod
    def generate_curve_batch(self, physical_input: torch.Tensor, scalers: Tuple, device: torch.device) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate IV curves for evaluation and plotting.
        NOTE: This function MUST handle the truncation of sequences if applicable.
        
        Args:
            physical_input: Input physical parameters (batch_size, physical_dim)
            scalers: (input_scaler, output_scaler) for data transformation
            device: Device for computation
            
        Returns:
            tuple: (generated_curves, lengths)
            - generated_curves: List of unscaled curves (variable length)
            - lengths: List of actual curve lengths
        """
        pass

    @abstractmethod
    def save_model(self, save_path: str, scalers: Tuple, params: Dict[str, Any]) -> None:
        """Save model state, scalers, and parameters."""
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, model_path: str, device: torch.device) -> Tuple['IVModelBase', Tuple, Optional[int]]:
        """
        Load model from file.
        
        Returns:
            tuple: (loaded_model, scalers, max_sequence_length_if_applicable)
        """
        pass

    def get_model_type(self) -> str:
        """Return model type identifier for trainer routing."""
        return self.__class__.__name__