import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional, Any, Dict
from src.utils.iv_model_base import IVModelBase


class BaselineMLP(IVModelBase):
    """
    Lightweight MLP model for fixed-length IV curve prediction.
    Uses the new preprocess_fixed_length preprocessing method.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = None):
        """
        Initialize MLP model.

        Args:
            input_dim: Number of input physical parameters
            output_dim: Number of output points (fixed length)
            hidden_dims: List of hidden layer dimensions
        """
        super(BaselineMLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Default architecture if not specified
        if hidden_dims is None:
            hidden_dims = [128, 64, 64]

        # Build base network for shared representation
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim

        self.net_base = nn.Sequential(*layers)

        # one head for both
        self.output_head = nn.Sequential(
            nn.Linear(prev_dim, output_dim * 2),  # output_dim for current + voltage
            nn.Sigmoid()  # Both current and voltage in [0, 1]
        )

    def forward(
        self,
        physical: torch.Tensor,
        target_seq: Optional[torch.Tensor] = None,
        lengths: Optional[List[int]] = None,
        teacher_forcing_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Forward pass for fixed-length prediction.

        Args:
            physical: Input physical parameters (batch_size, physical_dim)
            target_seq: Not used for MLP but kept for interface compatibility
            lengths: Not used for fixed-length prediction
            teacher_forcing_ratio: Not used for MLP

        Returns:
            Predicted IV curves (batch_size, output_dim)
        """
        # Shared encoding
        hidden = self.net_base(physical)
        # Predict current and voltage positions
        output = self.output_head(hidden)

        batch_size = output.size(0)
        # Split output into current and voltage predictions
        output = output.view(batch_size, self.output_dim, 2)
        curr_pred = output[:, :, 0]  # Current predictions
        volt_pred = output[:, :, 1]  # Voltage predictions 

        # Return both heads
        return curr_pred, volt_pred

    def generate_curve_batch(
        self, physical_input: torch.Tensor, scalers: Tuple, device: torch.device
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate IV curves for evaluation.

        Args:
            physical_input: Input physical parameters (batch_size, physical_dim)
            scalers: (input_scaler, output_scaler) for data transformation
            device: Device for computation

        Returns:
            tuple: (generated_curves, lengths)
            - generated_curves: List of unscaled curves (all same length)
            - lengths: List of actual curve lengths (all same for fixed-length)
        """
        self.eval()
        input_scaler, output_scaler = scalers

        with torch.no_grad():
            # Forward pass produces (current_pred, voltage_pred)
            curr_pred, volt_pred = self.forward(physical_input.to(device))

            # Convert to numpy and inverse transform
            curr_np = curr_pred.cpu().numpy()
            volt_np = volt_pred.cpu().numpy()
            generated_curves = []
            lengths = []

            for i in range(curr_np.shape[0]):
                # Current inverse transform
                curve_c = curr_np[i].reshape(-1, 1)
                curve_physical = output_scaler.inverse_transform(curve_c).flatten()
                # Voltage raw predictions (optionally apply activation or scaling)
                curve_voltage = volt_np[i]
                generated_curves.append((curve_voltage, curve_physical))
                lengths.append(len(curve_physical))

        return generated_curves, lengths

    def save_model(
        self, save_path: str, scalers: Tuple, params: Dict[str, Any]
    ) -> None:
        """Save model state, scalers, and parameters."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_params": params,
            "scalers": scalers,
            "model_type": "BaselineMLP",
        }
        torch.save(checkpoint, save_path)

    @classmethod
    def load_model(
        cls, model_path: str, device: torch.device
    ) -> Tuple["BaselineMLP", Tuple, Optional[int]]:
        """
        Load model from file.

        Returns:
            tuple: (loaded_model, scalers, None) - None because fixed-length has no max_seq_len
        """
        checkpoint = torch.load(model_path, map_location=device)

        # Extract parameters
        params = checkpoint["model_params"]
        scalers = checkpoint["scalers"]

        # Create model
        model = cls(
            input_dim=params["input_dim"],
            output_dim=params["output_dim"],
            hidden_dims=params.get("hidden_dims"),
        )

        # Load state
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        return model, scalers, None
