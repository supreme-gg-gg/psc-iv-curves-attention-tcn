import torch
import numpy as np
from torch import nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class FourierFeatures(nn.Module):
    def __init__(
        self,
        num_bands: int,
        max_val: float = 1.4,
        mode: str = "logspace",
        sigma: float = 10.0,
        input_dim: int = 1,
    ):
        """
        mode: 'logspace' (default, positional encoding) or 'gaussian' (Gaussian RFF)
        sigma: standard deviation for Gaussian RFF
        input_dim: input dimension (for Gaussian RFF, usually 1 for voltage)
        """
        super().__init__()
        self.mode = mode
        self.max_val = max_val
        self.num_bands = num_bands
        self.sigma = sigma
        self.input_dim = input_dim
        if mode == "logspace":
            self.register_buffer("B", torch.logspace(0, 3, steps=num_bands))
        elif mode == "gaussian":
            # B: (num_bands, input_dim)
            B = torch.randn(num_bands, input_dim) * sigma
            self.register_buffer("B", B)
        else:
            raise ValueError(f"Unknown FourierFeatures mode: {mode}")

    def forward(self, v_grid: torch.Tensor) -> torch.Tensor:
        if self.mode == "logspace":
            v_norm = (v_grid / self.max_val).unsqueeze(-1)  # (batch, seq, 1)
            v_freq = v_norm * self.B  # (batch, seq, num_bands)
            return torch.cat(
                [torch.sin(2 * np.pi * v_freq), torch.cos(2 * np.pi * v_freq)], dim=-1
            )
        elif self.mode == "gaussian":
            # v_grid: (batch, seq) or (batch, seq, input_dim)
            if v_grid.dim() == 2 and self.input_dim == 1:
                v = v_grid.unsqueeze(-1)  # (batch, seq, 1)
            else:
                v = v_grid  # (batch, seq, input_dim)
            # (batch, seq, num_bands)
            v_proj = torch.matmul(v, self.B.t())  # (batch, seq, num_bands)
            return torch.cat(
                [torch.sin(2 * np.pi * v_proj), torch.cos(2 * np.pi * v_proj)], dim=-1
            )
        else:
            raise ValueError(f"Unknown FourierFeatures mode: {self.mode}")


def print_model_info(model: nn.Module):
    print("\n--- Model Architecture ---")
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")
    print("-------------------------------------------------\n\n")
