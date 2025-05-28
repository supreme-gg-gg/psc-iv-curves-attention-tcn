import torch
from typing import Dict, Any, Tuple
import numpy as np

from .base import BaseTrainer
from models.cvae import CVAE

class CVAETrainer(BaseTrainer):
    """Trainer for CVAE model with physics-informed loss."""
    
    def __init__(self, model: CVAE, **kwargs):
        super().__init__(model=model, **kwargs)
        self.kld_weight = self.config['model'].get('kld_weight', 1.0)
        self.physics_weight = self.config['model'].get('physics_weight', 0.1)
        self.monotonicity_weight = self.config['model'].get('monotonicity_weight', 0.7)
        self.smoothness_weight = self.config['model'].get('smoothness_weight', 0.3)

    def _compute_physics_loss(self, iv_curves: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate physics-based constraints loss for IV curves.
        
        Args:
            iv_curves: Tensor of shape (batch_size, num_voltage_points)
            
        Returns:
            Tuple of (monotonicity_loss, smoothness_loss)
        """
        # Monotonicity loss: penalize positive gradients
        diff = iv_curves[:, 1:] - iv_curves[:, :-1]
        monotonicity_loss = torch.relu(diff).pow(2).mean()
        
        # Smoothness loss: penalize large second derivatives
        second_diff = diff[:, 1:] - diff[:, :-1]
        smoothness_loss = second_diff.pow(2).mean()
        
        return monotonicity_loss, smoothness_loss

    def compute_loss(self, batch: tuple) -> torch.Tensor:
        """
        Compute CVAE loss with physics constraints.
        
        Args:
            batch: Tuple of (physical_params, true_curves) or 
                  (physical_params, true_curves, masks)
        
        Returns:
            Total loss combining reconstruction, KL divergence, and physics constraints
        """
        # Unpack batch
        if len(batch) == 2:  # No masks
            physical_params, true_curves = batch
            masks = None
        else:  # With masks
            physical_params, true_curves, masks = batch
            
        physical_params = physical_params.to(self.device)
        true_curves = true_curves.to(self.device)
        if masks is not None:
            masks = masks.to(self.device)
        
        # Forward pass
        reconstructed_curves, mu, logvar = self.model(physical_params, true_curves)
        
        # Reconstruction loss
        if masks is not None:
            # Masked MSE
            mse = (((reconstructed_curves - true_curves) ** 2) * masks).sum() / masks.sum()
        else:
            mse = torch.nn.functional.mse_loss(reconstructed_curves, true_curves, reduction='mean')
        
        # KL divergence
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # Physics constraints
        mono_loss_pred, smooth_loss_pred = self._compute_physics_loss(reconstructed_curves)
        mono_loss_true, smooth_loss_true = self._compute_physics_loss(true_curves)
        
        # Use true curve losses as scaling factors
        mono_scale = torch.clamp(mono_loss_true, min=1e-6)
        smooth_scale = torch.clamp(smooth_loss_true, min=1e-6)
        
        # Normalized physics loss
        physics_loss = (
            self.physics_weight * 
            (self.monotonicity_weight * (mono_loss_pred / mono_scale) +
             self.smoothness_weight * (smooth_loss_pred / smooth_scale))
        )
        
        # Total loss
        total_loss = mse + self.kld_weight * kld + physics_loss
        
        # Log components if needed
        if hasattr(self, 'logger'):
            self.logger.info({
                'mse_loss': mse.item(),
                'kl_loss': kld.item(),
                'physics_loss': physics_loss.item(),
                'total_loss': total_loss.item()
            })
        
        return total_loss

    def train_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int) -> float:
        """
        Train for one epoch with KL annealing.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        # KL annealing logic
        if 'kl_annealing' in self.config['training']:
            annealing_config = self.config['training']['kl_annealing']
            total_epochs = self.config['training']['epochs']
            start_weight = annealing_config.get('start_weight', 0.0)
            
            if epoch < annealing_config.get('start_epoch', total_epochs // 3):
                # Linear or sigmoid annealing
                if annealing_config.get('type', 'linear') == 'linear':
                    progress = epoch / annealing_config['start_epoch']
                    self.kld_weight = start_weight + (
                        self.config['model']['kld_weight'] - start_weight
                    ) * progress
                else:  # sigmoid
                    progress = epoch / annealing_config['start_epoch']
                    self.kld_weight = self.config['model']['kld_weight'] / (
                        1 + np.exp(-10 * (progress - 0.5))
                    )
            else:
                self.kld_weight = self.config['model']['kld_weight']
        
        return super().train_epoch(train_loader, epoch)