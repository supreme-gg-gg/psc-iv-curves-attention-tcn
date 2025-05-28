import torch
from typing import Dict, Any, List
import numpy as np

from .base import BaseTrainer
from models.seq_iv_model import SeqIVModel

class SeqTrainer(BaseTrainer):
    """Trainer for sequence-based IV curve model."""
    
    def __init__(self, model: SeqIVModel, **kwargs):
        super().__init__(model=model, **kwargs)
        self.teacher_forcing_ratio = self.config['model'].get('teacher_forcing_ratio', 0.5)
        self.min_teacher_forcing = self.config['model'].get('min_teacher_forcing', 0.1)
        self.teacher_forcing_decay = self.config['model'].get('teacher_forcing_decay', 0.98)

    def compute_loss(self, batch: tuple) -> torch.Tensor:
        """
        Compute sequence model loss with teacher forcing.
        
        Args:
            batch: Tuple of (physical_params, true_curves, masks)
            
        Returns:
            Mean squared error loss over valid sequence positions
        """
        physical_params, true_curves, masks = batch
        
        # Move to device
        physical_params = physical_params.to(self.device)
        true_curves = true_curves.to(self.device)
        masks = masks.to(self.device)
        
        # Get sequence lengths from masks
        lengths = masks.sum(dim=1).int()
        
        # Forward pass with teacher forcing
        outputs = self.model(
            physical_params,
            target_seq=true_curves,
            lengths=lengths,
            teacher_forcing_ratio=self.teacher_forcing_ratio
        )
        
        # Compute masked MSE loss
        loss = self._compute_sequence_loss(outputs, true_curves, masks, lengths)
        
        # Log components if needed
        if hasattr(self, 'logger'):
            self.logger.info({
                'sequence_loss': loss.item(),
                'teacher_forcing_ratio': self.teacher_forcing_ratio
            })
        
        return loss

    def _compute_sequence_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mean squared error loss over valid sequence positions.
        
        Args:
            outputs: Model outputs of shape (batch_size, max_len)
            targets: Target sequences of shape (batch_size, max_len)
            masks: Binary masks of shape (batch_size, max_len)
            lengths: Sequence lengths of shape (batch_size,)
            
        Returns:
            Average MSE loss over valid positions
        """
        # Compute squared errors
        squared_errors = (outputs - targets) ** 2
        
        # Apply mask to consider only valid positions
        masked_errors = squared_errors * masks
        
        # Compute mean over valid positions
        total_error = masked_errors.sum()
        num_valid = masks.sum()
        
        return total_error / (num_valid + 1e-8)  # Add small epsilon to prevent division by zero

    def _update_teacher_forcing_ratio(self, epoch: int) -> None:
        """
        Update teacher forcing ratio based on training progress.
        
        Args:
            epoch: Current epoch number
        """
        if self.config['model'].get('use_teacher_forcing_decay', True):
            self.teacher_forcing_ratio = max(
                self.min_teacher_forcing,
                self.teacher_forcing_ratio * self.teacher_forcing_decay
            )

    def train_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int) -> float:
        """
        Train for one epoch with dynamic teacher forcing.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        avg_loss = super().train_epoch(train_loader, epoch)
        self._update_teacher_forcing_ratio(epoch)
        return avg_loss

    def evaluate_generation(
        self,
        dataloader: torch.utils.data.DataLoader,
        scalers: tuple,
        num_samples: int = 4,
        save_dir: str = None
    ) -> Dict[str, float]:
        """
        Evaluate sequence generation quality.
        
        Args:
            dataloader: Test data loader
            scalers: Tuple of (input_scaler, output_scaler)
            num_samples: Number of samples to generate
            save_dir: Optional directory to save generation plots
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        _, output_scaler = scalers
        results = []
        
        with torch.no_grad():
            for batch in dataloader:
                physical_params, true_curves, masks = batch
                physical_params = physical_params.to(self.device)
                
                # Generate complete sequences
                outputs = self.model(
                    physical_params,
                    target_seq=None,  # No teacher forcing during generation
                    lengths=masks.sum(dim=1).int(),
                    teacher_forcing_ratio=0.0
                )
                
                # Convert to original scale
                for i in range(len(outputs)):
                    valid_len = int(masks[i].sum().item())
                    true_curve = true_curves[i, :valid_len]
                    pred_curve = outputs[i, :valid_len]
                    
                    # Inverse transform
                    true_unscaled = output_scaler.inverse_transform(
                        true_curve.cpu().numpy().reshape(-1, 1)
                    ).flatten()
                    pred_unscaled = output_scaler.inverse_transform(
                        pred_curve.cpu().numpy().reshape(-1, 1)
                    ).flatten()
                    
                    # Compute metrics
                    results.append({
                        'true': true_unscaled,
                        'pred': pred_unscaled,
                        'length': valid_len
                    })
                
                if len(results) >= num_samples:
                    break
        
        # Compute aggregate metrics
        metrics = self._compute_generation_metrics(results)
        
        # Save visualization if requested
        if save_dir:
            self._save_generation_plots(results, save_dir)
        
        return metrics

    def _compute_generation_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Compute metrics for generated sequences."""
        r2_scores = []
        
        for result in results:
            r2 = r2_score(result['true'], result['pred'])
            r2_scores.append(r2)
        
        return {
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'r2_median': np.median(r2_scores)
        }