"""
Loss functions for different IV model types.
Uses function pointer pattern for flexible loss computation.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

def sequence_loss_with_eos(outputs: Tuple[torch.Tensor, torch.Tensor], 
                          targets: torch.Tensor, 
                          mask: torch.Tensor,
                          eos_targets: Optional[torch.Tensor] = None,
                          eos_loss_weight: float = 1.0,
                          lambda_shape: float = 0.5,
                          **kwargs) -> torch.Tensor:
    """
    Loss function for sequence models (Transformer, RNN) with EOS prediction.
    
    Args:
        outputs: (sequence_predictions, eos_logits)
        targets: Target sequence values
        mask: Binary mask for valid positions
        eos_targets: Binary EOS targets
        eos_loss_weight: Weight for EOS loss component
        lambda_shape: Weight for shape loss component
    """
    seq_preds, eos_logits = outputs
    
    # Shape-aware reconstruction loss
    loss = compute_shape_aware_loss(seq_preds, targets, mask, lambda_shape=lambda_shape)
    
    # EOS loss if available
    if eos_logits is not None and eos_targets is not None and eos_loss_weight > 0.0:
        eos_loss = compute_eos_bce_loss(eos_logits, eos_targets)
        loss = loss + eos_loss_weight * eos_loss
    
    return loss

def simple_sequence_loss(outputs: torch.Tensor, 
                        targets: torch.Tensor, 
                        mask: torch.Tensor,
                        **kwargs) -> torch.Tensor:
    """Simple MSE loss for sequence models without EOS."""
    if isinstance(outputs, tuple):
        outputs = outputs[0]  # Take only sequence predictions, ignore EOS
    
    mse = ((outputs - targets) ** 2) * mask
    return mse.sum() / mask.sum()

def cvae_loss_function(outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                      targets: torch.Tensor,
                      mask: torch.Tensor,
                      eos_targets: Optional[torch.Tensor] = None,
                      kl_beta: float = 1.0,
                      eos_weight: float = 2.0,
                      physics_loss_weight: float = 0.5,
                      monotonicity_weight: float = 0.5,
                      smoothness_weight: float = 0.5,
                      **kwargs) -> torch.Tensor:
    """
    Loss function for CVAE model with physics constraints and EOS prediction.
    
    Args:
        outputs: (reconstructed_curves, eos_logits, mu, logvar)
        targets: Target IV curves
        mask: Binary mask for valid positions
        eos_targets: Binary EOS targets
        kl_beta: Weight for KL divergence
        eos_weight: Weight for EOS loss
        physics_loss_weight: Weight for physics constraints
    """
    reconstructed, eos_logits, mu, logvar = outputs
    
    # Masked reconstruction loss
    mse_element = (reconstructed - targets)**2 * mask
    mse = mse_element.sum() / mask.sum()

    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # EOS loss
    eos_loss = torch.tensor(0.0, device=targets.device)
    if eos_logits is not None and eos_targets is not None:
        if eos_targets.size(1) != eos_logits.size(1):
            eos_targets = eos_targets[:, :eos_logits.size(1)]
        eos_loss = F.binary_cross_entropy_with_logits(eos_logits, eos_targets.float())
    
    # Physics constraints
    mono_loss_pred, smooth_loss_pred = physics_constraints_loss(reconstructed * mask)
    mono_loss_true, smooth_loss_true = physics_constraints_loss(targets * mask)
    
    # Use true curve losses as scaling factors
    mono_scale = torch.clamp(mono_loss_true, min=1e-6)
    smooth_scale = torch.clamp(smooth_loss_true, min=1e-6)
    
    # Normalized physics losses
    physics_loss = physics_loss_weight * (
        monotonicity_weight * (mono_loss_pred / mono_scale) +
        smoothness_weight * (smooth_loss_pred / smooth_scale)
    )
    
    return mse + kl_beta * kld + eos_weight * eos_loss + physics_loss

def compute_shape_aware_loss(outputs, targets, mask, lambda_shape=0.25, mse_weight=1.0):
    """
    Computes a shape-aware loss combining MSE and derivative matching.
    """
    pointwise_mse = ((outputs - targets) ** 2) * mask
    mse_term = pointwise_mse.sum() / mask.sum()

    # Differential Loss (masked)
    shape_term1 = torch.tensor(0.0, device=outputs.device, requires_grad=outputs.requires_grad)
    shape_term2 = torch.tensor(0.0, device=outputs.device, requires_grad=outputs.requires_grad)

    # Ensure there's at least 2 points to compute difference for shape loss
    if outputs.size(1) > 1 and targets.size(1) > 1:
        # Calculate derivatives (differences between adjacent points)
        true_diff1 = targets[:, 1:] - targets[:, :-1]
        pred_diff1 = outputs[:, 1:] - outputs[:, :-1]

        # Mask for the differences: valid if both points in the original pair were valid
        diff_mask = mask[:, 1:] * mask[:, :-1]

        diff_mask_sum = diff_mask.sum()
        if diff_mask_sum > 0:
            shape_mse = ((pred_diff1 - true_diff1) ** 2) * diff_mask
            shape_term1 = shape_mse.sum() / diff_mask_sum

        if outputs.size(1) > 2 and targets.size(1) > 2:
            
            # Second derivatives
            true_diff2 = true_diff1[:, 1:] - true_diff1[:, :-1]
            pred_diff2 = pred_diff1[:, 1:] - pred_diff1[:, :-1]

            # Mask for second derivatives requires three consecutive valid points
            diff2_mask = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
            
            diff2_mask_sum = diff2_mask.sum()
            if diff2_mask_sum > 0:
                shape_mse2 = ((pred_diff2 - true_diff2) ** 2) * diff2_mask
                shape_term2 = shape_mse2.sum() / diff2_mask_sum

    # Total loss
    total_loss = mse_weight * mse_term + lambda_shape * shape_term1 + lambda_shape * shape_term2
    return total_loss

def compute_eos_bce_loss(eos_logits, eos_targets, gamma: float = 2.0):
    """
    Compute binary cross-entropy loss for EOS predictions with focal weighting.
    """
    # Align targets to eos_logits timesteps
    if eos_targets.size(1) != eos_logits.size(1):
        eos_targets = eos_targets[:, :eos_logits.size(1)]
    
    # Standard BCE per-element (no reduction)
    bce = F.binary_cross_entropy_with_logits(
        eos_logits,
        eos_targets.float(),
        reduction='none'
    )
    # Compute focal weighting: p_t = prob if target==1 else (1-prob)
    prob = torch.sigmoid(eos_logits)
    p_t = prob * eos_targets.float() + (1 - prob) * (1 - eos_targets.float())
    focal_factor = (1 - p_t) ** gamma
    # Apply focal factor
    loss = (focal_factor * bce).mean()
    return loss

def physics_constraints_loss(iv_curves):
    """Calculate physics-based constraints loss for IV curves.
    Args:
        iv_curves: Tensor of shape (batch_size, num_voltage_points)
    Returns:
        Tuple of (monotonicity_loss, smoothness_loss)
    """
    # Monotonicity loss: penalize positive gradients (current should decrease with voltage)
    diff = iv_curves[:, 1:] - iv_curves[:, :-1]  # First-order differences
    monotonicity_loss = torch.relu(diff).pow(2).mean()  # Squared ReLU to only penalize positive gradients
    
    # Smoothness loss: penalize large second derivatives
    second_diff = diff[:, 1:] - diff[:, :-1]  # Second-order differences
    smoothness_loss = second_diff.pow(2).mean()
    
    return monotonicity_loss, smoothness_loss