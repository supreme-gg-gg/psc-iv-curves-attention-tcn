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
                          physics_loss_weight: float = 0.5,
                          monotonicity_weight: float = 0.5,
                          smoothness_weight: float = 0.5,
                          **kwargs) -> torch.Tensor:
    """
    Loss function for sequence models (Transformer, RNN) with EOS prediction and physics constraints.
    
    Args:
        outputs: (sequence_predictions, eos_logits)
        targets: Target sequence values
        mask: Binary mask for valid positions
        eos_targets: Binary EOS targets
        eos_loss_weight: Weight for EOS loss component
        lambda_shape: Weight for shape loss component
        physics_loss_weight: Weight for physics constraints
        monotonicity_weight: Weight for monotonicity penalty
        smoothness_weight: Weight for smoothness penalty
    """
    seq_preds, eos_logits = outputs
    
    # Shape-aware reconstruction loss
    loss = compute_shape_aware_loss(seq_preds, targets, mask, lambda_shape=lambda_shape)
    
    # EOS loss if available
    if eos_logits is not None and eos_targets is not None and eos_loss_weight > 0.0:
        eos_loss = compute_eos_bce_loss(eos_logits, eos_targets)
        loss = loss + eos_loss_weight * eos_loss

    # Physics-based monotonicity and smoothness penalties
    if physics_loss_weight > 0.0:
        mono_pred, smooth_pred = physics_constraints_loss(seq_preds * mask)
        # Optional normalization by target physics loss (clamp to avoid div by zero)
        mono_true, smooth_true = physics_constraints_loss(targets * mask)
        mono_scale = torch.clamp(mono_true, min=1e-6)
        smooth_scale = torch.clamp(smooth_true, min=1e-6)
        phys_loss = physics_loss_weight * (
            monotonicity_weight * (mono_pred / mono_scale) +
            smoothness_weight * (smooth_pred / smooth_scale)
        )
        loss = loss + phys_loss
    
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

def _masked_loss_window(mask: torch.Tensor, orig_len: torch.Tensor, window: int):
    # indices shape: (seq_len,)
    seq_len = mask.size(1)
    idx = torch.arange(seq_len, device=mask.device).unsqueeze(0)  # (1, seq_len)
    mpp_idx = (orig_len - 1).unsqueeze(1)  # (batch,1)
    # boolean window mask
    w = (idx >= (mpp_idx - window)) & (idx <= (mpp_idx + window))
    return w.float()


def physics_informed_sequence_loss(
    outputs: Tuple[torch.Tensor, torch.Tensor],
    targets: torch.Tensor,
    mask: torch.Tensor,
    eos_targets: Optional[torch.Tensor] = None,
    eos_loss_weight: float = 1.0,
    loss_weights: dict = None,
    knee_window: int = 5,
    gamma: float = 2.0,
    **kwargs
) -> torch.Tensor:
    """
    Combined sequence loss: knee-weighted MSE + local monotonicity, curvature, J_sc, V_oc penalties + EOS BCE.
    loss_weights keys: mse, monotonicity, curvature, jsc, voc
    """
    seq_preds, eos_logits = outputs
    loss_weights = loss_weights or {}

    # lengths per sample
    orig_len = mask.sum(dim=1).long()

    # knee-weighted MSE
    se = (seq_preds - targets).pow(2)
    window_mask = _masked_loss_window(mask, orig_len, knee_window)
    knee_w = 1.0 + (loss_weights.get('mse_knee_factor', 1.0) - 1.0) * window_mask
    weighted_se = se * mask * knee_w
    mse_per = weighted_se.sum(dim=1) / (mask * knee_w).sum(dim=1).clamp(min=1)
    mse = mse_per.mean()
    total = loss_weights.get('mse', 1.0) * mse

    # monotonicity penalty
    diffs = seq_preds[:,1:] - seq_preds[:,:-1]
    diff_mask = mask[:,1:]*mask[:,:-1]
    win_d = window_mask[:,1:]*window_mask[:,:-1]
    mono = (torch.relu(diffs) * diff_mask * win_d).pow(2).sum(dim=1) / diff_mask.mul(win_d).sum(dim=1).clamp(min=1)
    total = total + loss_weights.get('monotonicity',1.0)*mono.mean()
    
    # curvature penalty
    curv = seq_preds[:,2:] - 2*seq_preds[:,1:-1] + seq_preds[:,:-2]
    curv_mask = mask[:,2:]*mask[:,1:-1]*mask[:,:-2]
    win_c = window_mask[:,2:]*window_mask[:,1:-1]*window_mask[:,:-2]
    curv_term = (curv.pow(2)*curv_mask*win_c).sum(dim=1)/curv_mask.mul(win_c).sum(dim=1).clamp(min=1)
    total = total + loss_weights.get('curvature',1.0)*curv_term.mean()

    # J_sc (first point) and V_oc (last)
    # J_sc: targ at idx0
    jsc_mse = ((targets[:,0]-seq_preds[:,0]).pow(2)*mask[:,0]).mean()
    total = total + loss_weights.get('jsc',1.0)*jsc_mse

    # V_oc: last valid point should be -1
    batch = seq_preds.size(0)
    idxs = torch.arange(batch, device=seq_preds.device)
    last = seq_preds[idxs, orig_len.clamp(min=1)-1]
    voc_mse = (last.add(1).pow(2)).mean()
    total = total + loss_weights.get('voc',1.0)*voc_mse

    # EOS loss
    if eos_logits is not None and eos_targets is not None and eos_loss_weight>0:
        from torch.nn.functional import binary_cross_entropy_with_logits as bce
        # align
        if eos_targets.size(1)!=eos_logits.size(1): eos_targets = eos_targets[:,:eos_logits.size(1)]
        b = bce(eos_logits, eos_targets.float(), reduction='none')
        p = torch.sigmoid(eos_logits)
        p_t = p* eos_targets.float() + (1-p)*(1-eos_targets.float())
        focal = (1-p_t).pow(gamma)
        eos_loss = (focal * b).mean()
        total = total + eos_loss_weight*eos_loss

    return total
