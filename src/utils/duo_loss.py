import torch
import torch.nn.functional as F


def enhanced_dual_output_loss(
    predicted_pairs: torch.Tensor,  # Shape: (batch, seq_len, 2) [current, voltage]
    target_pairs: torch.Tensor,  # Shape: (batch, seq_len, 2) [current, voltage]
    target_mask: torch.Tensor = None,  # Shape: (batch, seq_len), True for valid points
    current_weight: float = 0.5,
    voltage_weight: float = 0.3,
    monotonicity_weight: float = 0.1,
    smoothness_weight: float = 0.05,
    endpoint_weight: float = 0.05,  # For Jsc and Voc constraints
):
    """
    Enhanced loss for dual-output (current, voltage) sequence models with physics constraints.
    Includes monotonicity, smoothness, and endpoint (Jsc/Voc) penalties.
    """
    pred_currents = predicted_pairs[..., 0]
    pred_voltages = predicted_pairs[..., 1]
    target_currents = target_pairs[..., 0]
    target_voltages = target_pairs[..., 1]

    # Basic MSE losses
    current_loss_elementwise = F.mse_loss(
        pred_currents, target_currents, reduction="none"
    )
    voltage_loss_elementwise = F.mse_loss(
        pred_voltages, target_voltages, reduction="none"
    )

    if target_mask is not None:
        valid_points = target_mask.sum().clamp(min=1.0)
        current_loss = (current_loss_elementwise * target_mask).sum() / valid_points
        voltage_loss = (voltage_loss_elementwise * target_mask).sum() / valid_points
    else:
        current_loss = current_loss_elementwise.mean()
        voltage_loss = voltage_loss_elementwise.mean()

    # Basic reconstruction loss
    reconstruction_loss = (current_weight * current_loss) + (
        voltage_weight * voltage_loss
    )

    # Physics-based penalties
    physics_loss = torch.tensor(0.0, device=predicted_pairs.device)

    if pred_currents.size(1) > 1:  # Need at least 2 points for derivatives
        # 1. Monotonicity constraint: current should decrease as voltage increases
        diff_current = pred_currents[:, 1:] - pred_currents[:, :-1]
        diff_voltage = pred_voltages[:, 1:] - pred_voltages[:, :-1]

        # Mask for valid consecutive pairs
        if target_mask is not None and target_mask.size(1) > 1:
            mono_mask = target_mask[:, 1:] * target_mask[:, :-1]
            valid_mono_points = mono_mask.sum().clamp(min=1.0)
        else:
            mono_mask = torch.ones_like(
                diff_current, dtype=torch.bool, device=diff_current.device
            )
            valid_mono_points = mono_mask.sum().clamp(min=1.0)

        # Penalize current increase when voltage increases (unphysical)
        undesired_increase = (diff_current > 0) & (diff_voltage > 0)
        monotonicity_loss = (
            diff_current.pow(2) * undesired_increase.float() * mono_mask
        ).sum() / valid_mono_points

        # 2. Smoothness constraint: penalize large second derivatives
        if pred_currents.size(1) > 2:
            second_diff_current = diff_current[:, 1:] - diff_current[:, :-1]
            second_diff_voltage = diff_voltage[:, 1:] - diff_voltage[:, :-1]

            if target_mask is not None and target_mask.size(1) > 2:
                smooth_mask = (
                    target_mask[:, 2:] * target_mask[:, 1:-1] * target_mask[:, :-2]
                )
                valid_smooth_points = smooth_mask.sum().clamp(min=1.0)
            else:
                smooth_mask = torch.ones_like(second_diff_current, dtype=torch.bool)
                valid_smooth_points = smooth_mask.sum().clamp(min=1.0)

            current_smoothness = (
                second_diff_current.pow(2) * smooth_mask
            ).sum() / valid_smooth_points
            voltage_smoothness = (
                second_diff_voltage.pow(2) * smooth_mask
            ).sum() / valid_smooth_points
            smoothness_loss = current_smoothness + voltage_smoothness
        else:
            smoothness_loss = torch.tensor(0.0, device=predicted_pairs.device)

        physics_loss = (
            monotonicity_weight * monotonicity_loss
            + smoothness_weight * smoothness_loss
        )

    # 3. Endpoint constraints (Jsc and Voc)
    endpoint_loss = torch.tensor(0.0, device=predicted_pairs.device)
    if endpoint_weight > 0.0:
        # Jsc constraint: first point should be at V≈0, I should be positive (short circuit)
        jsc_voltage_penalty = (
            pred_voltages[:, 0].pow(2).mean()
        )  # First voltage should be near 0
        jsc_current_penalty = (
            torch.relu(-pred_currents[:, 0]).pow(2).mean()
        )  # Current should be positive

        # Voc constraint: last point should be at I≈0, V should be positive (open circuit)
        if target_mask is not None:
            # Find last valid point for each sequence
            seq_lengths = target_mask.sum(dim=1).long()
            batch_indices = torch.arange(
                predicted_pairs.size(0), device=predicted_pairs.device
            )
            last_currents = pred_currents[
                batch_indices, seq_lengths.clamp(max=pred_currents.size(1) - 1)
            ]
            last_voltages = pred_voltages[
                batch_indices, seq_lengths.clamp(max=pred_voltages.size(1) - 1)
            ]
        else:
            last_currents = pred_currents[:, -1]
            last_voltages = pred_voltages[:, -1]

        voc_current_penalty = last_currents.pow(
            2
        ).mean()  # Last current should be near 0
        voc_voltage_penalty = (
            torch.relu(-last_voltages).pow(2).mean()
        )  # Voltage should be positive

        endpoint_loss = (
            jsc_voltage_penalty
            + jsc_current_penalty
            + voc_current_penalty
            + voc_voltage_penalty
        )

    total_loss = reconstruction_loss + physics_loss + endpoint_weight * endpoint_loss

    return (
        total_loss,
        current_loss,
        voltage_loss,
        monotonicity_loss if "monotonicity_loss" in locals() else torch.tensor(0.0),
        smoothness_loss if "smoothness_loss" in locals() else torch.tensor(0.0),
        endpoint_loss,
    )


def minimal_dual_output_loss(
    predicted_pairs: torch.Tensor,  # Shape: (batch, seq_len, 2) [current, voltage]
    target_pairs: torch.Tensor,  # Shape: (batch, seq_len, 2) [current, voltage]
    target_mask: torch.Tensor = None,  # Shape: (batch, seq_len), True for valid points
    current_weight: float = 0.7,
    voltage_weight: float = 0.3,
    monotonicity_weight: float = 0.01,
):
    """
    Simplified loss for dual-output (current, voltage) sequence models.
    Use this for initial training or when physics constraints cause instability.
    """
    pred_currents = predicted_pairs[..., 0]
    pred_voltages = predicted_pairs[..., 1]
    target_currents = target_pairs[..., 0]
    target_voltages = target_pairs[..., 1]

    # Basic MSE losses
    current_loss_elementwise = F.mse_loss(
        pred_currents, target_currents, reduction="none"
    )
    voltage_loss_elementwise = F.mse_loss(
        pred_voltages, target_voltages, reduction="none"
    )

    if target_mask is not None:
        valid_points = target_mask.sum().clamp(min=1.0)
        current_loss = (current_loss_elementwise * target_mask).sum() / valid_points
        voltage_loss = (voltage_loss_elementwise * target_mask).sum() / valid_points
    else:
        current_loss = current_loss_elementwise.mean()
        voltage_loss = voltage_loss_elementwise.mean()

    total_loss = (current_weight * current_loss) + (voltage_weight * voltage_loss)

    # Simple monotonicity penalty
    if monotonicity_weight > 0.0 and pred_currents.size(1) > 1:
        diff_current = pred_currents[:, 1:] - pred_currents[:, :-1]
        diff_voltage = pred_voltages[:, 1:] - pred_voltages[:, :-1]

        if target_mask is not None and target_mask.size(1) > 1:
            mono_mask = target_mask[:, 1:] * target_mask[:, :-1]
            valid_mono_points = mono_mask.sum().clamp(min=1.0)
        else:
            mono_mask = torch.ones_like(
                diff_current, dtype=torch.bool, device=diff_current.device
            )
            valid_mono_points = mono_mask.sum().clamp(min=1.0)

        undesired_increase = (diff_current > 0) & (diff_voltage > 0)
        monotonicity_penalty = (
            diff_current.pow(2) * undesired_increase.float() * mono_mask
        ).sum() / valid_mono_points

        total_loss = total_loss + monotonicity_weight * monotonicity_penalty

    return total_loss, current_loss, voltage_loss


def shape_aware_dual_loss(
    predicted_pairs: torch.Tensor,
    target_pairs: torch.Tensor,
    target_mask: torch.Tensor = None,
    current_weight: float = 0.5,
    voltage_weight: float = 0.3,
    shape_weight: float = 0.2,  # Weight for derivative matching
):
    """
    Dual output loss with shape-aware component inspired by sequence_loss_with_eos.
    Focuses on matching the shape/curvature of IV curves.
    """
    pred_currents = predicted_pairs[..., 0]
    pred_voltages = predicted_pairs[..., 1]
    target_currents = target_pairs[..., 0]
    target_voltages = target_pairs[..., 1]

    # Basic MSE losses
    current_mse = F.mse_loss(pred_currents, target_currents, reduction="none")
    voltage_mse = F.mse_loss(pred_voltages, target_voltages, reduction="none")

    if target_mask is not None:
        valid_points = target_mask.sum().clamp(min=1.0)
        current_loss = (current_mse * target_mask).sum() / valid_points
        voltage_loss = (voltage_mse * target_mask).sum() / valid_points
    else:
        current_loss = current_mse.mean()
        voltage_loss = voltage_mse.mean()

    # Shape loss: match derivatives (inspired by compute_shape_aware_loss)
    shape_loss = torch.tensor(0.0, device=predicted_pairs.device)
    if pred_currents.size(1) > 1:
        # First derivatives
        pred_current_diff = pred_currents[:, 1:] - pred_currents[:, :-1]
        target_current_diff = target_currents[:, 1:] - target_currents[:, :-1]
        pred_voltage_diff = pred_voltages[:, 1:] - pred_voltages[:, :-1]
        target_voltage_diff = target_voltages[:, 1:] - target_voltages[:, :-1]

        if target_mask is not None and target_mask.size(1) > 1:
            diff_mask = target_mask[:, 1:] * target_mask[:, :-1]
            valid_diff_points = diff_mask.sum().clamp(min=1.0)
        else:
            diff_mask = torch.ones_like(pred_current_diff, dtype=torch.bool)
            valid_diff_points = diff_mask.sum().clamp(min=1.0)

        current_shape_loss = (
            (pred_current_diff - target_current_diff) ** 2 * diff_mask
        ).sum() / valid_diff_points
        voltage_shape_loss = (
            (pred_voltage_diff - target_voltage_diff) ** 2 * diff_mask
        ).sum() / valid_diff_points
        shape_loss = current_shape_loss + voltage_shape_loss

    total_loss = (
        current_weight * current_loss
        + voltage_weight * voltage_loss
        + shape_weight * shape_loss
    )

    return total_loss, current_loss, voltage_loss, shape_loss
