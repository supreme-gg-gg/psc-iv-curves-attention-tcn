"""
Train a dual-head MLP model for current and voltage prediction.
This is standalone because current trainer doesn't support this much customization with loss and evaluations (esp with interp logic).
In the third (or the latest forth) refactoring, this will be merged into the main system.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import random

from src.models.baseline_mlp import BaselineMLP
from src.utils.preprocess import preprocess_fixed_length_dual_output


def dual_head_loss_standalone(
    current_pred,
    voltage_pred,
    current_targets,
    voltage_targets,
    current_weight=0.8,
    voltage_weight=0.2,
    physics_weight=0.1,
):
    """
    Standalone dual-head loss function.
    """
    # Current prediction loss (MSE)
    current_loss = F.mse_loss(current_pred, current_targets)

    # Voltage prediction loss (MSE)
    voltage_loss = F.mse_loss(voltage_pred, voltage_targets)

    # Physics constraints on current predictions
    physics_loss = torch.tensor(0.0, device=current_pred.device)
    if physics_weight > 0.0:
        # Monotonicity loss: penalize positive gradients
        if current_pred.size(1) > 1:
            diff = current_pred[:, 1:] - current_pred[:, :-1]
            monotonicity_loss = torch.relu(diff).pow(2).mean()

            # Smoothness loss: penalize large second derivatives
            if current_pred.size(1) > 2:
                second_diff = diff[:, 1:] - diff[:, :-1]
                smoothness_loss = second_diff.pow(2).mean()
            else:
                smoothness_loss = torch.tensor(0.0, device=current_pred.device)

            physics_loss = physics_weight * (monotonicity_loss + smoothness_loss)

    # Combined loss
    total_loss = (
        current_weight * current_loss + voltage_weight * voltage_loss + physics_loss
    )

    return total_loss, current_loss, voltage_loss, physics_loss


class DualHeadTrainerStandalone:
    """
    Standalone trainer for dual-head models.
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device,
        current_weight=0.8,
        voltage_weight=0.2,
        physics_weight=0.1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.current_weight = current_weight
        self.voltage_weight = voltage_weight
        self.physics_weight = physics_weight

    def train_one_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_current_loss = 0.0
        total_voltage_loss = 0.0
        total_physics_loss = 0.0

        for batch in train_loader:
            physical, current_targets, voltage_targets = batch[:3]
            physical = physical.to(self.device)
            current_targets = current_targets.to(self.device)
            voltage_targets = voltage_targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass - model returns (current_pred, voltage_pred)
            current_pred, voltage_pred = self.model(physical)

            # Compute loss
            total_loss_batch, current_loss, voltage_loss, physics_loss = (
                dual_head_loss_standalone(
                    current_pred,
                    voltage_pred,
                    current_targets,
                    voltage_targets,
                    self.current_weight,
                    self.voltage_weight,
                    self.physics_weight,
                )
            )

            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += total_loss_batch.item()
            total_current_loss += current_loss.item()
            total_voltage_loss += voltage_loss.item()
            total_physics_loss += physics_loss.item()

        num_batches = len(train_loader)
        return {
            "total": total_loss / num_batches,
            "current": total_current_loss / num_batches,
            "voltage": total_voltage_loss / num_batches,
            "physics": total_physics_loss / num_batches,
        }

    def validate_one_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                physical, current_targets, voltage_targets = batch[:3]
                physical = physical.to(self.device)
                current_targets = current_targets.to(self.device)
                voltage_targets = voltage_targets.to(self.device)

                current_pred, voltage_pred = self.model(physical)

                total_loss_batch, _, _, _ = dual_head_loss_standalone(
                    current_pred,
                    voltage_pred,
                    current_targets,
                    voltage_targets,
                    self.current_weight,
                    self.voltage_weight,
                    self.physics_weight,
                )

                total_loss += total_loss_batch.item()

        return total_loss / len(val_loader)


def calculate_interpolation_r2_standalone(
    pred_voltage, pred_current, true_voltage, true_current
):
    """
    Calculate R² using interpolation to handle different voltage sampling.
    """
    try:
        # Define common evaluation grid
        v_min = min(true_voltage.min(), pred_voltage.min())
        v_max = max(true_voltage.max(), pred_voltage.max())
        eval_grid = np.linspace(v_min, v_max, 100)

        # Interpolate both curves onto common grid
        pred_current_interp = np.interp(eval_grid, pred_voltage, pred_current)
        true_current_interp = np.interp(eval_grid, true_voltage, true_current)

        # Calculate R² on interpolated values
        return r2_score(true_current_interp, pred_current_interp)
    except Exception as e:
        print(f"Error calculating interpolation R²: {e}")
        return float("nan")


def evaluate_dual_head_standalone(model, test_loader, scalers, device):
    """
    Standalone evaluation for dual-head models using interpolation-based R².
    """
    model.eval()
    all_r2_scores = []
    sample_data = []

    isc_scaler, input_scaler, voltage_scaler = scalers

    with torch.no_grad():
        for batch in test_loader:
            physical, current_targets, voltage_targets = batch[:3]
            physical = physical.to(device)

            # Forward pass
            current_pred, voltage_pred = model(physical)

            # Convert to numpy and inverse transform
            current_pred_np = current_pred.cpu().numpy()
            voltage_pred_np = voltage_pred.cpu().numpy()
            current_targets_np = current_targets.cpu().numpy()
            voltage_targets_np = voltage_targets.cpu().numpy()

            batch_size = physical.size(0)
            for i in range(batch_size):
                # NOTE: careful with how you do reshape here
                # Inverse transform current
                true_current = isc_scaler.inverse_transform(
                    current_targets_np[i].reshape(1, -1)
                ).flatten()
                pred_current = isc_scaler.inverse_transform(
                    current_pred_np[i].reshape(1, -1)
                ).flatten()

                # Inverse transform voltage
                true_voltage = voltage_scaler.inverse_transform(
                    voltage_targets_np[i].reshape(1, -1)
                ).flatten()
                pred_voltage = voltage_scaler.inverse_transform(
                    voltage_pred_np[i].reshape(1, -1)
                ).flatten()

                # Calculate interpolation-based R²
                r2 = calculate_interpolation_r2_standalone(
                    pred_voltage, pred_current, true_voltage, true_current
                )

                if not np.isnan(r2):
                    all_r2_scores.append(r2)
                    sample_data.append(
                        {
                            "pred_voltage": pred_voltage,
                            "pred_current": pred_current,
                            "true_voltage": true_voltage,
                            "true_current": true_current,
                            "r2": r2,
                        }
                    )

    mean_r2 = np.mean(all_r2_scores) if all_r2_scores else float("nan")

    print("\nDual-Head Model Evaluation Summary:")
    print(f"- Mean R² Score: {mean_r2:.4f} ({len(all_r2_scores)} valid samples)")

    negative_r2_count = sum(1 for r2 in all_r2_scores if r2 < 0)
    print(f"- Negative R² Scores: {negative_r2_count} samples")

    return mean_r2, sample_data


def plot_dual_head_results_standalone(sample_data, num_samples=4):
    """
    Plot results for dual-head models.
    """
    if len(sample_data) > num_samples:
        sample_data = random.sample(sample_data, num_samples)

    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i, data in enumerate(sample_data):
        ax = axes[i]

        # Plot predicted and true curves
        ax.plot(
            data["pred_voltage"],
            data["pred_current"],
            label="Predicted",
            color="blue",
            linewidth=2,
        )
        ax.plot(
            data["true_voltage"],
            data["true_current"],
            label="True",
            color="orange",
            linewidth=2,
        )

        ax.set_title(f"Sample {i + 1} - R²: {data['r2']:.4f}")
        ax.legend()
        ax.grid(True)
        ax.set_ylabel("Current (A)")

    axes[-1].set_xlabel("Voltage (V)")
    plt.tight_layout()
    plt.savefig("dual_head_results_standalone.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    """Main training function."""

    # Data paths
    TRAIN_INPUT_PATHS = [
        "dataset/Data_10k_sets/Data_10k_rng1/LHS_parameters_m.txt",
        "dataset/Data_10k_sets/Data_10k_rng2/LHS_parameters_m.txt",
        "dataset/Data_10k_sets/Data_10k_rng3/LHS_parameters_m.txt",
    ]

    TRAIN_OUTPUT_PATHS = [
        "dataset/Data_10k_sets/Data_10k_rng1/iV_m.txt",
        "dataset/Data_10k_sets/Data_10k_rng2/iV_m.txt",
        "dataset/Data_10k_sets/Data_10k_rng3/iV_m.txt",
    ]

    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 2e-3
    CURRENT_WEIGHT = 0.8
    VOLTAGE_WEIGHT = 0.2
    PHYSICS_WEIGHT = 0.1
    HIDDEN_DIMS = [128, 64, 64]

    # Fixed-length preprocessing parameters
    NUM_PRE = 3
    NUM_POST = 4
    HIGH_RES_POINTS = 10000
    ROUND_DIGITS = 2

    # Setup device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load and preprocess data
    print("\n--- Loading and Preprocessing Data ---")
    data = preprocess_fixed_length_dual_output(
        input_paths=TRAIN_INPUT_PATHS,
        output_paths=TRAIN_OUTPUT_PATHS,
        num_pre=NUM_PRE,
        num_post=NUM_POST,
        high_res_points=HIGH_RES_POINTS,
        round_digits=ROUND_DIGITS,
        test_size=0.2,
        isc_scaler_method="median",
        voc_scaler_method="median",
    )

    # Extract data shapes
    X_train, Y_current_train, Y_voltage_train, _ = data["train"]
    X_test, Y_current_test, Y_voltage_test, _ = data["test"]

    input_dim = X_train.shape[1]
    output_dim = Y_current_train.shape[1]

    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Current targets shape: {Y_current_train.shape}")
    print(f"Voltage targets shape: {Y_voltage_train.shape}")

    # Create data loaders
    train_dataset = TensorDataset(X_train, Y_current_train, Y_voltage_train)
    test_dataset = TensorDataset(X_test, Y_current_test, Y_voltage_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    print("\n--- Initializing Dual-Head Model ---")
    model = BaselineMLP(
        input_dim=input_dim, output_dim=output_dim, hidden_dims=HIDDEN_DIMS
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    print(
        f"Model architecture: {input_dim} -> {' -> '.join(map(str, HIDDEN_DIMS))} -> 2x{output_dim}"
    )

    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5, min_lr=1e-6
    )

    # Initialize trainer
    trainer = DualHeadTrainerStandalone(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        current_weight=CURRENT_WEIGHT,
        voltage_weight=VOLTAGE_WEIGHT,
        physics_weight=PHYSICS_WEIGHT,
    )

    # Training loop
    print("\n--- Starting Training ---")
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        # Train and validate
        train_loss_dict = trainer.train_one_epoch(train_loader)
        val_loss = trainer.validate_one_epoch(test_loader)

        train_losses.append(train_loss_dict["total"])
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"Train Loss: {train_loss_dict['total']:.6f} "
                f"(C: {train_loss_dict['current']:.4f}, "
                f"V: {train_loss_dict['voltage']:.4f}, "
                f"P: {train_loss_dict['physics']:.4f}) | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {current_lr:.2e}"
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scalers": data["scalers"],
                    "hyperparameters": {
                        "input_dim": input_dim,
                        "output_dim": output_dim,
                        "hidden_dims": HIDDEN_DIMS,
                        "current_weight": CURRENT_WEIGHT,
                        "voltage_weight": VOLTAGE_WEIGHT,
                        "physics_weight": PHYSICS_WEIGHT,
                    },
                },
                "checkpoints/dual_head_mlp_standalone.pth",
            )

    print("\n--- Training Complete ---")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Final evaluation
    print("\n--- Final Evaluation ---")
    mean_r2, sample_data = evaluate_dual_head_standalone(
        model, test_loader, data["scalers"], device
    )

    print(f"Final R² Score: {mean_r2:.4f}")

    # Plot results
    if sample_data:
        plot_dual_head_results_standalone(sample_data, num_samples=4)

    # Print final statistics
    print("\nTraining Summary:")
    print(f"- Final train loss: {train_losses[-1]:.6f}")
    print(f"- Final val loss: {val_losses[-1]:.6f}")
    print(f"- Best val loss: {best_val_loss:.6f}")
    print("- Model saved to: dual_head_mlp_standalone.pth")
    print("- Results plot saved as: dual_head_results_standalone.png")


if __name__ == "__main__":
    main()
