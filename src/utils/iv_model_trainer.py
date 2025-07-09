"""
Unified trainer for all IV models accepting any loss function.
"""

import torch
import numpy as np
from sklearn.metrics import r2_score
from typing import Dict, Any, Tuple, Type, Callable, List, Optional
from iv_model_base import IVModelBase
import random
import matplotlib.pyplot as plt
from src.utils.loss_functions import enhanced_dual_output_loss, minimal_dual_output_loss


class IVModelTrainer:
    """
    Unified trainer for all IV models accepting any loss function.
    """

    def __init__(
        self,
        model: IVModelBase,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable,
        device: torch.device = torch.device("cpu"),
        loss_params: Optional[dict] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        compile_model: bool = False,
    ):
        """
        Initialize unified trainer.

        Args:
            model: Any model inheriting from IVModelBase
            optimizer: PyTorch optimizer
            scheduler: Optional learning rate scheduler
            device: Device for training
            loss_function: Callable that computes loss from model outputs and targets
            loss_params: Dictionary of parameters to pass to loss function
            compile_model: If True, attempt to compile the model using torch.compile
        """
        self.model = model

        # If torch.compile is available, try to compile the model
        if hasattr(torch, "compile") and compile_model:
            try:
                self.model = torch.compile(self.model)
                print("Model compiled successfully using torch.compile")
            except Exception as e:
                print(f"torch.compile failed: {e}")

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.loss_function = loss_function
        self.loss_params = loss_params or {}

        if self.loss_function is None:
            raise ValueError(
                "loss_function must be provided. Import from src.models.loss_functions"
            )

    def train_one_epoch(self, train_loader, **epoch_kwargs) -> float:
        """Train for one epoch using the configured loss function."""
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch_data = self._unpack_batch(batch)

            self.optimizer.zero_grad()
            outputs = self.model(
                batch_data["physical"],
                target_seq=batch_data["padded_seq"],
                lengths=batch_data["lengths"],
                teacher_forcing_ratio=epoch_kwargs.get("teacher_forcing_ratio", 0.5),
            )

            loss_kwargs = {**self.loss_params, **epoch_kwargs}
            loss = self.loss_function(
                outputs=outputs,
                targets=batch_data["padded_seq"],
                mask=batch_data["mask"],
                lengths=batch_data["lengths"],
                eos_targets=batch_data.get("eos_targets"),
                **loss_kwargs,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * batch_data["physical"].size(0)

        return total_loss / len(train_loader.dataset)

    def validate_one_epoch(self, val_loader, **epoch_kwargs) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch_data = self._unpack_batch(batch)

                outputs = self.model(
                    batch_data["physical"],
                    target_seq=batch_data["padded_seq"],
                    lengths=batch_data["lengths"],
                    teacher_forcing_ratio=0.0,  # No teacher forcing during validation
                )

                loss_kwargs = {**self.loss_params, **epoch_kwargs}
                loss = self.loss_function(
                    outputs=outputs,
                    targets=batch_data["padded_seq"],
                    mask=batch_data["mask"],
                    eos_targets=batch_data.get("eos_targets"),
                    **loss_kwargs,
                )

                total_loss += loss.item() * batch_data["physical"].size(0)

        return total_loss / len(val_loader.dataset)

    def evaluate(
        self, test_loader, scalers, include_plots: bool = True
    ) -> Tuple[float, List]:
        """
        Unified evaluation using R² metrics and visual plots for all model types.
        """
        self.model.eval()
        all_r2_scores = []
        sample_data = []
        total_samples = 0
        length_mismatch_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch_data = self._unpack_batch(batch)
                physical = batch_data["physical"]
                padded_seq = batch_data["padded_seq"]

                if "lengths" in batch_data:
                    lengths = batch_data["lengths"]
                else:
                    raise ValueError("Lengths must be provided for evaluation")

                # Generate curves using model's generate_curve_batch method
                # Shape of generated_curves: [batch_size, max_length]
                # Shape of gen_lengths: [batch_size] or [batch_size, 1]
                generated_curves, gen_lengths = self.model.generate_curve_batch(
                    physical, scalers, self.device
                )
                _, output_scaler = scalers

                # Evaluate using R² metric
                batch_size = physical.size(0)
                for i in range(batch_size):
                    total_samples += 1

                    true_len = (
                        lengths[i].item()
                        if isinstance(lengths[i], torch.Tensor)
                        else lengths[i]
                    )
                    true_curve = padded_seq[i, :true_len].cpu().numpy()
                    true_unscaled = output_scaler.inverse_transform(
                        true_curve.reshape(-1, 1)
                    ).flatten()

                    # Get generated curve
                    gen_len = (
                        gen_lengths[i]
                        if isinstance(gen_lengths[i], (int, float))
                        else int(gen_lengths[i])
                    )
                    gen_curve = generated_curves[i]

                    # Count length mismatches
                    if gen_len != true_len:
                        length_mismatch_count += 1

                    # Compute R² on minimum length
                    min_len = min(true_len, gen_len)
                    if min_len > 1:
                        y_true_r2 = true_unscaled[:min_len]
                        y_pred_r2 = gen_curve[:min_len]
                        r2 = r2_score(y_true_r2, y_pred_r2)
                        all_r2_scores.append(r2)

                        # Store for plotting
                        sample_data.append(
                            (gen_curve, true_unscaled, r2, gen_len, true_len)
                        )

        mean_r2 = np.mean(all_r2_scores) if all_r2_scores else float("nan")
        print("\nModel Evaluation Summary:")
        print(f"- Mean R² Score: {mean_r2:.4f} ({len(all_r2_scores)} valid samples)")

        negative_r2_count = sum(1 for r2 in all_r2_scores if r2 < 0)
        print(f"- Negative R² Scores: {negative_r2_count} samples")
        print(
            f"- Length mismatches: {length_mismatch_count}/{total_samples} sequences ({100 * length_mismatch_count / total_samples:.1f}%)"
        )

        if include_plots:
            try:
                self.plot_generated_curves(sample_data, num_samples_to_plot=4)
            except Exception as e:
                print(f"Error plotting generated curves: {e}")

        return mean_r2, sample_data

    def _unpack_batch(self, batch) -> Dict[str, torch.Tensor]:
        """Unpack batch data to unified format."""
        # Handle different batch formats
        if len(batch) == 5:
            physical, padded_seq, mask, lengths, eos_targets = batch
        elif len(batch) == 4:
            physical, padded_seq, mask, lengths = batch
            eos_targets = None
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        return {
            "physical": physical.to(self.device),
            "padded_seq": padded_seq.to(self.device),
            "mask": mask.to(self.device),
            "lengths": lengths.to(self.device),
            "eos_targets": eos_targets.to(self.device)
            if eos_targets is not None
            else None,
        }

    @staticmethod
    def plot_generated_curves(sample_data, num_samples_to_plot=4):
        """Unified plotting for all model types."""
        # Random sampling of curves to plot
        if len(sample_data) > num_samples_to_plot:
            sample_data = random.sample(sample_data, num_samples_to_plot)

        num_samples = len(sample_data)
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2 * num_samples))
        if num_samples == 1:
            axes = [axes]

        Va = np.concatenate(
            (np.arange(0, 0.41, 0.1), np.arange(0.425, 1.401, 0.025))
        )  # applied voltage, V

        for i in range(num_samples):
            gen_curve, true_curve, r2, gen_len, true_len = sample_data[i]
            ax = axes[i]

            # Plot each curve truncated to available Va length
            gen_n = min(len(gen_curve), len(Va))
            true_n = min(len(true_curve), len(Va))
            # Plot generated curve
            ax.plot(
                Va[:gen_n],
                gen_curve[:gen_n],
                label=f"Generated (len={gen_len})",
                color="blue",
                linewidth=2,
            )
            # Plot true curve
            ax.plot(
                Va[:true_n],
                true_curve[:true_n],
                label=f"True (len={true_len})",
                color="orange",
                linewidth=2,
            )
            # Ensure x-axis spans full longer of the two
            max_n = max(gen_n, true_n)
            ax.set_xlim(0, Va[max_n - 1] if max_n > 0 else Va[0])

            ax.set_title(
                f"Sample {i + 1} - R²: {r2:.4f} | Gen: {gen_len}, True: {true_len}"
            )
            ax.legend()
            ax.grid()
            ax.set_ylabel("Current (A)")

        axes[-1].set_xlabel("Voltage (V)")
        plt.tight_layout()
        plt.savefig("generated_curves.png", dpi=150, bbox_inches="tight")
        plt.show()

    def save_model(self, save_path: str, scalers: Tuple, params: Dict[str, Any]):
        """Save model through its save_model implementation."""
        self.model.save_model(save_path, scalers, params)

    @staticmethod
    def load_model(
        model_class: Type[IVModelBase],
        model_path: str,
        device: torch.device = torch.device("cpu"),
    ):
        """Load model through the model class's load_model implementation."""
        return model_class.load_model(model_path, device)


# Re-using the R2 calculation from the MLP trainer example
def calculate_interpolation_r2_standalone(
    pred_voltage, pred_current, true_voltage, true_current
):
    try:
        # Define common evaluation grid
        # Ensure voltage arrays are not empty and have more than one unique value for interpolation
        if (
            len(true_voltage) < 2
            or len(pred_voltage) < 2
            or len(np.unique(true_voltage)) < 2
            or len(np.unique(pred_voltage)) < 2
        ):
            # print("Warning: Not enough data points for interpolation in R2 calculation.")
            return float("nan")

        v_min_true, v_max_true = true_voltage.min(), true_voltage.max()
        v_min_pred, v_max_pred = pred_voltage.min(), pred_voltage.max()

        # Determine overall min/max for the common grid, ensuring range is valid
        v_min_common = min(v_min_true, v_min_pred)
        v_max_common = max(v_max_true, v_max_pred)

        if (
            v_max_common <= v_min_common
        ):  # Avoid issues with single point or reversed range
            # print("Warning: Invalid voltage range for interpolation in R2 calculation.")
            return float("nan")

        eval_grid = np.linspace(
            v_min_common, v_max_common, 100
        )  # 100 points for common eval

        # Interpolate both curves onto common grid
        # Need to handle cases where pred_voltage might not be strictly monotonic
        # Sort pred_voltage and corresponding pred_current if necessary for np.interp
        sort_idx_pred = np.argsort(pred_voltage)
        pred_voltage_sorted = pred_voltage[sort_idx_pred]
        pred_current_sorted = pred_current[sort_idx_pred]

        sort_idx_true = np.argsort(
            true_voltage
        )  # True voltages should be sorted from preprocessing
        true_voltage_sorted = true_voltage[sort_idx_true]
        true_current_sorted = true_current[sort_idx_true]

        pred_current_interp = np.interp(
            eval_grid, pred_voltage_sorted, pred_current_sorted
        )
        true_current_interp = np.interp(
            eval_grid, true_voltage_sorted, true_current_sorted
        )

        return r2_score(true_current_interp, pred_current_interp)
    except Exception as e:
        # print(f"Error calculating interpolation R²: {e}")
        return float("nan")


class DualOutputIVModelTrainer:
    def __init__(
        self,
        model,  # TransformerDualOutputIVModel instance
        optimizer,
        scheduler,
        device,
        current_loss_weight=0.5,
        voltage_loss_weight=0.3,
        monotonicity_loss_weight=0.1,
        smoothness_loss_weight=0.05,
        endpoint_loss_weight=0.05,
        use_enhanced_loss=True,  # Switch between enhanced and minimal loss
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.current_loss_weight = current_loss_weight
        self.voltage_loss_weight = voltage_loss_weight
        self.monotonicity_loss_weight = monotonicity_loss_weight
        self.smoothness_loss_weight = smoothness_loss_weight
        self.endpoint_loss_weight = endpoint_loss_weight
        self.use_enhanced_loss = use_enhanced_loss

    def train_one_epoch(self, train_loader):
        self.model.train()
        epoch_total_loss = 0.0
        epoch_current_loss = 0.0
        epoch_voltage_loss = 0.0
        epoch_physics_loss = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            # Assuming batch_data from DataLoader is (physical, target_currents, target_voltages, ...)
            # The preprocess_fixed_length_dual_output returns (X, Y_current, V_voltage, isc_vals) for train/test
            physical_params = batch_data[0].to(self.device)
            target_currents = batch_data[1].to(self.device)
            target_voltages = batch_data[2].to(self.device)

            # Combine targets into pairs: (batch, seq_len, 2)
            target_pairs = torch.stack((target_currents, target_voltages), dim=-1)

            # For this fixed-length model, we might not need a sophisticated mask
            # if preprocess_fixed_length_dual_output ensures all sequences are full.
            # If padding could exist, a mask would be needed. Assuming no padding for now.
            target_mask = None  # Or create one if your preprocessing might pad

            self.optimizer.zero_grad()

            # Model expects target_pairs for teacher forcing
            predicted_pairs = self.model(
                physical_params, target_pairs=target_pairs, target_mask=target_mask
            )

            if self.use_enhanced_loss:
                loss_outputs = enhanced_dual_output_loss(
                    predicted_pairs,
                    target_pairs,
                    target_mask=target_mask,
                    current_weight=self.current_loss_weight,
                    voltage_weight=self.voltage_loss_weight,
                    monotonicity_weight=self.monotonicity_loss_weight,
                    smoothness_weight=self.smoothness_loss_weight,
                    endpoint_weight=self.endpoint_loss_weight,
                )
                loss, current_l, voltage_l = loss_outputs[:3]
                physics_l = sum(loss_outputs[3:])  # Sum all physics components
            else:
                loss, current_l, voltage_l = minimal_dual_output_loss(
                    predicted_pairs,
                    target_pairs,
                    target_mask=target_mask,
                    current_weight=self.current_loss_weight,
                    voltage_weight=self.voltage_loss_weight,
                    monotonicity_weight=self.monotonicity_loss_weight,
                )
                physics_l = torch.tensor(0.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_total_loss += loss.item()
            epoch_current_loss += current_l.item()
            epoch_voltage_loss += voltage_l.item()
            epoch_physics_loss += (
                physics_l.item() if isinstance(physics_l, torch.Tensor) else physics_l
            )

        num_batches = len(train_loader)
        return {
            "total": epoch_total_loss / num_batches,
            "current": epoch_current_loss / num_batches,
            "voltage": epoch_voltage_loss / num_batches,
            "physics": epoch_physics_loss / num_batches,
        }

    def validate_one_epoch(self, val_loader):
        self.model.eval()
        epoch_total_loss = 0.0

        with torch.no_grad():
            for batch_data in val_loader:
                physical_params = batch_data[0].to(self.device)
                target_currents = batch_data[1].to(self.device)
                target_voltages = batch_data[2].to(self.device)
                target_pairs = torch.stack((target_currents, target_voltages), dim=-1)
                target_mask = None  # Assuming no padding

                # For validation, we can still use teacher forcing to get comparable loss values
                # Or, run full autoregressive generation and compare, but that's more for evaluation.
                # Here, using teacher-forced path for validation loss.
                predicted_pairs = self.model(
                    physical_params, target_pairs=target_pairs, target_mask=target_mask
                )

                loss, _, _ = minimal_dual_output_loss(
                    predicted_pairs,
                    target_pairs,
                    target_mask=target_mask,
                    current_weight=self.current_loss_weight,
                    voltage_weight=self.voltage_loss_weight,
                    monotonicity_weight=self.monotonicity_loss_weight,
                )
                epoch_total_loss += loss.item()

        return epoch_total_loss / len(val_loader)

    def evaluate(self, test_loader, scalers):
        self.model.eval()
        all_r2_scores = []
        plotted_sample_data = []  # For storing data for plotting

        # scalers should be (current_scaler, input_scaler, voltage_scaler)
        # from preprocess_fixed_length_dual_output
        if len(scalers) != 3:
            raise ValueError("Evaluation expects 3 scalers: current, input, voltage.")

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader):
                physical_params = batch_data[0].to(
                    self.device
                )  # Already on device from main loop
                # True targets are needed for comparison, get them unscaled
                true_target_currents_scaled = (
                    batch_data[1].cpu().numpy()
                )  # For unscaling
                true_target_voltages_scaled = (
                    batch_data[2].cpu().numpy()
                )  # For unscaling

                # Generate curves (autoregressive)
                # gen_current_curves_unscaled and gen_voltage_curves_unscaled are lists of np arrays
                gen_current_curves_unscaled, gen_voltage_curves_unscaled = (
                    self.model.generate_curve_batch(
                        physical_params, scalers, self.device
                    )
                )

                batch_s = physical_params.size(0)
                for i in range(batch_s):
                    pred_v = gen_voltage_curves_unscaled[i]
                    pred_c = gen_current_curves_unscaled[i]

                    # Unscale true targets for this sample
                    true_c_sample_unscaled = (
                        scalers[0]
                        .inverse_transform(
                            true_target_currents_scaled[i].reshape(-1, 1)
                        )
                        .flatten()
                    )
                    true_v_sample_unscaled = (
                        scalers[2]
                        .inverse_transform(
                            true_target_voltages_scaled[i].reshape(-1, 1)
                        )
                        .flatten()
                    )

                    r2 = calculate_interpolation_r2_standalone(
                        pred_v, pred_c, true_v_sample_unscaled, true_c_sample_unscaled
                    )
                    if not np.isnan(r2):
                        all_r2_scores.append(r2)

                    # Store a few samples for plotting
                    if (
                        len(plotted_sample_data) < 4
                    ):  # Store e.g., first 4 from first batch
                        if batch_idx == 0:
                            plotted_sample_data.append(
                                {
                                    "pred_voltage": pred_v,
                                    "pred_current": pred_c,
                                    "true_voltage": true_v_sample_unscaled,
                                    "true_current": true_c_sample_unscaled,
                                    "r2": r2,
                                }
                            )

        mean_r2 = np.mean(all_r2_scores) if all_r2_scores else float("nan")
        print(
            f"\nEvaluation Mean R2 Score (Interpolated): {mean_r2:.4f} ({len(all_r2_scores)} valid samples)"
        )

        if plotted_sample_data:
            self.plot_results(plotted_sample_data)

        return mean_r2, plotted_sample_data

    def plot_results(self, sample_data_list, num_samples_to_plot=4):
        # sample_data_list contains dicts as prepared in evaluate method

        num_to_plot = min(len(sample_data_list), num_samples_to_plot)
        if num_to_plot == 0:
            print("No samples to plot for dual output trainer.")
            return

        fig, axes = plt.subplots(
            num_to_plot, 1, figsize=(10, 3 * num_to_plot), squeeze=False
        )

        for i in range(num_to_plot):
            data = sample_data_list[i]
            ax = axes[i, 0]

            ax.plot(
                data["pred_voltage"],
                data["pred_current"],
                label="Predicted",
                color="blue",
                marker="o",
                linestyle="-",
                markersize=3,
            )
            ax.plot(
                data["true_voltage"],
                data["true_current"],
                label="True",
                color="orange",
                marker="x",
                linestyle="--",
                markersize=3,
            )

            ax.set_title(f"Sample - R²: {data.get('r2', float('nan')):.4f}")
            ax.set_xlabel("Voltage (V)")
            ax.set_ylabel("Current (A)")
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig("dual_output_transformer_results.png", dpi=150)
        print("Results plot saved to dual_output_transformer_results.png")
        # plt.show() # Optionally show plot
