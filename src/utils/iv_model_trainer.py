"""
Unified trainer for all IV models accepting any loss function.
"""

import torch
import numpy as np
from sklearn.metrics import r2_score
from typing import Dict, Any, Tuple, Type, Callable, List
from iv_model_base import IVModelBase
import random
import matplotlib.pyplot as plt


class IVModelTrainer:
    """
    Unified trainer for all IV models accepting any loss function.
    """

    def __init__(
        self,
        model: IVModelBase,
        optimizer,
        scheduler=None,
        device: str = "cpu",
        loss_function: Callable = None,
        loss_params: dict = None,
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
        model_class: Type[IVModelBase], model_path: str, device: str = "cpu"
    ):
        """Load model through the model class's load_model implementation."""
        return model_class.load_model(model_path, device)
