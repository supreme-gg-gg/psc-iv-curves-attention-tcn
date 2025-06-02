"""
General shared interface for training sequence models.
"""
import torch
import numpy as np
from sklearn.metrics import r2_score
from typing import Dict, Any, Tuple, Type
from src.models.seq_model_base import SeqModelBase
import random

class SeqModelTrainer:
    def __init__(self, model: SeqModelBase, optimizer, scheduler=None, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_one_epoch(self, train_loader, teacher_forcing_ratio=0.5):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        for batch in train_loader:
            physical, padded_seq, mask, lengths = batch
            physical = physical.to(self.device)
            padded_seq = padded_seq.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(
                physical,
                target_seq=padded_seq,
                lengths=lengths,
                teacher_forcing_ratio=teacher_forcing_ratio
            )
            loss = self.compute_shape_aware_loss(outputs, padded_seq, mask)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * physical.size(0)
        return total_loss / len(train_loader.dataset)

    def validate_one_epoch(self, val_loader, teacher_forcing_ratio=0.0):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                physical, padded_seq, mask, lengths = batch
                physical = physical.to(self.device)
                padded_seq = padded_seq.to(self.device)
                mask = mask.to(self.device)
                
                outputs = self.model(
                    physical,
                    target_seq=padded_seq,
                    lengths=lengths,
                    teacher_forcing_ratio=teacher_forcing_ratio
                )
                loss = self.compute_shape_aware_loss(outputs, padded_seq, mask)
                total_loss += loss.item() * physical.size(0)
        return total_loss / len(val_loader.dataset)

    @staticmethod
    def compute_loss(outputs, targets, mask):
        """
        Simple masked MSE loss - predicts the curve values directly.
        
        Args:
            outputs: Model predictions (batch_size, seq_len)
            targets: Target values (batch_size, seq_len)
            mask: Binary mask for valid positions (batch_size, seq_len)
        """
        # Apply mask and compute MSE only on valid positions
        mse = ((outputs - targets) ** 2) * mask
        return mse.sum() / mask.sum()

    @staticmethod
    def compute_shape_aware_loss(outputs, targets, mask, lambda_shape=0.5, mse_weight=1.0):
        """
        Computes a shape-aware loss.

        Args:
            outputs: Model predictions (batch_size, seq_len)
            targets: Target values (batch_size, seq_len)
            mask: Binary mask for valid positions (batch_size, seq_len)
            lambda_shape: Weighting factor for the shape (derivative) loss component.
            mse_weight: Weighting factor for the main MSE loss component.
        """
        pointwise_mse = ((outputs - targets) ** 2) * mask
        mse_term = pointwise_mse.sum() / mask.sum()

        # Differential Loss (masked) - First Derivative
        shape_term = torch.tensor(0.0, device=outputs.device, requires_grad=outputs.requires_grad)

        # Ensure there's at least 2 points to compute difference for shape loss
        # and that the mask allows for it.
        if outputs.size(1) > 1 and targets.size(1) > 1:
            # Calculate derivatives (differences between adjacent points)
            # outputs/targets shape: (batch_size, seq_len)
            true_diff = targets[:, 1:] - targets[:, :-1]
            pred_diff = outputs[:, 1:] - outputs[:, :-1]

            # Mask for the differences: valid if both points in the original pair were valid.
            # mask shape: (batch_size, seq_len)
            # diff_mask will have shape (batch_size, seq_len - 1)
            diff_mask = mask[:, 1:] * mask[:, :-1] # Ensures both points forming the difference are valid

            diff_mask_sum = diff_mask.sum()
            if diff_mask_sum > 0: # Only compute shape loss if there are valid differences
                shape_mse = ((pred_diff - true_diff) ** 2) * diff_mask
                shape_term = shape_mse.sum() / diff_mask_sum
            # If diff_mask_sum is 0, shape_term remains 0.0 as initialized.

        # Total loss
        # You might want to log mse_term and shape_term separately during training
        # to see how each component is behaving.
        total_loss = mse_weight * mse_term + lambda_shape * shape_term
        return total_loss # Or return all three for logging: total_loss, mse_term, shape_term

    def evaluate(self, test_loader, scalers, include_plots=True):
        """
        Evaluate model performance.
        This returns the mean R² score across all samples and optionally plots some generated curves.
        """
        self.model.eval()
        all_r2_scores = []
        sample_data = []
        total_samples = 0
        length_mismatch_count = 0

        with torch.no_grad():
            for batch in test_loader:
                physical, padded_seq, mask, lengths = batch
                physical = physical.to(self.device)
                generated_curves, gen_lengths = self.model.generate_curve_batch(
                    physical, scalers, self.device
                )
                _, output_scaler = scalers

                for i in range(len(lengths)):
                    total_samples += 1
                    true_len = lengths[i].item() if isinstance(lengths[i], torch.Tensor) else lengths[i]
                    true_curve = padded_seq[i, :true_len].cpu().numpy()
                    true_unscaled = output_scaler.inverse_transform(true_curve.reshape(-1, 1)).flatten()
                    gen_curve = generated_curves[i]

                    if len(gen_curve) != len(true_unscaled):
                        length_mismatch_count += 1

                    min_len = min(len(true_unscaled), len(gen_curve))
                    if min_len > 1:
                        r2 = r2_score(true_unscaled[:min_len], gen_curve[:min_len])
                        all_r2_scores.append(r2)
                        sample_data.append((gen_curve, true_unscaled, r2))

        mean_r2 = np.mean(all_r2_scores) if all_r2_scores else float('nan')
        print(f"\nModel Evaluation Summary:")
        print(f"- Mean R² Score: {mean_r2:.4f} ({len(all_r2_scores)} valid samples)")

        # print number of samples with negative R² scores
        negative_r2_count = sum(1 for r2 in all_r2_scores if r2 < 0)
        print(f"- Negative R² Scores: {negative_r2_count} samples")
        print(f"- Length mismatches: {length_mismatch_count}/{total_samples} sequences")

        if include_plots:
            try:
                self.plot_generated_curves(sample_data, num_samples_to_plot=4)
            except Exception:
                print("Error occurred while plotting curves.")

        return mean_r2, sample_data
    
    @staticmethod
    def plot_generated_curves(sample_data, num_samples_to_plot=4):
        """Plot generated curves against true curves."""
        import matplotlib.pyplot as plt

        # Random sampling of curves to plot
        if len(sample_data) > num_samples_to_plot:
            sample_data = random.sample(sample_data, num_samples_to_plot)

        num_samples = len(sample_data)
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2 * num_samples), sharex=True)

        for i in range(num_samples):
            gen_curve, true_curve, r2 = sample_data[i]
            ax = axes[i] if num_samples > 1 else axes
            ax.plot(gen_curve, label='Generated Curve', color='blue')
            ax.plot(true_curve, label='True Curve', color='orange')
            ax.set_title(f'Sample {i + 1} - R²: {r2:.4f}')
            ax.legend()
            ax.grid()

        plt.tight_layout()
        plt.savefig('generated_curves.png')
        plt.show()

    def save_model(self, save_path: str, scalers: Tuple, params: Dict[str, Any]):
        """Save model through its save_model implementation."""
        self.model.save_model(save_path, scalers, params)

    @staticmethod
    def load_model(model_class: Type[SeqModelBase], model_path: str, device: str = 'cpu'):
        """Load model through the model class's load_model implementation."""
        return model_class.load_model(model_path, device)
