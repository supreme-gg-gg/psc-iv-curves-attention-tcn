"""
General shared interface for training sequence models.
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score
from typing import Dict, Any, Tuple, Type
from src.models.iv_model_base import IVModelBase 
import random
import matplotlib.pyplot as plt
import numpy as np

class SeqModelTrainer:
    def __init__(self, model: IVModelBase, optimizer, scheduler=None, device='cpu', eos_loss_weight=1.0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        # weight for per-step EOS classifier loss
        self.eos_loss_weight = eos_loss_weight

    def train_one_epoch(self, train_loader, teacher_forcing_ratio=0.5):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        for batch in train_loader:
            # unpack batch; optionally contains eos_targets as 5th element
            if len(batch) == 5:
                physical, padded_seq, mask, lengths, eos_targets = batch
            else:
                physical, padded_seq, mask, lengths = batch
                eos_targets = None
            physical = physical.to(self.device)
            padded_seq = padded_seq.to(self.device)
            mask = mask.to(self.device)
            if eos_targets is not None:
                eos_targets = eos_targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(
                physical,
                target_seq=padded_seq,
                lengths=lengths,
                teacher_forcing_ratio=teacher_forcing_ratio
            )
            
            # unpack model outputs (seq_preds, eos_logits)
            if isinstance(outputs, tuple):
                seq_preds, eos_logits = outputs
            else:
                seq_preds = outputs
                eos_logits = None

            loss = self.compute_shape_aware_loss(seq_preds, padded_seq, mask)
            # per-step EOS loss
            if eos_logits is not None and eos_targets is not None and self.eos_loss_weight > 0.0:
                loss = loss + self.eos_loss_weight * self.compute_eos_bce_loss(eos_logits, eos_targets)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * physical.size(0)

        return total_loss / len(train_loader.dataset)

    def validate_one_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 5:
                    physical, padded_seq, mask, lengths, eos_targets = batch
                else:
                    physical, padded_seq, mask, lengths = batch
                    eos_targets = None
                physical = physical.to(self.device)
                padded_seq = padded_seq.to(self.device)
                mask = mask.to(self.device)
                if eos_targets is not None:
                    eos_targets = eos_targets.to(self.device)
                
                outputs = self.model(
                    physical,
                    target_seq=padded_seq,
                    lengths=lengths,
                    teacher_forcing_ratio=0.0
                )

                # Unpack model outputs (seq_preds, eos_logits)
                if isinstance(outputs, tuple):
                    seq_preds, eos_logits = outputs
                else:
                    seq_preds = outputs
                    eos_logits = None
                    
                loss = self.compute_shape_aware_loss(seq_preds, padded_seq, mask)
                
                # Add EOS loss if available
                if eos_logits is not None and eos_targets is not None and self.eos_loss_weight > 0.0:
                    loss = loss + self.eos_loss_weight * self.compute_eos_bce_loss(eos_logits, eos_targets)
                    
                total_loss += loss.item() * physical.size(0)
        
        return total_loss / len(val_loader.dataset)
    
    @staticmethod
    def compute_eos_bce_loss(eos_logits, eos_targets, gamma: float = 2.0):
        """
        Compute binary cross-entropy loss for EOS predictions.
        Automatically handles size mismatches by aligning targets to logits timesteps.
        
        Args:
            eos_logits: Raw logits from the model (batch_size, seq_len_logits)
            eos_targets: Binary targets for EOS (batch_size, seq_len_targets)
        
        Returns:
            BCE loss value.
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

    @staticmethod
    def compute_mse_loss(outputs, targets, mask):
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
            for batch_idx, batch in enumerate(test_loader):
                # During evaluation, we don't need EOS targets
                physical, padded_seq, mask, lengths, _ = batch
                physical = physical.to(self.device)
                
                # Get generated sequences with truncation-based lengths
                generated_curves, gen_lengths = self.model.generate_curve_batch(
                    physical, scalers, self.device
                )
                _, output_scaler = scalers

                # Evaluate using actual curve lengths
                for i in range(len(lengths)):
                    total_samples += 1

                    # Get true curve at true length (no padding)
                    true_len = lengths[i].item() if isinstance(lengths[i], torch.Tensor) else lengths[i]
                    true_curve = padded_seq[i, :true_len].cpu().numpy()
                    true_unscaled = output_scaler.inverse_transform(true_curve.reshape(-1, 1)).flatten()

                    # Get generated curve at generated length (already cropped by truncation logic)
                    gen_len = gen_lengths[i] if isinstance(gen_lengths[i], (int, float)) else int(gen_lengths[i])
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
                        
                        # Store original length curves for plotting
                        sample_data.append((gen_curve, true_unscaled, r2, gen_len, true_len))

        mean_r2 = np.mean(all_r2_scores) if all_r2_scores else float('nan')
        print(f"\nModel Evaluation Summary:")
        print(f"- Mean R² Score: {mean_r2:.4f} ({len(all_r2_scores)} valid samples)")
        
        negative_r2_count = sum(1 for r2 in all_r2_scores if r2 < 0)
        print(f"- Negative R² Scores: {negative_r2_count} samples")
        print(f"- Length mismatches: {length_mismatch_count}/{total_samples} sequences ({100*length_mismatch_count/total_samples:.1f}%)")

        if include_plots:
            try:
                self.plot_generated_curves(sample_data, num_samples_to_plot=4)
            except Exception as e:
                print(f"Error plotting generated curves: {e}")

        return mean_r2, sample_data
    
    @staticmethod
    def plot_generated_curves(sample_data, num_samples_to_plot=4):
        """Plot generated curves against true curves at their actual lengths."""
        # Random sampling of curves to plot
        if len(sample_data) > num_samples_to_plot:
            sample_data = random.sample(sample_data, num_samples_to_plot)

        num_samples = len(sample_data)
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2 * num_samples))
        if num_samples == 1:
            axes = [axes]

        Va = np.concatenate((np.arange(0, 0.41, 0.1), np.arange(0.425, 1.401, 0.025)))  # applied voltage, V

        for i in range(num_samples):
            gen_curve, true_curve, r2, gen_len, true_len = sample_data[i]
            ax = axes[i]
            
            # Plot each curve truncated to available Va length
            gen_n = min(len(gen_curve), len(Va))
            true_n = min(len(true_curve), len(Va))
            # Plot generated curve
            ax.plot(Va[:gen_n], gen_curve[:gen_n], label=f'Generated (len={gen_len})', color='blue', linewidth=2)
            # Plot true curve
            ax.plot(Va[:true_n], true_curve[:true_n], label=f'True (len={true_len})', color='orange', linewidth=2)
            # Ensure x-axis spans full longer of the two
            max_n = max(gen_n, true_n)
            ax.set_xlim(0, Va[max_n-1] if max_n > 0 else Va[0])
            
            ax.set_title(f'Sample {i + 1} - R²: {r2:.4f} | Gen: {gen_len}, True: {true_len}')
            ax.legend()
            ax.grid()
            ax.set_ylabel('Current (A)')

        axes[-1].set_xlabel('Voltage (V)')
        plt.tight_layout()
        plt.savefig('generated_curves.png', dpi=150, bbox_inches='tight')
        plt.show()

    def save_model(self, save_path: str, scalers: Tuple, params: Dict[str, Any]):
        """Save model through its save_model implementation."""
        self.model.save_model(save_path, scalers, params)

    @staticmethod
    def load_model(model_class: Type[IVModelBase], model_path: str, device: str = 'cpu'):
        """Load model through the model class's load_model implementation."""
        return model_class.load_model(model_path, device)
