"""
General shared interface for training sequence models.
"""
import torch
import numpy as np
from sklearn.metrics import r2_score
from typing import Dict, Any, Tuple, Type
from src.models.seq_model_base import SeqModelBase

class SeqModelTrainer:
    def __init__(self, model: SeqModelBase, optimizer, scheduler=None, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_one_epoch(self, train_loader):
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
            )
            loss = self.compute_loss(outputs, padded_seq, mask)
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
                physical, padded_seq, mask, lengths = batch
                physical = physical.to(self.device)
                padded_seq = padded_seq.to(self.device)
                mask = mask.to(self.device)
                
                outputs = self.model(
                    physical,
                    target_seq=padded_seq,
                    lengths=lengths,
                )
                loss = self.compute_loss(outputs, padded_seq, mask)
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

    def evaluate(self, test_loader, scalers, include_plots=True):
        """Evaluate model performance."""
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
        print(f"- Length mismatches: {length_mismatch_count}/{total_samples} sequences")

        if include_plots:
            try:
                self.plot_generated_curves(sample_data, num_samples_to_plot=4)
            except Exception as e:
                print(f"Plotting failed: {e}")

        return mean_r2, sample_data
    
    @staticmethod
    def plot_generated_curves(sample_data, num_samples_to_plot=4):
        """Plot generated curves against true curves."""
        import matplotlib.pyplot as plt

        num_samples = min(num_samples_to_plot, len(sample_data))
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
