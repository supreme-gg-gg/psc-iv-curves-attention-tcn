import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader


class ModelTrainer:
    def __init__(self, model: nn.Module, config):
        """
        ModelTrainer for TCN/TCAN/DINA models.
        Pass the appropriate config class (TCNConfig, TCANConfig, DINAConfig).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.TRAIN_CONFIG.LEARNING_RATE
        )
        self.scheduler = None
        self.history = {"train_loss": [], "val_loss": []}
        self.best_model_state_dict = None
        self.config = config

    def _get_model_output(self, *args, **kwargs):
        """
        Helper to handle models that either return (y_pred, attn) or just y_pred.
        NOTE: This is important because some models (due to analysis purposes) return more stuff than just the outpu
        Returns (y_pred, attn) where attn is None if not present.
        """
        output = self.model(*args, **kwargs)
        if isinstance(output, tuple) and len(output) == 2:
            return output
        else:
            return output, None

    def train(self, dataloaders: dict, epochs: int):
        train_loader, val_loader = dataloaders["train"], dataloaders["val"]
        print(f"=== Starting Training on {self.device} for {epochs} epochs ===")
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs * len(train_loader)
        )
        best_val_loss = float("inf")
        for epoch in range(epochs):
            self.model.train()
            train_loss = self._run_epoch(train_loader, is_training=True)
            self.history["train_loss"].append(train_loss)
            self.model.eval()
            with torch.no_grad():
                val_loss = self._run_epoch(val_loader, is_training=False)
            self.history["val_loss"].append(val_loss)
            print(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model_state_dict = self.model.state_dict()
                print(f"  -> New best validation loss: {best_val_loss:.6f}")
        self.plot_training_history()

    def evaluate(self, test_loader: DataLoader):
        print("=== Evaluating Model on Test Set ===")
        if self.best_model_state_dict:
            self.model.load_state_dict(self.best_model_state_dict)
        self.model.eval()
        y_true_flat, y_pred_flat = [], []
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (tuple, list)) and len(batch) == 6:
                    X_params, y_scaled, v_grid, _, orig_len, iscs = batch
                    X_params = X_params.to(self.device)
                    y_scaled = y_scaled.to(self.device)
                    v_grid = v_grid.to(self.device)
                    orig_len = orig_len.to(self.device)
                    iscs = iscs.to(self.device)
                else:
                    # fallback for unexpected batch structure
                    continue
                y_pred_scaled, _ = self._get_model_output(X_params, v_grid)
                if isinstance(y_pred_scaled, tuple):
                    y_pred_scaled = y_pred_scaled[0]
                for i in range(y_scaled.shape[0]):
                    L, isc = orig_len[i].item(), iscs[i].item()
                    y_true_flat.extend(0.5 * (y_scaled[i, :L].cpu() + 1) * isc)
                    y_pred_flat.extend(0.5 * (y_pred_scaled[i, :L].cpu() + 1) * isc)
        metrics = {
            "MAE": mean_absolute_error(y_true_flat, y_pred_flat),
            "RMSE": np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)),
            "R2_Score": r2_score(y_true_flat, y_pred_flat),
        }
        print(f"Evaluation Metrics: {metrics}")
        return metrics

    def _run_epoch(self, data_loader, is_training):
        total_loss, num_batches = 0.0, 0
        for batch in data_loader:
            X_params, y_scaled, v_grid, mask, orig_len, _ = (
                d.to(self.device) for d in batch
            )
            if is_training:
                self.optimizer.zero_grad()
            y_pred_scaled, _ = self._get_model_output(X_params, v_grid)
            loss = self._calculate_combined_loss(
                y_pred_scaled, y_scaled, mask, orig_len
            )
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, device=self.device)
            if torch.isnan(loss):
                print("NaN loss detected. Skipping batch update.")
                continue
            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
            total_loss += loss.item() if isinstance(loss, torch.Tensor) else float(loss)
            num_batches += 1
        if num_batches > 0:
            return total_loss / num_batches
        else:
            return torch.tensor(0.0, device=self.device)

    def _calculate_combined_loss(self, y_pred, y_true, mask, orig_len):
        mse = self._masked_mse_knee_weighted(y_pred, y_true, mask, orig_len)
        mono = self._monotonicity_penalty(y_pred, mask)
        curv = self._curvature_penalty(y_pred, mask)
        jsc, voc = self._jsc_voc_penalty(y_pred, y_true, mask, orig_len)
        loss_components = [mse, mono, curv, jsc, voc]
        weights = [
            self.config.LOSS_WEIGHTS.mse,
            self.config.LOSS_WEIGHTS.monotonicity,
            self.config.LOSS_WEIGHTS.curvature,
            self.config.LOSS_WEIGHTS.jsc,
            self.config.LOSS_WEIGHTS.voc,
        ]
        return sum(w * loss_val for w, loss_val in zip(weights, loss_components))

    def _masked_mse_knee_weighted(self, y_pred, y_true, mask, orig_len):
        se = F.mse_loss(y_pred, y_true, reduction="none")
        idx = torch.arange(mask.shape[1], device=self.device).unsqueeze(0)
        mpp_idx = (orig_len - 1).unsqueeze(1)
        window_mask = (idx >= mpp_idx - self.config.LOSS_WEIGHTS.knee_window_size) & (
            idx <= mpp_idx + self.config.LOSS_WEIGHTS.knee_window_size
        )
        knee_weights = (
            1.0
            + (self.config.LOSS_WEIGHTS.knee_weight_factor - 1.0) * window_mask.float()
        )
        return (se * mask * knee_weights).sum() / ((mask * knee_weights).sum() + 1e-7)

    def _monotonicity_penalty(self, y_pred, mask):
        diffs = y_pred[:, 1:] - y_pred[:, :-1]
        violations = F.relu(diffs) * (mask[:, 1:] * mask[:, :-1])
        return violations.square().sum() / ((mask[:, 1:] * mask[:, :-1]).sum() + 1e-7)

    def _curvature_penalty(self, y_pred, mask):
        curv_mask = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
        curvature = (y_pred[:, 2:] - 2.0 * y_pred[:, 1:-1] + y_pred[:, :-2]) * curv_mask
        return curvature.square().sum() / (curv_mask.sum() + 1e-7)

    def _jsc_voc_penalty(self, y_pred, y_true, mask, orig_len):
        jsc_mse = F.mse_loss(y_pred[:, 0] * mask[:, 0], y_true[:, 0] * mask[:, 0])
        last_indices = torch.clamp(orig_len - 1, 0, y_pred.shape[1] - 1).long()
        last_pred = torch.gather(y_pred, 1, last_indices.unsqueeze(1)).squeeze(-1)
        voc_target, last_mask = (
            -1.0 * torch.ones_like(last_pred),
            (orig_len > 0).float(),
        )
        voc_mse = F.mse_loss(last_pred * last_mask, voc_target * last_mask)
        return jsc_mse, voc_mse

    def plot_training_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training History")
        plt.yscale("log")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("training_history.png")
        plt.close()
        print("Training history plot saved to training_history.png")

    def plot_results(self, test_loader, n_samples=8):
        print(f"Plotting {n_samples} sample predictions...")
        if self.best_model_state_dict:
            self.model.load_state_dict(self.best_model_state_dict)
        self.model.eval()

        batch = next(iter(test_loader))
        if isinstance(batch, (tuple, list)) and len(batch) == 6:
            X_params, y_scaled, v_grid, _, orig_len, iscs = batch
            X_params = X_params.to(self.device)
            y_scaled = y_scaled.to(self.device)
            v_grid = v_grid.to(self.device)
            orig_len = orig_len.to(self.device)
            iscs = iscs.to(self.device)
        else:
            raise ValueError("Batch does not have expected 6 elements for plotting.")

        with torch.no_grad():
            y_pred_scaled, _ = self._get_model_output(X_params, v_grid)
            if isinstance(y_pred_scaled, tuple):
                y_pred_scaled = y_pred_scaled[0]
            y_pred_scaled = (
                y_pred_scaled.cpu() if hasattr(y_pred_scaled, "cpu") else y_pred_scaled
            )

        idxs = random.sample(range(len(X_params)), min(n_samples, len(X_params)))
        fig, axes = plt.subplots(
            len(idxs), 1, figsize=(8, 4 * len(idxs)), squeeze=False
        )

        for i, idx in enumerate(idxs):
            L, isc = orig_len[idx].item(), iscs[idx].item()
            y_true_phys = 0.5 * (y_scaled[idx, :L].cpu() + 1) * isc
            y_pred_phys = 0.5 * (y_pred_scaled[idx, :L].cpu() + 1) * isc
            v_plot = v_grid[idx, :L].cpu()

            ax = axes[i, 0]
            ax.plot(v_plot, y_true_phys, "k-", label="Ground Truth", lw=2)
            ax.plot(v_plot, y_pred_phys, "r--", label="Predicted", lw=2)
            ax.set(
                ylabel="Current Density (A/m^2)",
                xlabel="Voltage (V)",
                title=f"Test Sample Index {idx}",
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("prediction_examples.png", dpi=300)
        plt.close(fig)
        print("Prediction examples plot saved to prediction_examples.png")

    def plot_attention_maps(self, test_loader, n_samples=4):
        print(f"Plotting {n_samples} attention maps...")
        if self.best_model_state_dict:
            self.model.load_state_dict(self.best_model_state_dict)
        self.model.eval()

        batch = next(iter(test_loader))
        X_params, _, v_grid, _, orig_len, _ = (d.to(self.device) for d in batch)

        with torch.no_grad():
            _, attention_weights_list = self.model(X_params, v_grid)
        idxs = random.sample(range(len(X_params)), min(n_samples, len(X_params)))

        for i, idx in enumerate(idxs):
            L = orig_len[idx].item()
            fig, axes = plt.subplots(
                self.config.N_LAYERS,
                getattr(self.config, "N_HEADS", 1),
                figsize=(16, 8),
                squeeze=False,
            )
            fig.suptitle(
                f"Attention Maps for Test Sample {idx} (Seq Len: {L})", fontsize=16
            )

            for layer_idx, layer_attns in enumerate(attention_weights_list):
                for head_idx in range(getattr(self.config, "N_HEADS", 1)):
                    ax = axes[layer_idx, head_idx]
                    attn_map = layer_attns[idx, head_idx, :L, :L].cpu().numpy()
                    im = ax.imshow(attn_map, cmap="viridis", interpolation="nearest")
                    ax.set_title(f"Layer {layer_idx + 1}, Head {head_idx + 1}")
                    ax.set_xlabel("Attends To (Key Position)")
                    ax.set_ylabel("Query Position")

            fig.tight_layout(rect=(0, 0.03, 1, 0.95))
            plt.savefig(f"attention_map_sample_{idx}.png", dpi=200)
            plt.close(fig)

        print("Attention map plots saved.")

    def plot_curves_from_params(
        self, dataloader, n_samples=8, filename="curves_from_params.png"
    ):
        """
        Generate and plot predicted curves from a dataloader, plotting both predicted and ground truth curves for each sample (no truncation, just full arrays).
        Plots up to n_samples from the first batch.
        """
        self.model.eval()
        batch = next(iter(dataloader))
        # Accepts (X_params, y_scaled, v_grid, _, _, iscs)
        X_params, y_scaled, v_grid, _, _, iscs = (d.to(self.device) for d in batch)

        with torch.no_grad():
            y_pred_scaled, _ = self._get_model_output(X_params, v_grid)
            if isinstance(y_pred_scaled, tuple):
                y_pred_scaled = y_pred_scaled[0]
            y_pred_scaled = y_pred_scaled.cpu()

        y_scaled = y_scaled.cpu()
        v_grid = v_grid.cpu()
        iscs = iscs.cpu()
        idxs = list(range(min(n_samples, X_params.shape[0])))
        fig, axes = plt.subplots(
            len(idxs), 1, figsize=(8, 4 * len(idxs)), squeeze=False
        )

        for i, idx in enumerate(idxs):
            v_plot = v_grid[idx]
            y_pred = y_pred_scaled[idx]
            y_true = y_scaled[idx]
            isc = iscs[idx].item()
            y_true_unscaled = 0.5 * (y_true + 1) * isc
            y_pred_unscaled = 0.5 * (y_pred + 1) * isc
            ax = axes[i, 0]
            ax.plot(
                v_plot, y_true_unscaled, "k-", label="Ground Truth (unscaled)", lw=2
            )
            ax.plot(
                v_plot, y_pred_unscaled, "b--", label="Predicted Curve (unscaled)", lw=2
            )
            ax.set(
                ylabel="Current (A/m^2)",
                xlabel="Voltage (V)",
                title=f"Predicted vs Ground Truth for Sample {idx}",
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close(fig)
        print(f"Predicted curves plot saved to {filename}")

    def save_model(self, filename="model.pth"):
        """
        Save the model's state_dict to a file.
        """
        torch.save(self.model.state_dict(), filename)
        print(f"Model weights saved to {filename}")

    def load_model(self, filename="model.pth"):
        """
        Load the model's state_dict from a file.
        """
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        print(f"Model weights loaded from {filename}")
