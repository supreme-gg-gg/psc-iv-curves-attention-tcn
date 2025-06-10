import numpy as np
import torch
import torch.nn as nn
from iv_model_base import IVModelBase

class CVAEModel(IVModelBase):
    def __init__(self, physical_dim, va_sweep_dim, latent_dim, output_iv_dim):
        super(CVAEModel, self).__init__()
        self.physical_dim = physical_dim
        self.va_sweep_dim = va_sweep_dim  # Length of Va sweep
        self.latent_dim = latent_dim
        self.output_iv_dim = output_iv_dim  # Should equal va_sweep_dim

        # Encoder remains unchanged
        encoder_input_dim = physical_dim + va_sweep_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        # Updated decoder architecture with increased capacity and dropout
        decoder_input_dim = latent_dim + physical_dim
        self.decoder_hidden = nn.Sequential(
            nn.Linear(decoder_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # Updated curve prediction head with linear activation for better fitting
        self.decoder_curve = nn.Sequential(
            nn.Linear(64, output_iv_dim)
        )
        # Updated EOS head with an extra hidden layer for improved EOS prediction
        self.decoder_eos = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_iv_dim)
        )

    def encode(self, x_physical, y_iv_curve_data):
        # x_physical: (batch_size, physical_dim)
        # y_iv_curve_data: (batch_size, va_sweep_dim)
        combined_input = torch.cat((x_physical, y_iv_curve_data), dim=1)
        h = self.encoder(combined_input)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # sigma = exp(0.5 * log(sigma^2))
        eps = torch.randn_like(std)   # Sample epsilon from N(0, I)
        return mu + eps * std         # z = mu + epsilon * sigma

    def decode(self, z, x_physical):
        combined_input = torch.cat((z, x_physical), dim=1)
        h_dec = self.decoder_hidden(combined_input)
        curve = self.decoder_curve(h_dec)
        eos_logits = self.decoder_eos(h_dec)
        return curve, eos_logits

    def forward(self, physical, target_seq, lengths=None, teacher_forcing_ratio=None):
        """
        CVAE forward pass for training only.
        
        Args:
            physical: Physical parameters (batch_size, physical_dim)
            target_seq: Target IV curves for training (required for CVAE)
            lengths: Not used for CVAE (fixed length)
            teacher_forcing_ratio: Not used for CVAE
            
        Returns:
            (reconstructed_curves, eos_logits, mu, logvar)
        """
        # Encode-decode with target (training mode)
        mu, logvar = self.encode(physical, target_seq)
        z = self.reparameterize(mu, logvar)
        reconstructed, eos_logits = self.decode(z, physical)
        return (reconstructed, eos_logits, mu, logvar)

    def generate_curve_batch(self, physical_input, scalers, device):
        """
        Generate single curve per input using sampling from prior.
        """
        self.eval()
        _, output_scaler = scalers
        batch_size = physical_input.size(0)
        
        with torch.no_grad():
            # Sample from prior distribution for inference
            z = torch.randn(batch_size, self.latent_dim, device=physical_input.device)
            generated_curves, eos_logits = self.decode(z, physical_input)
            eos_probs = torch.sigmoid(eos_logits)
            
            # Process outputs and determine lengths using EOS
            outputs_cpu = generated_curves.detach().cpu().numpy()
            eos_cpu = eos_probs.detach().cpu().numpy()
            gen_curves = []
            lengths = []
            
            for i in range(batch_size):
                seq = outputs_cpu[i]
                eos = eos_cpu[i]
                
                # Find first position where EOS probability crosses threshold
                eos_positions = np.where(eos > 0.5)[0]
                if len(eos_positions) > 0:
                    length = int(eos_positions[0]) + 1
                else:
                    # If no EOS found, use full sequence
                    length = len(seq)
                
                lengths.append(length)
                
                # Convert to actual values
                unscaled = output_scaler.inverse_transform(seq[:length].reshape(-1, 1)).flatten()
                gen_curves.append(unscaled)
            
            return gen_curves, lengths

    def save_model(self, save_path, scalers, params):
        """Save model state, scalers, and parameters."""
        save_dict = {
            "model_state_dict": self.state_dict(),
            "scalers": scalers,
            "params": params
        }
        torch.save(save_dict, save_path)

    @classmethod
    def load_model(cls, model_path, device):
        """Load model from file."""
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        params = checkpoint["params"]
        model = cls(
            physical_dim=params["physical_dim"],
            va_sweep_dim=params["va_sweep_dim"],
            latent_dim=params["latent_dim"],
            output_iv_dim=params["output_iv_dim"]
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, checkpoint["scalers"], None
