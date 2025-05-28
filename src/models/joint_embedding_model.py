import torch
import torch.nn as nn

class InputEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(InputEncoder, self).__init__()
        # Since input_dim is 31, use smaller layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, emb_dim)
        )
    def forward(self, x):
        return self.net(x)

class InputDecoder(nn.Module):
    def __init__(self, emb_dim, input_dim):
        super(InputDecoder, self).__init__()
        # Mirror of encoder architecture
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, input_dim)
        )
    def forward(self, x):
        return self.net(x)

# Try making these Tanh instead idk?
class CurveEncoder(nn.Module):
    def __init__(self, curve_dim, emb_dim):
        super(CurveEncoder, self).__init__()
        # Since curve_dim is 45, use smaller layers
        self.net = nn.Sequential(
            nn.Linear(curve_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, emb_dim)
        )
    def forward(self, x):
        return self.net(x)

class CurveDecoder(nn.Module):
    def __init__(self, emb_dim, curve_dim):
        super(CurveDecoder, self).__init__()
        # Mirror of encoder architecture
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, curve_dim)
        )
    def forward(self, x):
        return self.net(x)

class JointEmbeddingModel(nn.Module):
    def __init__(self, input_dim, curve_dim, emb_dim):
        super(JointEmbeddingModel, self).__init__()
        self.input_encoder = InputEncoder(input_dim, emb_dim)
        self.input_decoder = InputDecoder(emb_dim, input_dim)
        self.curve_encoder = CurveEncoder(curve_dim, emb_dim)
        self.curve_decoder = CurveDecoder(emb_dim, curve_dim)
    
    def forward(self, input_data, curve_data):
        # Encode both inputs
        emb_input = self.input_encoder(input_data)
        emb_curve = self.curve_encoder(curve_data)
        
        # Decode from embeddings
        recon_input = self.input_decoder(emb_input)
        recon_curve = self.curve_decoder(emb_curve)
        
        # Cross-domain generation
        generated_curve_from_input = self.curve_decoder(emb_input)
        generated_input_from_curve = self.input_decoder(emb_curve)
        
        return {
            'emb_input': emb_input,
            'emb_curve': emb_curve,
            'recon_input': recon_input,
            'recon_curve': recon_curve,
            'generated_curve': generated_curve_from_input,
            'generated_input': generated_input_from_curve
        }
    
    def generate_curve(self, input_data):
        """Generate IV curve from input parameters"""
        self.eval()
        with torch.no_grad():
            emb_input = self.input_encoder(input_data)
            generated_curve = self.curve_decoder(emb_input)
            return generated_curve

    def save_autoencoders(self, path_prefix="checkpoints/ae"):
        """Save autoencoders separately"""
        import os
        os.makedirs(path_prefix, exist_ok=True)
        
        # Save input autoencoder
        torch.save({
            'encoder_state_dict': self.input_encoder.state_dict(),
            'decoder_state_dict': self.input_decoder.state_dict(),
        }, f"{path_prefix}/input_autoencoder.pth")
        
        # Save curve autoencoder
        torch.save({
            'encoder_state_dict': self.curve_encoder.state_dict(),
            'decoder_state_dict': self.curve_decoder.state_dict(),
        }, f"{path_prefix}/curve_autoencoder.pth")

    def load_autoencoders(self, path_prefix="checkpoints/ae"):
        """Load autoencoders separately"""
        # Load input autoencoder
        input_ae = torch.load(f"{path_prefix}/input_autoencoder.pth")
        self.input_encoder.load_state_dict(input_ae['encoder_state_dict'])
        self.input_decoder.load_state_dict(input_ae['decoder_state_dict'])
        
        # Load curve autoencoder
        curve_ae = torch.load(f"{path_prefix}/curve_autoencoder.pth")
        self.curve_encoder.load_state_dict(curve_ae['encoder_state_dict'])
        self.curve_decoder.load_state_dict(curve_ae['decoder_state_dict'])

    def save_full_model(self, path="checkpoints/jem/model.pth", **kwargs):
        """Save the full model with any additional data"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            **kwargs  # Additional data like scalers, config, etc.
        }
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)

    @classmethod
    def load_full_model(cls, path="checkpoints/jem/model.pth", **model_kwargs):
        """Load the full model"""
        checkpoint = torch.load(path)
        model = cls(**model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint