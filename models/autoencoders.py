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