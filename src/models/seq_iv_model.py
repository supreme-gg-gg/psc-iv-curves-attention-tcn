import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
from sklearn.metrics import r2_score
from utils.preprocess import preprocess_data

class SeqIVModel(nn.Module):
    def __init__(self, physical_dim, hidden_dim, num_layers=1, dropout=0.2):
        super(SeqIVModel, self).__init__()
        self.physical_dim = physical_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Map physical features to initial hidden state
        self.physical_enc = nn.Linear(physical_dim, hidden_dim)
        # LSTM for sequence generation; input size is 1 (current value)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        # Final projection to output one value per time step
        self.fc = nn.Linear(hidden_dim, 1)
        # Learnable start token (scalar)
        self.init_input = nn.Parameter(torch.zeros(1))

    def init_hidden(self, physical):
        h0 = torch.tanh(self.physical_enc(physical))
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h0

    def init_cell(self, physical):
        # Initialize cell state as zeros
        return torch.zeros(self.num_layers, physical.size(0), self.hidden_dim, device=physical.device)

    def forward(self, physical, target_seq=None, lengths=None, teacher_forcing_ratio=0.5):
        """
        Decodes an IV curve from physical features.
        If target_seq is provided, uses teacher forcing with the given ratio.
        Otherwise, generates the sequence in an auto-regressive manner.
        """
        batch_size = physical.size(0)
        if target_seq is not None:
            max_len = target_seq.size(1)
        elif lengths is not None:
            max_len = max(lengths)
        else:
            max_len = 100  # default maximum length

        device = physical.device
        hidden = self.init_hidden(physical)
        cell = self.init_cell(physical)
        # Expand the learnable start token to serve as initial input; shape: (batch, 1, 1)
        input_token = self.init_input.expand(batch_size, 1).unsqueeze(1)
        outputs = []

        for t in range(max_len):
            out, (hidden, cell) = self.lstm(input_token, (hidden, cell))  # out: (batch, 1, hidden_dim)
            pred = self.fc(out.squeeze(1))  # shape: (batch, 1)
            outputs.append(pred)
            # Decide whether to use teacher forcing
            use_teacher = (target_seq is not None) and (random.random() < teacher_forcing_ratio)
            if use_teacher:
                next_input = target_seq[:, t].unsqueeze(1)  # shape: (batch, 1)
            else:
                next_input = pred
            input_token = next_input.unsqueeze(1)  # shape: (batch, 1, 1)
        outputs = torch.cat(outputs, dim=1)  # shape: (batch, max_len)
        return outputs

def compute_loss(outputs, targets, lengths):
    """
    Computes mean squared error loss over valid time steps based on each sequence length.
    """
    loss = 0.0
    for i, l in enumerate(lengths):
        loss += nn.functional.mse_loss(outputs[i, :l], targets[i, :l])
    return loss / len(lengths)

def train_model_epoch(model, train_loader, optimizer, device, teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        physical, padded_seq, lengths = batch
        physical = physical.to(device)
        padded_seq = padded_seq.to(device)
        optimizer.zero_grad()
        outputs = model(physical, target_seq=padded_seq, lengths=lengths, teacher_forcing_ratio=teacher_forcing_ratio)
        loss = compute_loss(outputs, padded_seq, lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * physical.size(0)
    return total_loss / len(train_loader.dataset)

def load_trained_model(model_path, device):
    """
    Load a trained sequence model with its scalers and parameters.
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model onto
    
    Returns:
        model: Loaded model
        scalers: Tuple of (input_scaler, output_scaler)
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with saved parameters
    params = checkpoint['params']
    model = SeqIVModel(
        physical_dim=params['physical_dim'],
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['scalers']

def generate_curve(model, physical_input, scalers, device, max_length=50):
    """
    Generate an IV curve from physical parameters and convert back to original scale.
    
    Args:
        model: The trained model
        physical_input: Tensor of shape (1, physical_dim) containing scaled physical parameters
        scalers: Tuple of (input_scaler, output_scaler)
        device: Device to run inference on
        max_length: Maximum sequence length to generate
        
    Returns:
        Tuple of (generated_curve, valid_length), where generated_curve is in original scale
        and valid_length is the number of valid points (before padding)
    """
    model.eval()
    _, output_scaler = scalers
    
    with torch.no_grad():
        # Move input to device if needed
        if physical_input.device != device:
            physical_input = physical_input.to(device)
            
        # Generate sequence
        outputs = model(physical_input, target_seq=None,
                       lengths=[max_length], teacher_forcing_ratio=0.0)
        
        # Get the generated sequence (remove batch dimension)
        generated = outputs[0].cpu().numpy()
        
        # Find the first negative value (if any) to determine valid length
        neg_indices = np.where(generated < 0)[0]
        valid_length = len(generated) if len(neg_indices) == 0 else neg_indices[0] + 1
        
        # Inverse transform scaling
        valid_seq = generated[:valid_length]
        unscaled = output_scaler.inverse_transform(valid_seq.reshape(-1, 1)).flatten()
        
        return unscaled, valid_length

def test_model_epoch(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            physical, padded_seq, lengths = batch
            physical = physical.to(device)
            padded_seq = padded_seq.to(device)
            outputs = model(physical, target_seq=padded_seq, lengths=lengths, teacher_forcing_ratio=0.0)
            loss = compute_loss(outputs, padded_seq, lengths)
            total_loss += loss.item() * physical.size(0)
    return total_loss / len(test_loader.dataset)

def evaluate_model(model, test_loader, scalers, device, num_samples=4):
    """
    Evaluate model on test set, computing R² scores and generating visualization.
    Returns mean R² score and saves visualization of sample curves.
    """
    model.eval()
    r2_scores = []
    sample_curves = []
    sample_targets = []
    sample_lengths = []
    
    with torch.no_grad():
        for batch_idx, (physical, padded_seq, lengths) in enumerate(test_loader):
            physical = physical.to(device)
            padded_seq = padded_seq.to(device)
            
            # Generate curves
            for i in range(min(physical.size(0), num_samples - len(sample_curves))):
                generated_curve, valid_len = generate_curve(
                    model, physical[i:i+1], scalers, device, max_length=lengths[i].item()
                )
                target_curve = padded_seq[i, :lengths[i]].cpu().numpy()
                
                # Inverse transform target
                _, output_scaler = scalers
                target_unscaled = output_scaler.inverse_transform(target_curve.reshape(-1, 1)).flatten()
                
                print(len(target_unscaled), len(generated_curve))

                # Compute R² score
                r2 = r2_score(target_unscaled, generated_curve[:len(target_unscaled)])
                r2_scores.append(r2)
                
                if len(sample_curves) < num_samples:
                    sample_curves.append(generated_curve)
                    sample_targets.append(target_unscaled)
                    sample_lengths.append(valid_len)
            
            if len(sample_curves) >= num_samples:
                break
    
    # Plot sample curves
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    for i in range(min(len(sample_curves), 4)):
        ax = fig.add_subplot(gs[i//2, i%2])
        ax.plot(sample_targets[i], 'b-', label='True', alpha=0.7)
        ax.plot(sample_curves[i][:len(sample_targets[i])], 'r--', label='Generated', alpha=0.7)
        ax.set_title(f'Sample {i+1} (R²={r2_scores[i]:.4f})')
        ax.set_xlabel('Index')
        ax.set_ylabel('Current Density (A/m²)')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('seq_model_samples.png')
    plt.close()
    
    mean_r2 = np.mean(r2_scores)
    print(f"\nModel Evaluation:")
    print(f"Mean R² Score: {mean_r2:.4f}")
    print(f"Sample curves saved to seq_model_samples.png")
    
    return mean_r2
    
if __name__ == "__main__":
    # Hyperparameters
    physical_dim = 31
    hidden_dim = 64
    num_layers = 1
    dropout = 0.2
    learning_rate = 1e-3
    batch_size = 32
    epochs = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define input and output file paths for the dataset
    input_paths = [
        "dataset/Data_10k_sets/Data_10k_rng1/LHS_parameters_m.txt",
        "dataset/Data_10k_sets/Data_10k_rng2/LHS_parameters_m.txt",
        "dataset/Data_10k_sets/Data_10k_rng3/LHS_parameters_m.txt"
    ]

    output_paths = [
        "dataset/Data_10k_sets/Data_10k_rng1/iV_m.txt",
        "dataset/Data_10k_sets/Data_10k_rng2/iV_m.txt",
        "dataset/Data_10k_sets/Data_10k_rng3/iV_m.txt"
    ]

    # Preprocess data using the new sequential function with masks
    data = preprocess_data(input_paths, output_paths, return_masks=True)
    X_train, padded_y_train, mask_train = data['train']
    X_test, padded_y_test, mask_test = data['test']

    # Get lengths from masks for printing info
    lengths_train = mask_train.sum(dim=1).int().tolist()
    lengths_test = mask_test.sum(dim=1).int().tolist()

    print("\nDataset information:")
    print(f"Training set: {len(lengths_train)} samples")
    print(f"Test set: {len(lengths_test)} samples")
    print(f"Sequence lengths - Train: min={min(lengths_train)}, max={max(lengths_train)}")
    print(f"Sequence lengths - Test: min={min(lengths_test)}, max={max(lengths_test)}")

    # Create TensorDatasets with masks
    train_dataset = TensorDataset(X_train, padded_y_train, torch.tensor(lengths_train), mask_train)
    test_dataset = TensorDataset(X_test, padded_y_test, torch.tensor(lengths_test), mask_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("\nInitializing and training model...")
    model = SeqIVModel(physical_dim, hidden_dim, num_layers, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train_loss = train_model_epoch(model, train_loader, optimizer, device, teacher_forcing_ratio=0.5)
        test_loss = test_model_epoch(model, test_loader, device)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

    save_dict = {
        'model_state_dict': model.state_dict(),
        'scalers': data['scalers'],
        'params': {
            'physical_dim': physical_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout
        }
    }
    torch.save(save_dict, 'seq_iv_model.pth')
    print("\nModel saved to seq_iv_model.pth")