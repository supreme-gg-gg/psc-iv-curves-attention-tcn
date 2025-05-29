import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random, math
from sklearn.metrics import r2_score
from ..utils.preprocess import preprocess_data_no_eos

class SeqIVModel(nn.Module):
    def __init__(self, physical_dim, hidden_dim, num_layers=2, dropout=0.2, scaled_zero_threshold=0.0):
        super(SeqIVModel, self).__init__()
        self.physical_dim = physical_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.scaled_zero_threshold = scaled_zero_threshold

        # Enhanced physical features encoder with layer norm
        self.physical_enc = nn.Sequential(
            nn.Linear(physical_dim, hidden_dim * 2, bias=True),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim)
        )

        # Bidirectional LSTM with residual connections
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Will be used only during training
        )

        # Layer normalization for LSTM outputs
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)  # *2 for bidirectional

        # Enhanced current predictor with residual path
        self.current_projection = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        self.current_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=True)
        )

        # Learnable start token with better initialization
        self.init_input = nn.Parameter(torch.randn(1) * 0.02)

    def init_hidden(self, physical):
        h0 = torch.tanh(self.physical_enc(physical))
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h0

    def init_cell(self, physical):
        # Initialize cell state as zeros
        return torch.zeros(self.num_layers, physical.size(0), self.hidden_dim, device=physical.device)

    def forward(self, physical, target_seq=None, lengths=None, teacher_forcing_ratio=0.5):
        """
        Decodes an IV curve from physical features with enhanced architecture.
        During training: Uses bidirectional LSTM with teacher forcing
        During inference: Uses unidirectional generation with stopping at negative values
        
        Args:
            scaled_zero_threshold: The threshold value (in scaled space) corresponding to 0 current
        """
        batch_size = physical.size(0)
        if target_seq is not None:
            max_len = target_seq.size(1)
        elif lengths is not None:
            max_len = max(lengths)
        else:
            max_len = 100

        device = physical.device
        hidden = self.init_hidden(physical)
        cell = self.init_cell(physical)
        
        # Initialize with learned token
        input_token = self.init_input.expand(batch_size, 1).unsqueeze(1)
        current_outputs = []
        
        # Track finished sequences during inference
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(max_len):
            # Run LSTM step
            out, (hidden, cell) = self.lstm(input_token, (hidden, cell))
            
            # Apply layer normalization
            out = self.layer_norm(out.squeeze(1))
            
            # Project to prediction space with residual connection
            projected = self.current_projection(out)
            current_pred = self.current_head(projected)
            
            # Add residual connection for better gradient flow
            if t > 0:  # Skip residual for first prediction
                prev_pred = current_outputs[-1]
                # Weighted residual connection
                current_pred = 0.8 * current_pred + 0.2 * prev_pred
            
            current_outputs.append(current_pred)

            # During inference (no teacher forcing), check for scaled negative values
            if target_seq is None:
                # Mark sequences as finished if they predict below scaled zero threshold
                finished = finished | (current_pred < self.scaled_zero_threshold).squeeze()
                if finished.all():
                    break

            # Decide whether to use teacher forcing
            use_teacher = (target_seq is not None) and (random.random() < teacher_forcing_ratio)
            if use_teacher:
                next_input = target_seq[:, t].unsqueeze(1)
            else:
                next_input = current_pred
            input_token = next_input.unsqueeze(1)

        current_outputs = torch.cat(current_outputs, dim=1)
        return current_outputs

def compute_loss(current_outputs, current_targets, mask_or_lengths):
    """Enhanced loss function with:
    1. MSE loss weighted by position (more weight near sequence end)
    2. Additional loss term for prediction of negative transitions
    3. Smoothness regularization
    """
    device = current_outputs.device
    if isinstance(mask_or_lengths, torch.Tensor):
        mask = mask_or_lengths
        seq_lengths = mask.sum(dim=1)
    else:
        # Convert lengths to mask
        max_len = current_outputs.size(1)
        batch_size = current_outputs.size(0)
        mask = torch.zeros((batch_size, max_len), device=device)
        for i, l in enumerate(mask_or_lengths):
            mask[i, :l] = 1
        seq_lengths = torch.tensor(mask_or_lengths, device=device)

    # 1. Position-weighted MSE loss
    position_weights = torch.arange(current_outputs.size(1), device=device, dtype=torch.float32)
    position_weights = position_weights.unsqueeze(0) / seq_lengths.unsqueeze(1)
    position_weights = torch.exp(position_weights) * mask  # Exponential weighting
    mse_loss = ((current_outputs - current_targets) ** 2) * position_weights
    mse_loss = mse_loss.sum() / mask.sum()

    # 2. Transition prediction loss - encourage accurate negative value prediction
    transitions = (current_targets < 0) & (mask.bool())
    if transitions.any():
        transition_loss = nn.functional.mse_loss(
            current_outputs[transitions],
            current_targets[transitions]
        )
    else:
        transition_loss = torch.tensor(0.0, device=device)

    # 3. Smoothness regularization
    diff = current_outputs[:, 1:] - current_outputs[:, :-1]
    smoothness_loss = (diff ** 2).mean() * 0.1

    # Combine losses with weights
    total_loss = mse_loss + 2.0 * transition_loss + smoothness_loss
    
    return total_loss

def train_model_epoch(model, train_loader, optimizer, device, teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        physical, padded_seq, mask, lengths = batch
        physical = physical.to(device)
        padded_seq = padded_seq.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        current_outputs = model(
            physical, target_seq=padded_seq, 
            lengths=lengths, 
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        loss = compute_loss(current_outputs, padded_seq, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * physical.size(0)
    return total_loss / len(train_loader.dataset)

def test_model_epoch(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            physical, padded_seq, mask, lengths = batch
            physical = physical.to(device)
            padded_seq = padded_seq.to(device)
            mask = mask.to(device)
            # if we don't provide target_seq the model won't know how long to generate!
            # this isn't an issue when inferencing or evaluating, but when computing loss we need both of same length
            current_outputs = model(
                physical, target_seq=padded_seq,
                lengths=lengths,
                teacher_forcing_ratio=0.0  # No teacher forcing during testing
            )
            # Compute loss only up to the actual sequence lengths
            loss = compute_loss(current_outputs, padded_seq, mask)
            total_loss += loss.item() * physical.size(0)
    return total_loss / len(test_loader.dataset)

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

def generate_curve_batch(model, physical_input, scalers, device, max_length_limit=100):
    """Generate curves in batch, stopping at first unscaled negative prediction or max length."""
    model.eval()
    _, output_scaler = scalers
    
    # Calculate the scaled value corresponding to 0
    scaled_zero = output_scaler.transform([[0]])[0][0]
    
    batch_size = physical_input.size(0)
    with torch.no_grad():
        hidden = model.init_hidden(physical_input.to(device))
        cell = model.init_cell(physical_input.to(device))
        input_token = model.init_input.expand(batch_size, 1).unsqueeze(1).to(device)
        current_outputs = []
        finished = np.zeros(batch_size, dtype=bool)
        lengths = np.zeros(batch_size, dtype=int)
        
        for t in range(max_length_limit):
            out, (hidden, cell) = model.lstm(input_token, (hidden, cell))
            current_pred = model.current_head(out.squeeze(1))
            current_pred_np = current_pred.cpu().numpy().flatten()
            current_outputs.append(current_pred_np)
            
            # Use scaled_zero for checking termination
            for i in range(batch_size):
                if not finished[i] and current_pred_np[i] < scaled_zero:
                    finished[i] = True
                    lengths[i] = t + 1
            if finished.all():
                break
            input_token = current_pred.unsqueeze(1)
        
        # Count and report sequences that never finished
        unfinished_count = 0
        for i in range(batch_size):
            if lengths[i] == 0:
                unfinished_count += 1
                lengths[i] = len(current_outputs)
        
        if unfinished_count > 0:
            print(f"\nWarning: {unfinished_count}/{batch_size} sequences never predicted a negative value")
            
        current_outputs = np.stack(current_outputs, axis=1)  # (batch, seq)
        unscaled_curves = []
        for i in range(batch_size):
            seq = current_outputs[i, :lengths[i]]
            unscaled = output_scaler.inverse_transform(seq.reshape(-1, 1)).flatten()
            unscaled_curves.append(unscaled)
        return unscaled_curves, lengths.tolist()

def plot_generated_curves(sample_data, num_samples_to_plot=4):
    if len(sample_data) > num_samples_to_plot:
        selected_samples = random.sample(sample_data, num_samples_to_plot)
    else:
        selected_samples = sample_data
    
    fig = plt.figure(figsize=(15, 5 * ((len(selected_samples)+1) // 2)))
    n_cols = 2
    n_rows = (len(selected_samples)+1) // 2
    gs = GridSpec(n_rows, n_cols, figure=fig)
    for idx, (gen_curve, true_curve, r2) in enumerate(selected_samples):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        ax.plot(true_curve, label=f"True (len={len(true_curve)})", alpha=0.7)
        ax.plot(gen_curve, label=f"Generated (len={len(gen_curve)})", alpha=0.7)
        ax.set_title(f"Sample {idx+1}, R² = {r2:.4f}")
        ax.legend()
    plt.tight_layout()
    plt.savefig("rnn_model_sample.png")
    plt.close()

def evaluate_model_batched(model, test_loader, scalers, device, max_sequence_length, include_plots=True):
    """Evaluate model performance in batches using generated curves.

    For each sample, unscale the predicted and true (padded) sequence,
    compute the R² score over the valid region, and plot a few examples.

    Returns:
        mean_r2 (float): Mean R² score over all valid test samples.
        samples_data (list of tuples): Each tuple contains (generated_curve, true_curve, r2_score).
    """
    model.eval()
    all_r2_scores = []
    sample_data = []
    total_samples = 0
    length_mismatch_count = 0
    
    with torch.no_grad():
        for physical_b, padded_seq_b, mask_b, lengths_b in test_loader:
            physical_b = physical_b.to(device)
            generated_curves, gen_lengths = generate_curve_batch(
                model, physical_b, scalers, device, max_length_limit=max_sequence_length
            )
            _, output_scaler = scalers
            
            batch_length_mismatches = 0
            for i in range(len(lengths_b)):
                total_samples += 1
                true_len = lengths_b[i].item()
                true_curve = padded_seq_b[i, :true_len].cpu().numpy()
                true_unscaled = output_scaler.inverse_transform(true_curve.reshape(-1, 1)).flatten()
                gen_curve = generated_curves[i]
                
                if len(gen_curve) != len(true_unscaled):
                    batch_length_mismatches += 1
                    length_mismatch_count += 1
                
                min_len = min(len(true_unscaled), len(gen_curve))
                if min_len > 1:
                    r2 = r2_score(true_unscaled[:min_len], gen_curve[:min_len])
                    all_r2_scores.append(r2)
                    sample_data.append((gen_curve, true_unscaled, r2))
            
            if batch_length_mismatches > 0:
                print(f"\nBatch length mismatches: {batch_length_mismatches}/{len(lengths_b)} sequences")
    mean_r2 = np.mean(all_r2_scores) if all_r2_scores else float('nan')
    print(f"\nModel Evaluation Summary:")
    print(f"- Mean R² Score over {len(all_r2_scores)} valid test samples: {mean_r2:.4f}")
    print(f"- Length mismatches: {length_mismatch_count}/{total_samples} sequences ({(length_mismatch_count/total_samples)*100:.1f}%)")
    
    if include_plots:
        print("\nPlotting a few generated vs true curves...")
        plot_generated_curves(sample_data, num_samples_to_plot=4)
    return mean_r2, sample_data
    
        
if __name__ == "__random__":
    # Enhanced hyperparameters
    physical_dim = 31
    hidden_dim = 128  # Increased capacity
    num_layers = 2    # Multiple layers
    dropout = 0.3     # Slightly more regularization
    learning_rate = 5e-4  # Smaller learning rate for stability
    batch_size = 128  # Larger batch size for better statistics
    epochs = 50      # Train longer
    
    # Learning rate scheduling
    scheduler_patience = 3
    scheduler_factor = 0.5
    min_lr = 1e-5

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

    # Use preprocess_data_no_eos instead of preprocess_data_with_eos
    data = preprocess_data_no_eos(input_paths, output_paths)
    X_train, padded_y_train, mask_train, lengths_train = data['train']
    X_test, padded_y_test, mask_test, lengths_test = data['test']

    print("\nDataset information:")
    print(f"Training set: {len(lengths_train)} samples")
    print(f"Test set: {len(lengths_test)} samples")
    print(f"Sequence lengths - Train: min={min(lengths_train)}, max={max(lengths_train)}")
    print(f"Sequence lengths - Test: min={min(lengths_test)}, max={max(lengths_test)}")

    # Create TensorDatasets with masks
    train_dataset = TensorDataset(X_train, padded_y_train, mask_train, torch.tensor(lengths_train))
    test_dataset = TensorDataset(X_test, padded_y_train, mask_test, torch.tensor(lengths_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("\nInitializing and training model...")
    model = SeqIVModel(physical_dim, hidden_dim, num_layers, dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=scheduler_patience, 
        factor=scheduler_factor, min_lr=min_lr, verbose=True
    )

    best_test_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    max_patience = 10

    for epoch in range(1, epochs + 1):
        train_loss = train_model_epoch(model, train_loader, optimizer, device, 
                                     teacher_forcing_ratio=0.5)
        test_loss = test_model_epoch(model, test_loader, device)
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        # Early stopping with model saving
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model with test loss: {best_test_loss:.4f}")

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
    torch.save(save_dict, 'rnn_seq_model.pth')
    print("\nModel saved to rnn_seq_model.pth")

if __name__ == "__main__":
    # Load the trained model and scalers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scalers = load_trained_model('checkpoints/rnn_seq_model.pth', device)
    
    # Calculate scaled zero threshold
    _, output_scaler = scalers
    scaled_zero = output_scaler.transform([[0]])[0][0]
    
    # Test dataset loading
    test_input_path = "dataset/Data_10k_sets/Data_10k_rng2/LHS_parameters_m.txt"
    test_output_path = "dataset/Data_10k_sets/Data_10k_rng2/iV_m.txt"
    
    print("\nGenerating raw curve visualizations...")
    plot_raw_curves_with_truncation(model, test_loader, scalers, device, num_samples=6)

    data = preprocess_data_no_eos([test_input_path], [test_output_path], test_size=1.0)
    X_test, padded_y_test, mask_test, lengths_test = data['test']
    print("\nTest dataset information:")
    print(f"Test set: {len(lengths_test)} samples")
    print(f"Sequence lengths - Test: min={min(lengths_test)}, max={max(lengths_test)}")

    # Create TensorDataset with masks
    test_dataset = TensorDataset(X_test, padded_y_test, mask_test, torch.tensor(lengths_test))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("\nEvaluating the model on the test set...")
    mean_r2 = evaluate_model_batched(model, test_loader, scalers, device, max_sequence_length=100)
    print(f"\nMean R² Score on Test Set: {mean_r2:.4f}")