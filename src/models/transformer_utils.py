import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random, math
from sklearn.metrics import r2_score
from utils.preprocess import preprocess_data_with_eos
from transformer_model import TransformerIVModel

def compute_loss(value_outputs, eos_logits, targets, lengths, eos_targets):
    """
    Computes combined MSE loss for values and BCEWithLogitsLoss for EOS token,
    considering valid time steps based on each sequence length.
    """
    loss = 0.0
    for i, l in enumerate(lengths):
        # MSE for value prediction: only up to the actual sequence length
        value_loss = nn.functional.mse_loss(value_outputs[i, :l], targets[i, :l])

        # BCEWithLogitsLoss for EOS prediction: for all steps up to l+1 (where EOS is 1)
        # eos_logits has length (max_len + 1) from Transformer output
        # eos_targets has length (max_len + 1) from preprocessing
        eos_loss = nn.functional.binary_cross_entropy_with_logits(
            eos_logits[i, :l+1], eos_targets[i, :l+1]
        )

        loss += value_loss + eos_loss # Combine losses
    return loss / len(lengths)

def train_model_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (physical, padded_seq, lengths, eos_targets) in enumerate(train_loader):
        physical = physical.to(device)
        padded_seq = padded_seq.to(device)
        lengths = lengths.to(device)
        eos_targets = eos_targets.to(device) # Move EOS targets to device

        optimizer.zero_grad()

        value_outputs, eos_logits = model(physical, target_seq=padded_seq,
                                          lengths=lengths, eos_targets=eos_targets)

        loss = compute_loss(value_outputs, eos_logits, padded_seq, lengths, eos_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * physical.size(0) # Use physical.size(0) for batch size
    return total_loss / len(train_loader.dataset)

def test_model_epoch(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (physical, padded_seq, lengths, eos_targets) in enumerate(test_loader):
            physical = physical.to(device)
            padded_seq = padded_seq.to(device)
            lengths = lengths.to(device)
            eos_targets = eos_targets.to(device)

            # In test mode, we pass target_seq to trigger the parallel computation path
            # This ensures value_outputs and eos_logits have the correct padded lengths for loss calculation.
            value_outputs, eos_logits = model(physical, target_seq=padded_seq,
                                              lengths=lengths, eos_targets=eos_targets)

            loss = compute_loss(value_outputs, eos_logits, padded_seq, lengths, eos_targets)
            total_loss += loss.item() * physical.size(0)
    return total_loss / len(test_loader.dataset)

def generate_curve(model, physical_input, scalers, device, max_length_limit=None, eos_threshold=0.5):
    model.eval()
    _, output_scaler = scalers
    with torch.no_grad():
        physical_input = physical_input.to(device)
        if max_length_limit is None:
            max_length_limit = model.max_seq_len

        # Model's forward pass in inference mode
        value_outputs_scaled, eos_logits_scaled = model(physical_input, max_gen_length=max_length_limit)

        generated_curve_scaled_list = []
        # Process for batch_size=1 (as physical_input is usually a single sample here)
        if value_outputs_scaled.numel() > 0:
            eos_probs = torch.sigmoid(eos_logits_scaled.squeeze(0)).cpu().numpy()
            values = value_outputs_scaled.squeeze(0).cpu().numpy()

            valid_length = len(values) # Default to full generated length
            for k in range(len(eos_probs)):
                if eos_probs[k] > eos_threshold:
                    valid_length = k # EOS detected, sequence ends here
                    break
            generated_curve_scaled_list = values[:valid_length]

        # if empty
        if len(generated_curve_scaled_list) == 0:
            print("No valid curve generated.")
            return np.array([]), 0

        unscaled_curve = output_scaler.inverse_transform(np.array(generated_curve_scaled_list).reshape(-1, 1)).flatten()
        return unscaled_curve, len(unscaled_curve)


def evaluate_model(model, test_loader, scalers, device, max_sequence_length, num_samples_to_plot=4):
    model.eval()
    all_r2_scores = []
    all_sample_data = []

    with torch.no_grad():
        for batch_idx, (physical, padded_seq, lengths, eos_targets) in enumerate(test_loader):
            physical_b, padded_seq_b, lengths_b = physical.to(device), padded_seq.to(device), lengths.to(device)
            _, output_scaler = scalers

            for i in range(physical_b.size(0)):
                current_physical = physical_b[i:i+1]
                target_len = lengths_b[i].item()

                generated_curve_unscaled, _ = generate_curve(
                    model, current_physical, scalers, device, max_length_limit=max_sequence_length
                )

                target_curve_scaled = padded_seq_b[i, :target_len].cpu().numpy()
                if target_curve_scaled.size == 0:
                    target_unscaled = np.array([])
                else:
                    target_unscaled = output_scaler.inverse_transform(target_curve_scaled.reshape(-1, 1)).flatten()

                min_len = min(len(generated_curve_unscaled), len(target_unscaled))
                r2 = float('nan')
                if min_len > 1: # R2 score needs at least 2 points
                    r2 = r2_score(target_unscaled[:min_len], generated_curve_unscaled[:min_len])

                all_r2_scores.append(r2)
                all_sample_data.append((generated_curve_unscaled, target_unscaled, r2))

    valid_r2_scores = [score for score in all_r2_scores if not math.isnan(score)]
    mean_r2 = np.mean(valid_r2_scores) if valid_r2_scores else float('nan')
    print(f"\nModel Evaluation (Overall):")
    print(f"Mean R² Score over {len(valid_r2_scores)} test samples: {mean_r2:.4f}")

    if len(all_sample_data) > num_samples_to_plot:
        selected_samples = random.sample(all_sample_data, num_samples_to_plot)
    else:
        selected_samples = all_sample_data

    fig = plt.figure(figsize=(15, min(5 * ((num_samples_to_plot +1)//2) , 15) )) # Adjust figure height
    # Ensure GridSpec rows/cols are at least 1
    n_rows_plot = max(1, (num_samples_to_plot + 1) // 2)
    n_cols_plot = max(1, min(num_samples_to_plot, 2))

    gs = GridSpec(n_rows_plot, n_cols_plot, figure=fig)

    for i, (generated_curve, target_unscaled, r2) in enumerate(selected_samples):
        ax = fig.add_subplot(gs[i//n_cols_plot, i%n_cols_plot])
        ax.plot(target_unscaled, 'b-', label=f'True (len={len(target_unscaled)})', alpha=0.7)
        ax.plot(generated_curve, 'r--', label=f'Generated (len={len(generated_curve)})', alpha=0.7)
        r2_text = f'R² = {r2:.4f}' if not math.isnan(r2) else 'R²: N/A'
        ax.set_title(f'Sample {i+1}, {r2_text}')
        ax.set_xlabel('Index')
        ax.set_ylabel('Current Density (A/m²)') # Assuming this unit
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.savefig('sample_generations.png')
    plt.close()
    print(f"Sample curves saved to sample_generations.png")
    return mean_r2

def batch_generate_curves(model, physical_inputs, scalers, device, max_length_limit=None, eos_threshold=0.5):
    """
    Generates curves for a batch of physical inputs using the model.

    Args:
        model: The trained model (Transformer or Mamba based).
        physical_inputs (torch.Tensor): A batch of physical input tensors (batch_size, input_dim).
        scalers (tuple): A tuple containing (input_scaler, output_scaler) for unscaling.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to run inference on.
        max_length_limit (int, optional): Maximum sequence length to generate. Defaults to model.max_seq_len.
        eos_threshold (float, optional): Probability threshold for End-of-Sequence (EOS) detection.

    Returns:
        tuple: A tuple containing:
            - generated_curves_unscaled (list): List of unscaled generated curves (numpy arrays).
            - generated_lengths (list): List of lengths of the generated curves.
    """
    model.eval()
    _, output_scaler = scalers

    generated_curves_unscaled = []
    generated_lengths = []

    with torch.no_grad():
        physical_inputs = physical_inputs.to(device)
        if max_length_limit is None:
            # Ensure model.max_seq_len is accessible and appropriate
            max_length_limit = getattr(model, 'max_seq_len', 512)

        # Model's forward pass in inference mode - now handles a batch
        # value_outputs_scaled: (batch_size, max_gen_length, output_dim)
        # eos_logits_scaled: (batch_size, max_gen_length)
        value_outputs_scaled, eos_logits_scaled = model(physical_inputs, max_gen_length=max_length_limit)

        # Process EOS and unscale for the entire batch
        # Squeeze the last dimension if output_dim is 1 for value_outputs_scaled
        if value_outputs_scaled.shape[-1] == 1:
            value_outputs_scaled = value_outputs_scaled.squeeze(-1) # (batch_size, max_gen_length)

        # Convert EOS logits to probabilities
        eos_probs_batch = torch.sigmoid(eos_logits_scaled).cpu().numpy() # (batch_size, max_gen_length)
        values_batch = value_outputs_scaled.cpu().numpy() # (batch_size, max_gen_length)

        for i in range(physical_inputs.size(0)): # Iterate through each sample in the batch
            eos_probs = eos_probs_batch[i]
            values = values_batch[i]

            valid_length = len(values) # Default to full generated length
            # Find the first index where EOS is detected
            for k in range(len(eos_probs)):
                if eos_probs[k] > eos_threshold:
                    valid_length = k # EOS detected, sequence ends here
                    break

            current_generated_curve_scaled = values[:valid_length]

            if len(current_generated_curve_scaled) == 0:
                # Handle cases where no valid curve is generated (e.g., immediate EOS)
                generated_curves_unscaled.append(np.array([]))
                generated_lengths.append(0)
            else:
                # Unscale the generated curve
                unscaled_curve = output_scaler.inverse_transform(
                    current_generated_curve_scaled.reshape(-1, 1)
                ).flatten()
                generated_curves_unscaled.append(unscaled_curve)
                generated_lengths.append(len(unscaled_curve))

    return generated_curves_unscaled, generated_lengths


def evaluate_model_batched(model, test_loader, scalers, device, max_sequence_length, num_samples_to_plot=4):
    """
    Evaluates the model's performance using batched inference.

    Args:
        model: The trained model.
        test_loader: DataLoader for the test dataset.
        scalers (tuple): A tuple containing (input_scaler, output_scaler) for unscaling.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to run inference on.
        max_sequence_length (int): Maximum sequence length for generation.
        num_samples_to_plot (int, optional): Number of sample curves to plot. Defaults to 4.

    Returns:
        float: Mean R² score over all valid test samples.
    """
    model.eval()
    all_r2_scores = []
    all_sample_data = [] # Stores (generated_curve, target_curve, r2) for plotting

    with torch.no_grad():
        for batch_idx, (physical_b, padded_seq_b, lengths_b, eos_targets_b) in enumerate(test_loader):
            physical_b, padded_seq_b, lengths_b = physical_b.to(device), padded_seq_b.to(device), lengths_b.to(device)
            _, output_scaler = scalers

            # Perform batched inference for the entire physical input batch
            generated_curves_unscaled_batch, generated_lengths_batch = batch_generate_curves(
                model, physical_b, scalers, device, max_length_limit=max_sequence_length
            )

            # Iterate through the results of the batched inference
            for i in range(len(generated_curves_unscaled_batch)):
                generated_curve_unscaled = generated_curves_unscaled_batch[i]
                current_generated_length = generated_lengths_batch[i]
                target_len = lengths_b[i].item() # Original length of the target sequence

                # Unscale the target curve for R2 calculation
                target_curve_scaled = padded_seq_b[i, :target_len].cpu().numpy()
                if target_curve_scaled.size == 0:
                    target_unscaled = np.array([])
                else:
                    target_unscaled = output_scaler.inverse_transform(target_curve_scaled.reshape(-1, 1)).flatten()

                # Calculate R2 score
                min_len = min(current_generated_length, len(target_unscaled))
                r2 = float('nan')
                if min_len > 1: # R2 score needs at least 2 points
                    r2 = r2_score(target_unscaled[:min_len], generated_curve_unscaled[:min_len])

                all_r2_scores.append(r2)
                all_sample_data.append((generated_curve_unscaled, target_unscaled, r2))

    valid_r2_scores = [score for score in all_r2_scores if not math.isnan(score)]
    mean_r2 = np.mean(valid_r2_scores) if valid_r2_scores else float('nan')
    print(f"\nModel Evaluation (Overall):")
    print(f"Mean R² Score over {len(valid_r2_scores)} valid test samples: {mean_r2:.4f}")

    if len(all_sample_data) > num_samples_to_plot:
        selected_samples = random.sample(all_sample_data, num_samples_to_plot)
    else:
        selected_samples = all_sample_data

    fig = plt.figure(figsize=(15, min(5 * ((num_samples_to_plot + 1) // 2), 15)))
    n_rows_plot = max(1, (num_samples_to_plot + 1) // 2)
    n_cols_plot = max(1, min(num_samples_to_plot, 2))

    gs = GridSpec(n_rows_plot, n_cols_plot, figure=fig)

    for i, (generated_curve, target_unscaled, r2) in enumerate(selected_samples):
        ax = fig.add_subplot(gs[i // n_cols_plot, i % n_cols_plot])
        ax.plot(target_unscaled, 'b-', label=f'True (len={len(target_unscaled)})', alpha=0.7)
        ax.plot(generated_curve, 'r--', label=f'Generated (len={len(generated_curve)})', alpha=0.7)
        r2_text = f'R² = {r2:.4f}' if not math.isnan(r2) else 'R²: N/A'
        ax.set_title(f'Sample {i+1}, {r2_text}')
        ax.set_xlabel('Index')
        ax.set_ylabel('Current Density (A/m²)') # Assuming this unit
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.savefig('mamba_model_samples.png')
    plt.close()
    print(f"Sample curves saved to mamba_model_samples.png")
    return mean_r2


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define input and output file paths for the dataset
    input_paths = [
        "/content/drive/MyDrive/PSC_IV_CURVES/Data_10k_sets/Data_10k_rng1/LHS_parameters_m.txt",
        "/content/drive/MyDrive/PSC_IV_CURVES/Data_10k_sets/Data_10k_rng2/LHS_parameters_m.txt",
        "/content/drive/MyDrive/PSC_IV_CURVES/Data_10k_sets/Data_10k_rng3/LHS_parameters_m.txt"
        # "/content/drive/MyDrive/PSC_IV_CURVES/Data_100k/LHS_parameters_m.txt",
    ]

    output_paths = [
        "/content/drive/MyDrive/PSC_IV_CURVES/Data_10k_sets/Data_10k_rng1/iV_m.txt",
        "/content/drive/MyDrive/PSC_IV_CURVES/Data_10k_sets/Data_10k_rng2/iV_m.txt",
        "/content/drive/MyDrive/PSC_IV_CURVES/Data_10k_sets/Data_10k_rng3/iV_m.txt"
        # "/content/drive/MyDrive/PSC_IV_CURVES/Data_100k/iV_m.txt",
    ]

    # Hyperparameters for Transformer Model
    physical_dim = 31
    d_model = 32            # Embedding dimension for Transformer. Keep it relatively small for 'lightweight'.
    nhead = 4               # Number of attention heads. Should divide d_model.
    num_decoder_layers = 2  # Number of Transformer Decoder layers.
    dropout = 0.1           # Dropout rate.
    learning_rate = 1e-4    # Slightly lower learning rate for Transformers can be beneficial.
    batch_size = 64
    epochs = 50            # Increased epochs. Transformers might need more epochs but each epoch is faster.

    # Hyperparam for Mambda
    mamba_d_state = 16      # SSM state dimension
    mamba_d_conv = 4        # 1D conv kernel size
    mamba_expand = 2        # Expansion factor for d_inner
    num_mamba_layers = 2

    # Preprocess data (returns padded sequences, lengths, and EOS targets)
    data = preprocess_data_with_eos(input_paths, output_paths)
    X_train, padded_y_train, lengths_train, eos_targets_train = data['train']
    X_test, padded_y_test, lengths_test, eos_targets_test = data['test']
    max_sequence_length = data['max_sequence_length'] # Max length for inference

    print("\nDataset information:")
    print(f"Training set: {len(lengths_train)} samples")
    print(f"Test set: {len(lengths_test)} samples")
    print(f"Sequence lengths - Train: min={torch.min(lengths_train).item()}, max={torch.max(lengths_train).item()}")
    print(f"Sequence lengths - Test: min={torch.min(lengths_test).item()}, max={torch.max(lengths_test).item()}")
    print(f"Maximum sequence length (for generation): {max_sequence_length}")

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, padded_y_train, lengths_train, eos_targets_train)
    test_dataset = TensorDataset(X_test, padded_y_test, lengths_test, eos_targets_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    """Select either Transformer or Mamba to create"""

    model = TransformerIVModel(
        physical_dim=physical_dim,
        d_model=d_model,
        nhead=nhead,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout,
        max_seq_len=max_sequence_length
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # model = MambaIVModel(
    #     physical_dim=physical_dim,
    #     d_model=d_model,
    #     num_mamba_layers=num_mamba_layers,
    #     dropout=dropout,
    #     max_seq_len=max_sequence_length,
    #     mamba_d_state=mamba_d_state,
    #     mamba_d_conv=mamba_d_conv,
    #     mamba_expand=mamba_expand
    # ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train_loss = train_model_epoch(model, train_loader, optimizer, device)
        test_loss = test_model_epoch(model, test_loader, device)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

    """Use either of the functions to save the model"""

    # Save the trained model with scalers and max_sequence_length
    save_dict = {
        'model_state_dict': model.state_dict(),
        'scalers': data['scalers'],
        'params': {
            'physical_dim': physical_dim,
            'd_model': d_model,
            'nhead': nhead,
            'num_decoder_layers': num_decoder_layers,
            'dropout': dropout
        },
        'max_sequence_length': max_sequence_length # Save for inference
    }
    torch.save(save_dict, 'transformer_iv_model.pth')
    print("\nModel saved to transformer_iv_model.pth")

    # save_dict = {
    #     'model_state_dict': model.state_dict(),
    #     'scalers': data['scalers'],
    #     'params': {
    #         'physical_dim': physical_dim,
    #         'd_model': d_model,
    #         'num_mamba_layers': num_mamba_layers,
    #         'dropout': dropout,
    #         'mamba_d_state': mamba_d_state,
    #         'mamba_d_conv': mamba_d_conv,
    #         'mamba_expand': mamba_expand,
    #     },
    #     'max_sequence_length': max_sequence_length
    # }
    # torch.save(save_dict, 'mamba_iv_model.pth')
    # print("\nModel saved to mamba_iv_model.pth")
