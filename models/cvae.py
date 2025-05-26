import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.optim.lr_scheduler import CosineAnnealingLR
from ..utils.preprocess import prepare_data

class CVAE(nn.Module):
    def __init__(self, physical_dim, va_sweep_dim, latent_dim, output_iv_dim):
        super(CVAE, self).__init__()
        self.physical_dim = physical_dim
        self.va_sweep_dim = va_sweep_dim # Dimension of the Va sweep (length of Va_np)
        self.latent_dim = latent_dim
        self.output_iv_dim = output_iv_dim # Should be equal to va_sweep_dim (num_voltage_points)

        # Encoder: Takes physical qualities and Va sweep, outputs latent distribution (mu, logvar)
        encoder_input_dim = physical_dim + va_sweep_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
        )
        
        # Separate heads for mu and logvar
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder: Takes latent vector z and physical qualities, outputs IV curve
        decoder_input_dim = latent_dim + physical_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(256, output_iv_dim),
            nn.Tanh()  # Help constrain the output range
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
        # z: (batch_size, latent_dim)
        # x_physical: (batch_size, physical_dim)
        combined_input = torch.cat((z, x_physical), dim=1)
        return self.decoder(combined_input)

    def forward(self, x_physical, y_iv_cuirve_data):
        mu, logvar = self.encode(x_physical, y_iv_cuirve_data)
        z = self.reparameterize(mu, logvar)
        reconstructed_y_iv = self.decode(z, x_physical)
        return reconstructed_y_iv, mu, logvar
    
def physics_constraints_loss(iv_curves):
    """Calculate physics-based constraints loss for IV curves.
    Args:
        iv_curves: Tensor of shape (batch_size, num_voltage_points)
    Returns:
        Tuple of (monotonicity_loss, smoothness_loss)
    """
    # Monotonicity loss: penalize positive gradients (current should decrease with voltage)
    diff = iv_curves[:, 1:] - iv_curves[:, :-1]  # First-order differences
    monotonicity_loss = torch.relu(diff).pow(2).mean()  # Squared ReLU to only penalize positive gradients
    
    # Smoothness loss: penalize large second derivatives
    second_diff = diff[:, 1:] - diff[:, :-1]  # Second-order differences
    smoothness_loss = second_diff.pow(2).mean()
    
    return monotonicity_loss, smoothness_loss

def loss_function_cvae(reconstructed_y_iv, true_y_iv, mu, logvar, kl_beta=1.0,
                      physics_weight=0.1, monotonicity_weight=0.7, smoothness_weight=0.3):
    """CVAE loss function with physics constraints.
    Args:
        reconstructed_y_iv: Reconstructed IV curves
        true_y_iv: True IV curves
        mu, logvar: Latent space parameters
        kl_beta: Weight for KL divergence term
        physics_weight: Overall weight for physics constraints
        monotonicity_weight: Relative weight for monotonicity within physics constraints
        smoothness_weight: Relative weight for smoothness within physics constraints
    """
    # Reconstruction Loss (Mean Squared Error)
    mse = nn.functional.mse_loss(reconstructed_y_iv, true_y_iv, reduction='sum')

    # KL Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Physics constraints (apply to both predicted and true curves)
    mono_loss_pred, smooth_loss_pred = physics_constraints_loss(reconstructed_y_iv)
    mono_loss_true, smooth_loss_true = physics_constraints_loss(true_y_iv)
    
    # Use true curve losses as scaling factors
    mono_scale = torch.clamp(mono_loss_true, min=1e-6)
    smooth_scale = torch.clamp(smooth_loss_true, min=1e-6)
    
    # Normalized physics losses
    physics_loss = (physics_weight *
                   (monotonicity_weight * (mono_loss_pred / mono_scale) +
                    smoothness_weight * (smooth_loss_pred / smooth_scale)))
    
    # Total loss
    return mse + kl_beta * kld + physics_loss

def train_model_epoch(model, train_loader, optimizer, scheduler, epoch_num, device,
                     kl_beta_weight, max_grad_norm=1.0, print_every_n_batches=20):
    model.train() # Set model to training mode
    cumulative_train_loss = 0
    cumulative_train_kl = 0.0
    
    for batch_idx, (x_physical_batch, y_iv_batch) in enumerate(train_loader):
        x_physical_batch = x_physical_batch.to(device)
        y_iv_batch = y_iv_batch.to(device)

        optimizer.zero_grad() # Clear previous gradients
        reconstructed_y_iv, mu, logvar = model(x_physical_batch, y_iv_batch)
        
        loss = loss_function_cvae(reconstructed_y_iv, y_iv_batch, mu, logvar, kl_beta_weight)
        loss.backward() # Compute gradients
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step() # Update model parameters
        
        # Compute KL divergence per batch (averaged per sample)
        batch_kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).mean().item()
        cumulative_train_kl += batch_kl
        cumulative_train_loss += loss.item()

        if batch_idx % print_every_n_batches == 0:
            print(f'Train Epoch: {epoch_num} [{batch_idx * len(x_physical_batch)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tBatch Loss: {loss.item() / len(x_physical_batch):.6f}')
    
    avg_epoch_train_loss = cumulative_train_loss / len(train_loader.dataset)
    avg_epoch_train_kl = cumulative_train_kl / len(train_loader)
    print(f'====> Epoch: {epoch_num} Average train loss: {avg_epoch_train_loss:.4f}')
    print(f'====> Epoch: {epoch_num} Average train KL divergence: {avg_epoch_train_kl:.4f}')
    return avg_epoch_train_loss

def test_model_epoch(model, test_loader, epoch_num, device, kl_beta_weight):
    model.eval() # Set model to evaluation mode
    cumulative_test_loss = 0
    cumulative_test_kl = 0.0
    with torch.no_grad(): # Disable gradient calculations for testing
        for x_physical_batch, y_iv_batch in test_loader:
            x_physical_batch = x_physical_batch.to(device)
            y_iv_batch = y_iv_batch.to(device)

            reconstructed_y_iv, mu, logvar = model(x_physical_batch, y_iv_batch)
            loss = loss_function_cvae(reconstructed_y_iv, y_iv_batch, mu, logvar, kl_beta_weight)
            # Compute KL divergence for the batch 
            batch_kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).mean().item()
            cumulative_test_kl += batch_kl
            cumulative_test_loss += loss.item()

    avg_epoch_test_loss = cumulative_test_loss / len(test_loader.dataset)
    avg_epoch_test_kl = cumulative_test_kl / len(test_loader)
    print(f'====> Epoch: {epoch_num} Average test loss: {avg_epoch_test_loss:.4f}')
    print(f'====> Epoch: {epoch_num} Average test KL divergence: {avg_epoch_test_kl:.4f}')
    return avg_epoch_test_loss

def load_model(model_path, device):
    """
    Load a trained CVAE model from a file.
    Args:
        model_path: Path to the saved model state dictionary.
            NOTE: assumes contains the model AND scalers
        device: PyTorch device to load the model onto.
    Returns:
        model: The loaded CVAE model.
        input_physical_scaler: Scaler for input physical quantities.
        output_transformer: QuantileTransformer for output IV curves.
    """
    model = CVAE(physical_dim=31, va_sweep_dim=41, latent_dim=25, output_iv_dim=41)
    data = torch.load(model_path, map_location=device)
    if 'model_state_dict' in data:
        model.load_state_dict(data['model_state_dict'])
        model.to(device)
        input_physical_scaler = data['input_physical_scaler']
        output_transformer = data['output_transformer']
        print("Loaded model with scalers.")
    else:
        print("Loading model without scalers, assuming they are not needed.")
        input_physical_scaler = None
        output_transformer = None

    return model, input_physical_scaler, output_transformer

def generate_iv_curves_from_scaled_physical_quantities(model, scaled_physical_qualities_tensor, 
                                    device, num_latent_samples_per_input=1):
    """
    Generates IV curve(s) from SCALED physical qualities.
    Args:
        model: The trained CVAE model.
        scaled_physical_qualities_tensor: A 2D tensor of shape (num_inputs, num_physical_qualities)
                                           containing SCALED physical qualities.
        device: PyTorch device.
        num_latent_samples_per_input: Number of latent samples to draw for each input set of physical qualities.
    Returns:
        A tensor of shape (num_inputs, num_latent_samples_per_input, num_voltage_points)
        containing predicted SCALED IV curves. Note: The returned curves are scaled using
        QuantileTransformer and arcsinh transformation.
    """
    model.eval() # Set model to evaluation mode
    
    # Ensure input is 2D
    if scaled_physical_qualities_tensor.ndim == 1:
        scaled_physical_qualities_tensor = scaled_physical_qualities_tensor.unsqueeze(0)
    
    num_input_sets = scaled_physical_qualities_tensor.shape[0]

    with torch.no_grad():
        x_physical_repeated = scaled_physical_qualities_tensor.repeat_interleave(num_latent_samples_per_input, dim=0).to(device)

        # Generate latent samples from the standard normal, shape (num_input_sets * num_latent_samples_per_input, latent_dim)
        z_samples = torch.randn(num_input_sets * num_latent_samples_per_input, model.latent_dim, device=device)

        # Decode using the sampled z and the repeated physical quantities
        generated_iv_curves_flat = model.decode(z_samples, x_physical_repeated)

        all_generated_iv_curves = generated_iv_curves_flat.view((num_input_sets, num_latent_samples_per_input, model.output_iv_dim))
            
    return all_generated_iv_curves

def sample_inference_pipeline():
    """
    Sample inference pipeline to demonstrate how to use the CVAE model for generating IV curves.
    This function assumes the model has been trained and saved, and it will load the model,
    prepare some sample input data, and generate IV curves.
    """
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'cvae_model_full.pth'
    cvae_model, input_physical_scaler, output_transformer = load_model(model_path, device)

    # Example input physical qualities (scaled)
    example_physical_qualities = np.random.rand(5, 31)  # 5 samples, 31 physical qualities
    example_physical_qualities_scaled = input_physical_scaler.transform(example_physical_qualities)  # Scale the input
    example_physical_qualities_tensor = torch.tensor(example_physical_qualities_scaled, dtype=torch.float32, device=device)

    print(f"Input physical qualities for inference (scaled): shape {example_physical_qualities_tensor.shape}")

    # Generate IV curves from these scaled physical qualities
    generated_iv_curves_scaled_tensor = generate_iv_curves_from_scaled_physical_quantities(
        cvae_model, example_physical_qualities_tensor, device, num_latent_samples_per_input=3
    )

    # inverse transform the generated IV curves to their original scale (A/m^2)
    generated_iv_curves_scaled_flat = generated_iv_curves_scaled_tensor.reshape(-1, 41).cpu().numpy()
    # First apply inverse quantile transform
    generated_iv_curves_arcsinh = output_transformer.inverse_transform(generated_iv_curves_scaled_flat)
    # Then unapply the arcsinh transformation to fully unscale
    generated_iv_curves_unscaled_flat = np.sinh(generated_iv_curves_arcsinh) * 150.0

    # Reshape back to (num_examples, num_latent_samples, num_voltage_points)
    num_examples = example_physical_qualities_tensor.shape[0]

    generated_iv_curves_unscaled_array = generated_iv_curves_unscaled_flat.reshape(
        num_examples, -1, 41
    )

    # plot the the first sample and first generated sample
    plt.figure(figsize=(12, 7))
    V_a = np.concatenate((np.arange(0, 0.41, 0.1), np.arange(0.425, 1.401, 0.025)))
    plt.plot(V_a, generated_iv_curves_unscaled_array[0, 0, :], label='Generated IV Curve (Sample 1)', linestyle='--', alpha=0.6)
    plt.xlabel('Applied Voltage (V)')
    plt.ylabel('Current Density (A/m^2)')
    plt.title('Generated IV Curves from Scaled Physical Qualities')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cvae_inference_example.png")

def main():
    # Constants
    Va_np = np.concatenate((np.arange(0, 0.41, 0.1), np.arange(0.425, 1.401, 0.025)))
    num_voltage_points = len(Va_np)
    # Number of input physical qualities
    num_physical_qualities = 31

    # Hyperparameters
    latent_space_dim = 25       # Dimensionality of the latent space z
    adam_learning_rate = 2e-4   # Initial learning rate for Adam optimizer
    training_batch_size = 32    # Batch size for training
    training_epochs = 100       # Number of training epochs
    kld_loss_beta_target = 0.5  # Final target weight for KL divergence
    warmup_epochs = 5          # Learning rate warmup epochs

    # Setup device (cpu or cuda or mps)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load and prepare data
    print("\n--- Preparing Data ---")

    train_input_paths = [
        "dataset/Data_10k_sets/Data_10k_rng1/LHS_parameters_m.txt",
        "dataset/Data_10k_sets/Data_10k_rng2/LHS_parameters_m.txt",
        "dataset/Data_10k_sets/Data_10k_rng3/LHS_parameters_m.txt"
        # "dataset/Data_100k/LHS_parameters_m.txt",
    ]

    train_output_paths = [
        "dataset/Data_10k_sets/Data_10k_rng1/iV_m.txt",
        "dataset/Data_10k_sets/Data_10k_rng2/iV_m.txt",
        "dataset/Data_10k_sets/Data_10k_rng3/iV_m.txt"
        # "dataset/Data_100k/iV_m.txt",
    ]
    
    # Prepare data
    data = prepare_data(train_input_paths, train_output_paths)
    X_train_tensor, y_train_tensor = data['train']
    X_test_tensor, y_test_tensor = data['test']
    input_physical_scaler, output_transformer = data['scalers']
    y_test_original_scale = data['original_test_data_y'] # For plotting during inference

    # Create DataLoader instances
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    # Use pin_memory=True for faster data transfer to GPU if CUDA is used
    train_loader = DataLoader(train_dataset, batch_size=training_batch_size, shuffle=True, 
                              pin_memory=(device.type == 'cuda'))

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=training_batch_size, shuffle=False, 
                             pin_memory=(device.type == 'cuda'))

    # 2. Initialize model and optimizer
    print("\n--- Initializing Model ---")
    cvae_model = CVAE(physical_dim=num_physical_qualities, 
                      va_sweep_dim=num_voltage_points, # va_sweep_dim is the length of Va_np
                      latent_dim=latent_space_dim, 
                      output_iv_dim=num_voltage_points).to(device) # Move model to device
    
    optimizer = optim.Adam(cvae_model.parameters(), lr=adam_learning_rate)
    
    print(cvae_model)
    total_trainable_params = sum(p.numel() for p in cvae_model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_trainable_params:,}")

    # 3. Training and Testing Loop
    print("\n--- Starting Training ---")
    epoch_train_losses, epoch_test_losses = [], []
    
    # Initialize learning rate scheduler with cosine annealing
    scheduler = CosineAnnealingLR(optimizer, T_max=training_epochs - warmup_epochs)
    
    # Smooth KL annealing parameters
    annealing_epochs = training_epochs // 3  # Use one-third of epochs for KL annealing
    
    for epoch in range(1, training_epochs + 1):
        # Warmup learning rate
        if epoch <= warmup_epochs:
            lr = adam_learning_rate * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step()
        
        # Smooth KL annealing with sigmoid function
        if epoch < annealing_epochs:
            progress = epoch / annealing_epochs
            kl_beta_current = kld_loss_beta_target / (1 + np.exp(-10 * (progress - 0.5)))
        else:
            kl_beta_current = kld_loss_beta_target
            
        print(f'\nEpoch {epoch}, LR: {optimizer.param_groups[0]["lr"]:.6f}, KL Beta: {kl_beta_current:.4f}')
        
        avg_train_loss = train_model_epoch(cvae_model, train_loader, optimizer, scheduler,
                                         epoch, device, kl_beta_current)
        avg_test_loss = test_model_epoch(cvae_model, test_loader, epoch, device, kl_beta_current)
        
        epoch_train_losses.append(avg_train_loss)
        epoch_test_losses.append(avg_test_loss)
    print("--- Training Finished ---")

    # Plot training and testing loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, training_epochs + 1), epoch_train_losses, label='Average Train Loss per Epoch')
    plt.plot(range(1, training_epochs + 1), epoch_test_losses, label='Average Test Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss (MSE + beta*KLD) per Sample')
    plt.title('CVAE Training and Test Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cvae_training_loss_curves.png")
    print("Saved training/test loss plot to cvae_training_loss_curves.png")

    # 4. Save trained model with the scalers for later use
    model_save_path = 'cvae_model_full.pth'
    torch.save({
        'model_state_dict': cvae_model.state_dict(),
        'input_physical_scaler': input_physical_scaler,
        'output_transformer': output_transformer
    }, model_save_path)

    # 5. Inference Example
    print("\n--- Performing Inference Example ---")
    
    # Select a few samples from the test set for inference demonstration
    num_examples_for_inference = 5  # Number of examples to show
    test_indices = np.random.choice(len(X_test_tensor), num_examples_for_inference, replace=False)
    X_test_tensor = X_test_tensor[test_indices]
    # Get their SCALED physical qualities
    example_physical_qualities_scaled_tensor = X_test_tensor[:num_examples_for_inference]

    # For each example, generate multiple IV curves by sampling different z from latent space
    num_latent_draws_per_physics_input = 5
    
    print(f"Input physical qualities for inference (scaled, {num_examples_for_inference} samples): shape {example_physical_qualities_scaled_tensor.shape}")

    # Generate IV curves (these will be on the SCALED output space)
    generated_iv_curves_scaled_tensor = generate_iv_curves_from_scaled_physical_quantities(
        cvae_model, example_physical_qualities_scaled_tensor, device, 
        num_latent_samples_per_input=num_latent_draws_per_physics_input
    )
    # Expected shape: (num_examples_for_inference, num_latent_draws_per_physics_input, num_voltage_points)
    
    print(f"Generated IV curves (scaled): shape {generated_iv_curves_scaled_tensor.shape}")

    # Inverse transform the generated IV curves to their original scale (A/m^2) for interpretation/plotting
    # The scaler expects 2D input (total_samples, n_features), so reshape first
    generated_iv_curves_scaled_flat = generated_iv_curves_scaled_tensor.reshape(-1, num_voltage_points).cpu().numpy()
    # First apply inverse quantile transform
    generated_iv_curves_arcsinh = output_transformer.inverse_transform(generated_iv_curves_scaled_flat)
    # Then unapply the arcsinh transformation to fully unscale
    generated_iv_curves_unscaled_flat = np.sinh(generated_iv_curves_arcsinh) * 150.0
    
    # Reshape back to (num_examples_for_inference, num_latent_draws_per_physics_input, num_voltage_points)
    generated_iv_curves_unscaled_array = generated_iv_curves_unscaled_flat.reshape(
        num_examples_for_inference, num_latent_draws_per_physics_input, num_voltage_points
    )
    print(f"Generated IV curves (unscaled, A/m^2): shape {generated_iv_curves_unscaled_array.shape}")

    # Plotting all examples in subplots
    num_rows = (num_examples_for_inference + 2) // 3  # Ceiling division to get number of rows
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5*num_rows))
    fig.suptitle('CVAE: True vs. Generated IV Curves', fontsize=16, y=1.02)
    
    # Flatten axes for easier indexing if multiple rows
    axes_flat = axes.flatten() if num_rows > 1 else axes
    
    # Plot each example in its subplot
    for i in range(num_examples_for_inference):
        ax = axes_flat[i]
        
        # Plot true IV curve
        true_iv_curve_unscaled = y_test_original_scale[i]
        ax.plot(Va_np, true_iv_curve_unscaled, label='True', color='black',
                linewidth=2.5, marker='o', markersize=4, zorder=5)

        # Plot generated curves
        for j in range(num_latent_draws_per_physics_input):
            ax.plot(Va_np, generated_iv_curves_unscaled_array[i, j, :],
                    label=f'Gen {j+1}', linestyle='--', alpha=0.6)
        
        ax.set_xlabel('Applied Voltage (V)')
        ax.set_ylabel('Current Density (A/m^2)')
        ax.set_title(f'Sample {i+1}')
        ax.legend(loc='best', fontsize='small')
        ax.grid(True)
    
    # Hide empty subplots if any
    for i in range(num_examples_for_inference, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("cvae_inference_examples.png", bbox_inches='tight')
    print("Saved all inference plots to cvae_inference_examples.png")

    # Example: Calculate Mean Squared Error for the first generated sample of the first example physics input
    if num_examples_for_inference > 0 and num_latent_draws_per_physics_input > 0:
        first_example_true_iv = y_test_original_scale[0]
        first_example_first_generated_iv = generated_iv_curves_unscaled_array[0, 0, :]
        mse_for_first_example = np.mean((first_example_first_generated_iv - first_example_true_iv)**2)
        print(f"\nMSE between first generated sample and true IV for example 0 (unscaled): {mse_for_first_example:.4e} (A/m^2)^2")

    # --- Per-curve R² calculation for all test samples (using first latent sample for each) ---
    n_test = min(len(y_test_original_scale), generated_iv_curves_unscaled_array.shape[0])
    r2_scores = []
    for i in range(n_test):
        true_curve = y_test_original_scale[i]
        r2_sample_scores = []
        for j in range(num_latent_draws_per_physics_input):
            pred_curve = generated_iv_curves_unscaled_array[i, j, :]
            r2_sample_scores.append(r2_score(true_curve, pred_curve))
        r2_scores.append(np.mean(r2_sample_scores))  # Average R² over latent samples
    if r2_scores:
        print(f"\nPer-curve R² statistics (first latent sample per test curve):")
        print(f"  Mean R²:   {np.mean(r2_scores):.4f}")
        print(f"  Median R²: {np.median(r2_scores):.4f}")
        print(f"  Std R²:    {np.std(r2_scores):.4f}")
        print(f"  Min R²:    {np.min(r2_scores):.4f}")
        print(f"  Max R²:    {np.max(r2_scores):.4f}")

    # --- Analytics Section ---
    print("\n--- Analytics ---")
    # Latent Space Visualization using PCA and t-SNE
    model_for_analytics = cvae_model
    model_for_analytics.eval()
    latents = []
    with torch.no_grad():
        for x_physical_batch, y_iv_batch in test_loader:
            x_physical_batch = x_physical_batch.to(device)
            y_iv_batch = y_iv_batch.to(device)
            mu, _ = model_for_analytics.encode(x_physical_batch, y_iv_batch)
            latents.append(mu.cpu().numpy())
    latents = np.concatenate(latents, axis=0)
    
    # PCA visualization
    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(latents)
    plt.figure()
    plt.scatter(latents_pca[:, 0], latents_pca[:, 1], alpha=0.6)
    plt.title("Latent Space Visualization (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("latent_space_pca.png")
    print("Saved latent space PCA plot to latent_space_pca.png")
    
    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    latents_tsne = tsne.fit_transform(latents)
    plt.figure()
    plt.scatter(latents_tsne[:, 0], latents_tsne[:, 1], alpha=0.6)
    plt.title("Latent Space Visualization (t-SNE)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("latent_space_tsne.png")
    print("Saved latent space t-SNE plot to latent_space_tsne.png")

if __name__ == "__main__":
    main()
