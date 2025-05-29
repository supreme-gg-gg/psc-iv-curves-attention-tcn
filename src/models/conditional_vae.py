import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.preprocess import preprocess_data_with_masks

class CVAE(nn.Module):
    def __init__(self, physical_dim, va_sweep_dim, latent_dim, output_iv_dim):
        super(CVAE, self).__init__()
        self.physical_dim = physical_dim
        self.va_sweep_dim = va_sweep_dim  # Length of Va sweep
        self.latent_dim = latent_dim
        self.output_iv_dim = output_iv_dim  # Should equal va_sweep_dim

        # Simplified Encoder: two Linear layers with ReLU activations
        encoder_input_dim = physical_dim + va_sweep_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        # Simplified Decoder: two Linear layers with ReLU and final Tanh activation
        decoder_input_dim = latent_dim + physical_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_iv_dim),
            nn.Tanh()
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

    def forward(self, x_physical, y_iv_curve_data):  # fixed parameter name
        mu, logvar = self.encode(x_physical, y_iv_curve_data)
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

def loss_function_cvae_masked(reconstructed_y_iv, true_y_iv, mask, mu, logvar, kl_beta=1.0):
    """
    CVAE loss function with physics constraints and masking.
    Here, 'mask' is a tensor of the same shape as true_y_iv (and reconstructed_y_iv),
    with 1 indicating valid data and 0 for padded values.
    """
    # Masked Reconstruction Loss
    mse_element = (reconstructed_y_iv - true_y_iv)**2 * mask
    mse = mse_element.sum() / mask.sum()

    # KL Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return mse + kl_beta * kld

def train_model_epoch(model, train_loader, optimizer, scheduler, epoch_num, device,
                     kl_beta_weight, max_grad_norm=1.0, print_every_n_batches=20):
    model.train()
    cumulative_train_loss = 0
    cumulative_train_kl = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        # Expecting batch: (x_physical, y_iv_batch, mask)
        x_physical_batch, y_iv_batch, mask_batch = batch
        x_physical_batch = x_physical_batch.to(device)
        y_iv_batch = y_iv_batch.to(device)
        mask_batch = mask_batch.to(device)
        
        optimizer.zero_grad()
        reconstructed_y_iv, mu, logvar = model(x_physical_batch, y_iv_batch)
        
        loss = loss_function_cvae_masked(reconstructed_y_iv, y_iv_batch, mask_batch, mu, logvar, kl_beta_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
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
    model.eval()
    cumulative_test_loss = 0
    cumulative_test_kl = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x_physical_batch, y_iv_batch, mask_batch = batch
            x_physical_batch = x_physical_batch.to(device)
            y_iv_batch = y_iv_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            reconstructed_y_iv, mu, logvar = model(x_physical_batch, y_iv_batch)
            loss = loss_function_cvae_masked(reconstructed_y_iv, y_iv_batch, mask_batch, mu, logvar, kl_beta_weight)
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
        output_scaler: QuantileTransformer for output IV curves.
    """
    model = CVAE(physical_dim=31, va_sweep_dim=41, latent_dim=25, output_iv_dim=41)
    data = torch.load(model_path, map_location=device)
    if 'model_state_dict' in data:
        model.load_state_dict(data['model_state_dict'])
        model.to(device)
        input_physical_scaler = data['input_physical_scaler']
        output_scaler = data['output_scaler']
        print("Loaded model with scalers.")
    else:
        print("Loading model without scalers, assuming they are not needed.")
        input_physical_scaler = None
        output_scaler = None

    return model, input_physical_scaler, output_scaler

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

def evaluate_model(model, x_test, y_test, input_physical_scaler, output_scaler, y_test_original_scale):
    """
    Evaluate model performance by comparing generated curves with test data.
    Args:
        model: CVAE model
        x_test: Test input physical parameters
        y_test: Test IV curves (scaled)
        input_physical_scaler: Scaler for input physical quantities
        output_scaler: Transformer for output IV curves
        y_test_original_scale: Original unscaled test data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Use test data instead of random samples
    example_physical_qualities_tensor = x_test[:4].to(device)  # Take first 4 samples for visualization
    print(f"Test physical qualities for inference: shape {example_physical_qualities_tensor.shape}")

    # Define number of latent draws per input
    num_latent_draws_per_physics_input = 3

    # Generate IV curves (SCALED output space)
    generated_iv_curves_scaled_tensor = generate_iv_curves_from_scaled_physical_quantities(
        model, example_physical_qualities_tensor, device, 
        num_latent_samples_per_input=num_latent_draws_per_physics_input
    )
    print(f"Generated IV curves (scaled): shape {generated_iv_curves_scaled_tensor.shape}")

    # Get output dimension from the model
    output_dim = model.output_iv_dim

    # Inverse transform to original scale with correct dimensions
    generated_iv_curves_scaled_flat = generated_iv_curves_scaled_tensor.reshape(-1, output_dim).cpu().numpy()
    generated_iv_curves = output_scaler.inverse_transform(generated_iv_curves_scaled_flat)
    num_examples = example_physical_qualities_tensor.shape[0]
    generated_iv_curves_unscaled_array = generated_iv_curves.reshape(num_examples, -1, output_dim)

    # Plot comparison between true and generated curves
    plt.figure(figsize=(16, 12))
    V_a = np.concatenate((np.arange(0, 0.41, 0.1), np.arange(0.425, 1.401, 0.025)))
    
    for i in range(4):  # Loop through the first 4 test samples
        true_curve = y_test_original_scale[i]
        plt.subplot(2, 2, i + 1)  # Create a 2x2 grid of subplots
        plt.plot(V_a[:len(true_curve)], true_curve, 'k-', label='True IV Curve', linewidth=2)
        
        for j in range(num_latent_draws_per_physics_input):
            plt.plot(V_a, generated_iv_curves_unscaled_array[i, j, :], 
                     label=f'Generated {j+1}', linestyle='--', alpha=0.6)
        
        plt.xlabel('Applied Voltage (V)')
        plt.ylabel('Current Density (A/m^2)')
        plt.title(f'Test Sample {i + 1}: True vs Generated IV Curves')
        plt.legend(loc='best')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("model_evaluation_subplots.png")

    # R² computation by cropping predictions
    n_test = min(len(y_test_original_scale), generated_iv_curves_unscaled_array.shape[0])
    r2_scores = []
    for i in range(n_test):
        true_curve = y_test_original_scale[i]
        true_length = len(true_curve)
        r2_sample_scores = []
        for j in range(num_latent_draws_per_physics_input):
            pred_curve = generated_iv_curves_unscaled_array[i, j, :true_length]
            r2_sample_scores.append(r2_score(true_curve, pred_curve))
        r2_scores.append(np.mean(r2_sample_scores))

    print(f"\nPer-curve R² statistics (after cropping predictions):")
    print(f"  Mean R²:   {np.mean(r2_scores):.4f}")
    print(f"  Median R²: {np.median(r2_scores):.4f}")
    print(f"  Std R²:    {np.std(r2_scores):.4f}")
    print(f"  Min R²:    {np.min(r2_scores):.4f}")
    print(f"  Max R²:    {np.max(r2_scores):.4f}")

    return r2_scores

def main():
    # Constants
    Va_np = np.concatenate((np.arange(0, 0.41, 0.1), np.arange(0.425, 1.401, 0.025)))
    num_voltage_points = len(Va_np)
    num_physical_qualities = 31

    # Hyperparameters
    latent_space_dim = 20       # Dimensionality of the latent space z
    adam_learning_rate = 2e-4   # Initial learning rate for Adam optimizer
    training_batch_size = 64    # Batch size for training
    training_epochs = 50       # Number of training epochs
    kld_loss_beta_target = 0.5  # Final target weight for KL divergence
    warmup_epochs = 5          # Learning rate warmup epochs

    # Setup device (cpu or cuda or mps)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load and prepare data with zero padding
    print("\n--- Preparing Data ---")
    train_input_paths = [
        "dataset/Data_10k_sets/Data_10k_rng1/LHS_parameters_m.txt",
        "dataset/Data_10k_sets/Data_10k_rng2/LHS_parameters_m.txt",
        "dataset/Data_10k_sets/Data_10k_rng3/LHS_parameters_m.txt"
    ]

    train_output_paths = [
        "dataset/Data_10k_sets/Data_10k_rng1/iV_m.txt",
        "dataset/Data_10k_sets/Data_10k_rng2/iV_m.txt",
        "dataset/Data_10k_sets/Data_10k_rng3/iV_m.txt"
    ]
    
    # Use new preprocess function with zero padding
    data = preprocess_data_with_masks(train_input_paths, train_output_paths)
    X_train_tensor, y_train_tensor, train_mask_tensor = data['train']
    X_test_tensor, y_test_tensor, test_mask_tensor = data['test']
    input_physical_scaler, output_scaler = data['scalers']
    y_test_original_scale = data['original_test_y']

    # Create DataLoader instances with masks
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, train_mask_tensor)
    train_loader = DataLoader(train_dataset, batch_size=training_batch_size, shuffle=True, 
                            pin_memory=(device.type == 'cuda'))

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, test_mask_tensor)
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
        'output_scaler': output_scaler
    }, model_save_path)

    # 5. Run model evaluation
    print("\n--- Running Model Evaluation ---")
    r2_scores = evaluate_model(
        model=cvae_model,
        x_test=X_test_tensor,
        y_test=y_test_tensor,
        input_physical_scaler=input_physical_scaler,
        output_scaler=output_scaler,
        y_test_original_scale=y_test_original_scale
    )
    
    # # --- Analytics Section ---
    # print("\n--- Analytics ---")
    # # Latent Space Visualization using PCA and t-SNE
    # model_for_analytics = cvae_model
    # model_for_analytics.eval()
    # latents = []
    # with torch.no_grad():
    #     for x_physical_batch, y_iv_batch in test_loader:
    #         x_physical_batch = x_physical_batch.to(device)
    #         y_iv_batch = y_iv_batch.to(device)
    #         mu, _ = model_for_analytics.encode(x_physical_batch, y_iv_batch)
    #         latents.append(mu.cpu().numpy())
    # latents = np.concatenate(latents, axis=0)
    
    # # PCA visualization
    # pca = PCA(n_components=2)
    # latents_pca = pca.fit_transform(latents)
    # plt.figure()
    # plt.scatter(latents_pca[:, 0], latents_pca[:, 1], alpha=0.6)
    # plt.title("Latent Space Visualization (PCA)")
    # plt.xlabel("Principal Component 1")
    # plt.ylabel("Principal Component 2")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("latent_space_pca.png")
    # print("Saved latent space PCA plot to latent_space_pca.png")
    
    # # t-SNE visualization
    # tsne = TSNE(n_components=2, random_state=42)
    # latents_tsne = tsne.fit_transform(latents)
    # plt.figure()
    # plt.scatter(latents_tsne[:, 0], latents_tsne[:, 1], alpha=0.6)
    # plt.title("Latent Space Visualization (t-SNE)")
    # plt.xlabel("t-SNE Dimension 1")
    # plt.ylabel("t-SNE Dimension 2")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("latent_space_tsne.png")
    # print("Saved latent space t-SNE plot to latent_space_tsne.png")

if __name__ == "__main__":
    main()
