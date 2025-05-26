import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, QuantileTransformer
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# --- 1. Data Loading and Preparation ---

def load_data(file_paths):
    """Load and stack data from multiple files."""
    data = [np.loadtxt(p, delimiter=',') for p in file_paths]
    return np.vstack(data)

def prepare_full_data(input_paths, output_paths, test_size=0.2, random_state=42):
    """
    Load and preprocess both input (X) and output (y) data.
    Inputs are prepared for the Autoencoder.
    Outputs are prepared for the MLP Regressor.
    """
    print("\nData Loading and Full Preprocessing:")
    
    # Load raw data
    X_data = load_data(input_paths)  # Expected shape: (n_samples, 31)
    y_data = load_data(output_paths)  # Expected shape: (n_samples, n_iv_points)
    
    print(f"Raw X data shape: {X_data.shape}, Raw y data shape: {y_data.shape}")
    print(f"Raw X data ranges: Min={X_data.min():.2e}, Max={X_data.max():.2e}")
    print(f"Raw y data ranges: Min={y_data.min():.2e}, Max={y_data.max():.2e}")
    
    # Split into train/test
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_data, y_data, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    # --- Preprocess Inputs (X) for Autoencoder ---
    epsilon = 1e-40
    X_train_log = np.log10(X_train_raw + epsilon)
    X_test_log = np.log10(X_test_raw + epsilon)
    
    # Handle potential NaNs/Infs from log transform
    X_train_log = np.nan_to_num(X_train_log, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_log = np.nan_to_num(X_test_log, nan=0.0, posinf=0.0, neginf=0.0)

    input_scaler = RobustScaler(quantile_range=(5.0, 95.0))
    X_train_scaled_ae = input_scaler.fit_transform(X_train_log)
    X_test_scaled_ae = input_scaler.transform(X_test_log)
    
    X_train_tensor_ae = torch.tensor(X_train_scaled_ae, dtype=torch.float32)
    X_test_tensor_ae = torch.tensor(X_test_scaled_ae, dtype=torch.float32)
    
    print(f"\nProcessed Inputs (for AE):")
    print(f"  X_train_tensor_ae shape: {X_train_tensor_ae.shape}")
    print(f"  X_test_tensor_ae shape: {X_test_tensor_ae.shape}")

    # --- Preprocess Outputs (y) for MLP ---
    scale_factor_I = 150.0 
    y_train_arcsinh = np.arcsinh(y_train_raw / scale_factor_I)
    y_test_arcsinh = np.arcsinh(y_test_raw / scale_factor_I)
    
    output_transformer = QuantileTransformer(output_distribution='normal',
                                               n_quantiles=min(len(y_train_raw), 1000))
    y_train_scaled = output_transformer.fit_transform(y_train_arcsinh)
    y_test_scaled = output_transformer.transform(y_test_arcsinh)
    
    y_train_tensor_mlp = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_test_tensor_mlp = torch.tensor(y_test_scaled, dtype=torch.float32)

    print(f"\nProcessed Outputs (for MLP):")
    print(f"  y_train_tensor_mlp shape: {y_train_tensor_mlp.shape}")
    print(f"  y_test_tensor_mlp shape: {y_test_tensor_mlp.shape}")
        
    return {
         'X_train_tensor_ae': X_train_tensor_ae,
         'X_test_tensor_ae': X_test_tensor_ae,
         'y_train_tensor_mlp': y_train_tensor_mlp,
         'y_test_tensor_mlp': y_test_tensor_mlp,
         'input_scaler': input_scaler,
         'y_test_raw': y_test_raw,
         'output_transformer': output_transformer,
         'scale_factor_I': scale_factor_I
    }

# --- 2. Model Definitions ---

class Autoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim) # Bottleneck
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim) # Output reconstruction
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_embedding(self, x):
        return self.encoder(x)

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim): # input_dim is AE's embedding_dim
        super(MLPRegressor, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3), # Added dropout for regularization
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim) # Predicts processed IV curve points
        )
        
    def forward(self, x):
        return self.fc_layers(x)

# --- 3. Training Functions ---

def train_model(model, dataloader, criterion, optimizer, num_epochs, device, is_ae=True):
    """Generic training loop for AE or MLP."""
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader: # For AE, batch_y is batch_X
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        
        avg_epoch_loss = epoch_loss / len(dataloader.dataset)
        if (epoch + 1) % (num_epochs // 10 if num_epochs >= 10 else 1) == 0:
             print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.6f}")
    print("Training complete.")

# --- 4. Utility Functions ---

def get_embeddings_from_ae(autoencoder, data_tensor, device):
    """Extract embeddings from a trained autoencoder."""
    autoencoder.eval()
    with torch.no_grad():
        embeddings = autoencoder.get_embedding(data_tensor.to(device))
    return embeddings.cpu()

def inverse_transform_predictions(predictions_tensor, output_transformer, scale_factor_I):
    """Inverse transform the predictions to the original scale."""
    predictions_scaled = predictions_tensor.cpu().numpy()
    predictions_arcsinh = output_transformer.inverse_transform(predictions_scaled)
    predictions_original = np.sinh(predictions_arcsinh) * scale_factor_I
    return predictions_original

# --- 5. Main Execution Block ---

if __name__ == '__main__':
    # Configuration
    INPUT_DIM = 31  # Number of physical parameters
    # OUTPUT_DIM_IV will be determined by the shape of y_data
    AE_EMBEDDING_DIM = 16 # Example embedding dimension
    MLP_LEARNING_RATE = 1e-4
    AE_LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    AE_EPOCHS = 100  # Adjust as needed
    MLP_EPOCHS = 100 # Adjust as needed

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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and preprocess data
    data_dict = prepare_full_data(train_input_paths, train_output_paths, test_size=0.2)
    
    X_train_ae_tensor = data_dict['X_train_tensor_ae']
    X_test_ae_tensor = data_dict['X_test_tensor_ae']
    y_train_mlp_tensor = data_dict['y_train_tensor_mlp']
    y_test_mlp_tensor = data_dict['y_test_tensor_mlp']
    y_test_raw = data_dict['y_test_raw']
    output_transformer = data_dict['output_transformer']
    scale_factor_I = data_dict['scale_factor_I']

    OUTPUT_DIM_IV = y_train_mlp_tensor.shape[1] # Determine from data

    # Create DataLoaders for AE
    # For AE, input and target are the same (X_train_ae_tensor)
    train_dataset_ae = TensorDataset(X_train_ae_tensor, X_train_ae_tensor)
    train_loader_ae = DataLoader(train_dataset_ae, batch_size=BATCH_SIZE, shuffle=True)
    
    test_dataset_ae = TensorDataset(X_test_ae_tensor, X_test_ae_tensor) # For evaluating AE reconstruction
    test_loader_ae = DataLoader(test_dataset_ae, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Initialize and Train Autoencoder
    print("\n--- Training Autoencoder ---")
    autoencoder = Autoencoder(input_dim=INPUT_DIM, embedding_dim=AE_EMBEDDING_DIM).to(device)
    criterion_ae = nn.MSELoss()
    optimizer_ae = optim.Adam(autoencoder.parameters(), lr=AE_LEARNING_RATE)
    
    train_model(autoencoder, train_loader_ae, criterion_ae, optimizer_ae, AE_EPOCHS, device, is_ae=True)

    # 3. Get Embeddings from Trained Autoencoder
    print("\n--- Generating Embeddings from Autoencoder ---")
    X_train_embeddings = get_embeddings_from_ae(autoencoder, X_train_ae_tensor, device)
    X_test_embeddings = get_embeddings_from_ae(autoencoder, X_test_ae_tensor, device)
    
    print(f"X_train_embeddings shape: {X_train_embeddings.shape}")
    print(f"X_test_embeddings shape: {X_test_embeddings.shape}")

    # Create DataLoaders for MLP
    train_dataset_mlp = TensorDataset(X_train_embeddings, y_train_mlp_tensor)
    train_loader_mlp = DataLoader(train_dataset_mlp, batch_size=BATCH_SIZE, shuffle=True)
    
    test_dataset_mlp = TensorDataset(X_test_embeddings, y_test_mlp_tensor)
    test_loader_mlp = DataLoader(test_dataset_mlp, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Initialize and Train MLP Regressor
    print("\n--- Training MLP Regressor ---")
    mlp_regressor = MLPRegressor(input_dim=AE_EMBEDDING_DIM, output_dim=OUTPUT_DIM_IV).to(device)
    criterion_mlp = nn.MSELoss() # MSE for regression
    optimizer_mlp = optim.Adam(mlp_regressor.parameters(), lr=MLP_LEARNING_RATE)
    
    train_model(mlp_regressor, train_loader_mlp, criterion_mlp, optimizer_mlp, MLP_EPOCHS, device, is_ae=False)

    # 5. Evaluate MLP
    print("\n--- Evaluating MLP Regressor ---")
    mlp_regressor.eval()
    test_loss_mlp = 0
    all_predictions = []
    with torch.no_grad():
        for batch_X_emb, batch_y_mlp in test_loader_mlp:
            batch_X_emb, batch_y_mlp = batch_X_emb.to(device), batch_y_mlp.to(device)
            predictions = mlp_regressor(batch_X_emb)
            loss = criterion_mlp(predictions, batch_y_mlp)
            test_loss_mlp += loss.item() * batch_X_emb.size(0)
            all_predictions.append(predictions.cpu())
            
    avg_test_loss_mlp = test_loss_mlp / len(test_loader_mlp.dataset)
    print(f"MLP Average Test Loss: {avg_test_loss_mlp:.6f}")
    
    final_predictions_tensor = torch.cat(all_predictions, dim=0)
    final_predictions_original = inverse_transform_predictions(final_predictions_tensor, output_transformer, scale_factor_I)
    
    print(f"Shape of final predictions (original scale): {final_predictions_original.shape}")
    print(f"Example of first 5 predicted IV curve points for the first test sample:")
    print(final_predictions_original[0, :5])
    print(f"Corresponding actual first 5 IV curve points for the first test sample:")
    print(y_test_raw[0, :5])

    # Compute per-curve R²
    r2_scores = []
    for i in range(y_test_raw.shape[0]):
        r2 = r2_score(y_test_raw[i], final_predictions_original[i])
        r2_scores.append(r2)

    print(f"Average R² across all test curves: {np.mean(r2_scores):.6f}")

    # Plot 6 random sample curves
    num_samples_to_plot = 6
    random_indices = np.random.choice(y_test_raw.shape[0], num_samples_to_plot, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, ax in zip(random_indices, axes):
        ax.plot(y_test_raw[idx], label="Actual", marker='o')
        ax.plot(final_predictions_original[idx], label="Predicted", marker='x')
        ax.set_title(f"Sample {idx} (R²={r2_scores[idx]:.4f})")
        ax.set_xlabel("IV Curve Points")
        ax.set_ylabel("Current (A)")
        ax.legend()

    plt.tight_layout()
    plt.show()

    # Save trained Autoencoder
    ae_model_path = "checkpoints/autoencoder_model.pth"
    torch.save(autoencoder.state_dict(), ae_model_path)
    print(f"Autoencoder model saved to {ae_model_path}")

    # Save trained MLP Regressor
    mlp_model_path = "checkpoints/mlp_regressor_model.pth"
    torch.save(mlp_regressor.state_dict(), mlp_model_path)
    print(f"MLP Regressor model saved to {mlp_model_path}")
