import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from models.joint_embedding_model import JointEmbeddingModel

def load_data(file_paths):
    """Load and stack data from multiple files"""
    data = [np.loadtxt(p, delimiter=',') for p in file_paths]
    return np.vstack(data)

def prepare_data(input_paths, output_paths, test_size=0.2):
    """Load and preprocess data with robust validation"""
    print("\nData Loading and Preprocessing:")
    
    # Load raw data
    X_data = load_data(input_paths)
    y_data = load_data(output_paths)
    
    print(f"Raw data ranges:")
    print(f"  Inputs: [{X_data.min():.2f}, {X_data.max():.2f}]")
    print(f"  IV Curves: [{y_data.min():.2f}, {y_data.max():.2f}]")
    
    # First split into train/test to prevent data leakage
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_data, y_data, test_size=test_size, random_state=42)
    
    # Preprocessing input parameters with log transform
    epsilon = 1e-40
    X_train_log = np.log10(X_train_raw + epsilon)
    X_test_log = np.log10(X_test_raw + epsilon)
    
    print(f"\nAfter log transform (inputs):")
    print(f"  Train: [{X_train_log.min():.2f}, {X_train_log.max():.2f}]")
    print(f"  Test: [{X_test_log.min():.2f}, {X_test_log.max():.2f}]")
    
    # For inputs: Use RobustScaler with quantile_range to handle outliers
    input_scaler = RobustScaler(quantile_range=(5, 95))
    
    # For IV curves: Use MinMaxScaler to ensure consistent range
    # Use (0, 1) range for IV curves since they typically don't have extreme negative values
    output_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Transform data
    X_train_scaled = input_scaler.fit_transform(X_train_log)
    y_train_scaled = output_scaler.fit_transform(y_train_raw)
    X_test_scaled = input_scaler.transform(X_test_log)
    y_test_scaled = output_scaler.transform(y_test_raw)
    
    print(f"\nAfter scaling:")
    print(f"Inputs - Train: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
    print(f"Inputs - Test: [{X_test_scaled.min():.2f}, {X_test_scaled.max():.2f}]")
    print(f"IV Curves - Train: [{y_train_scaled.min():.2f}, {y_train_scaled.max():.2f}]")
    print(f"IV Curves - Test: [{y_test_scaled.min():.2f}, {y_test_scaled.max():.2f}]")
    
    # Validate scaled data
    for name, data in [
        ("X_train", X_train_scaled),
        ("X_test", X_test_scaled),
        ("y_train", y_train_scaled),
        ("y_test", y_test_scaled)
    ]:
        if np.isnan(data).any():
            raise ValueError(f"NaN values found in {name}")
        if np.isinf(data).any():
            raise ValueError(f"Inf values found in {name}")
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)
    
    return {
        'train': (X_train_tensor, y_train_tensor),
        'test': (X_test_tensor, y_test_tensor),
        'scalers': (input_scaler, output_scaler)
    }

def train_autoencoders(model, train_loader, epochs=50, device='cpu'):
    """Train input and curve autoencoders separately"""
    model = model.to(device)
    
    # Train input autoencoder
    print("Training input autoencoder...")
    input_optimizer = optim.Adam(
        list(model.input_encoder.parameters()) + 
        list(model.input_decoder.parameters()), 
        lr=1e-3
    )
    
    for epoch in range(epochs):
        model.train()
        input_losses = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            input_optimizer.zero_grad()
            emb_input = model.input_encoder(batch_x)
            recon_input = model.input_decoder(emb_input)
            loss = nn.functional.mse_loss(recon_input, batch_x)
            loss.backward()
            input_optimizer.step()
            input_losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            avg_loss = np.mean(input_losses)
            print(f"  Input autoencoder epoch {epoch+1}: Loss = {avg_loss:.6f}")
    
    # Train curve autoencoder
    print("\nTraining curve autoencoder...")
    curve_optimizer = optim.Adam(
        list(model.curve_encoder.parameters()) + 
        list(model.curve_decoder.parameters()), 
        lr=1e-3
    )
    
    for epoch in range(epochs):
        model.train()
        curve_losses = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            curve_optimizer.zero_grad()
            emb_curve = model.curve_encoder(batch_y)
            recon_curve = model.curve_decoder(emb_curve)
            loss = nn.functional.mse_loss(recon_curve, batch_y)
            loss.backward()
            curve_optimizer.step()
            curve_losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            avg_loss = np.mean(curve_losses)
            print(f"  Curve autoencoder epoch {epoch+1}: Loss = {avg_loss:.6f}")
    
    # Save trained autoencoders
    model.save_autoencoders()
    return model

def compute_loss(outputs, input_data, curve_data, alpha_input=1.0, alpha_curve=1.0, beta=1.0, gamma=0.5):
    """Compute multi-component loss with balanced weighting"""
    # Reconstruction losses
    recon_input_loss = nn.functional.mse_loss(outputs['recon_input'], input_data)
    recon_curve_loss = nn.functional.mse_loss(outputs['recon_curve'], curve_data)
    
    # Alignment loss
    alignment_loss = nn.functional.mse_loss(outputs['emb_input'], outputs['emb_curve'])
    
    # Cross-domain losses
    cycle_input_loss = nn.functional.mse_loss(outputs['generated_input'], input_data)
    cycle_curve_loss = nn.functional.mse_loss(outputs['generated_curve'], curve_data)
    
    # Weighted combination
    total_loss = (alpha_input * recon_input_loss + 
                 alpha_curve * recon_curve_loss + 
                 beta * alignment_loss +
                 gamma * (cycle_input_loss + cycle_curve_loss))
    
    return total_loss, {
        'recon_input': recon_input_loss.item(),
        'recon_curve': recon_curve_loss.item(),
        'alignment': alignment_loss.item(),
        'cycle_input': cycle_input_loss.item(),
        'cycle_curve': cycle_curve_loss.item(),
        'total': total_loss.item()
    }

def train_joint(model, train_loader, test_loader, epochs=50, device='cpu'):
    """Train joint embedding model after autoencoder training"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_test_loss = float('inf')
    
    print("Training joint embedding model...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x, batch_y)
            loss, loss_components = compute_loss(outputs, batch_x, batch_y,
                                              alpha_input=0.5, alpha_curve=0.5, 
                                              beta=3.0, gamma=0.5)
            loss.backward()
            optimizer.step()
            train_losses.append(loss_components)
        
        # Validation
        model.eval()
        test_losses = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x, batch_y)
                _, loss_components = compute_loss(outputs, batch_x, batch_y,
                                               alpha_input=0.5, alpha_curve=0.5,
                                               beta=3.0, gamma=0.5)
                test_losses.append(loss_components)
        
        # Average losses
        avg_train_loss = {k: np.mean([l[k] for l in train_losses]) for k in train_losses[0].keys()}
        avg_test_loss = {k: np.mean([l[k] for l in test_losses]) for k in test_losses[0].keys()}
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Joint training epoch {epoch+1}/{epochs}")
            print(f"  Train - Total: {avg_train_loss['total']:.4f}, Alignment: {avg_train_loss['alignment']:.4f}")
            print(f"  Test  - Total: {avg_test_loss['total']:.4f}, Alignment: {avg_test_loss['alignment']:.4f}")
        
        # Save best model
        if avg_test_loss['total'] < best_test_loss:
            best_test_loss = avg_test_loss['total']
            model.save_full_model()
    
    return model

def test_model(model, test_loader, output_scaler, device='cpu'):
    """Test the trained model's performance using MSE and R^2 metrics in original scale"""
    model.eval()
    all_real_curves = []
    all_pred_curves = []
    
    print("\n=== Testing Model Performance ===")
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # Get model predictions
            pred_curves = model.generate_curve(batch_x)
            
            # Store real and predicted curves
            all_real_curves.append(batch_y.cpu().numpy())
            all_pred_curves.append(pred_curves.cpu().numpy())
    
    # Convert to numpy arrays in scaled space
    y_true_scaled = np.vstack(all_real_curves)
    y_pred_scaled = np.vstack(all_pred_curves)
    
    # Inverse transform to original scale
    y_true = output_scaler.inverse_transform(y_true_scaled)
    y_pred = output_scaler.inverse_transform(y_pred_scaled)
    
    print("\nValue ranges after inverse scaling:")
    print(f"True curves: [{y_true.min():.2f}, {y_true.max():.2f}]")
    print(f"Predicted curves: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    
    # Calculate metrics in original scale
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Calculate per-curve R² scores
    r2_scores = []
    rel_errors = []
    n_curves = y_true.shape[0]
    
    for i in range(n_curves):
        true_curve = y_true[i]
        pred_curve = y_pred[i]
        
        # Calculate R² for each curve
        ss_res = np.sum((true_curve - pred_curve) ** 2)
        ss_tot = np.sum((true_curve - np.mean(true_curve)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        r2_scores.append(r2)
        
        # Calculate relative error
        # Handle division by small numbers
        rel_error = np.abs(true_curve - pred_curve) / (np.abs(true_curve) + 1e-6)
        rel_errors.append(np.mean(rel_error))
    
    # Calculate statistics
    r2_mean = np.mean(r2_scores)
    r2_median = np.median(r2_scores)
    r2_std = np.std(r2_scores)
    rel_error_mean = np.mean(rel_errors)
    rel_error_std = np.std(rel_errors)
    
    print(f"\nTest Results (in original scale):")
    print(f"  Mean Squared Error: {mse:.2f}")
    print(f"  R² Statistics:")
    print(f"    Mean: {r2_mean:.4f} ± {r2_std:.4f}")
    print(f"    Median: {r2_median:.4f}")
    print(f"  Relative Error: {rel_error_mean:.4f} ± {rel_error_std:.4f}")
    
    return {
        'mse': mse,
        'r2_mean': r2_mean,
        'r2_median': r2_median,
        'r2_std': r2_std,
        'r2_scores': r2_scores,
        'rel_error_mean': rel_error_mean,
        'rel_error_std': rel_error_std,
        'true_curves': y_true,
        'pred_curves': y_pred
    }

def main():
    # Data paths
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
    
    # Prepare data
    data = prepare_data(train_input_paths, train_output_paths)
    X_train, y_train = data['train']
    X_test, y_test = data['test']
    input_scaler, output_scaler = data['scalers']
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Model configuration
    input_dim = X_train.shape[1]
    curve_dim = y_train.shape[1]
    emb_dim = 16
    
    # Initialize model
    model = JointEmbeddingModel(input_dim, curve_dim, emb_dim)
    
    # Training stage 1: Train autoencoders
    print("\n=== Stage 1: Training Autoencoders ===")
    model = train_autoencoders(model, train_loader)
    
    # Training stage 2: Joint training
    print("\n=== Stage 2: Joint Training ===")
    model = train_joint(model, train_loader, test_loader)
    
    # Save final model with scalers
    model.save_full_model(
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        model_config={'input_dim': input_dim, 'curve_dim': curve_dim, 'emb_dim': emb_dim}
    )
    
    # Test the model with proper scaling
    test_results = test_model(model, test_loader, output_scaler)
    
    # Plot example curves
    import matplotlib.pyplot as plt
    n_curves = len(test_results['true_curves'])
    sample_indices = np.random.choice(n_curves, size=3, replace=False)
    
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(sample_indices, 1):
        plt.subplot(1, 3, i)
        true_curve = test_results['true_curves'][idx]
        pred_curve = test_results['pred_curves'][idx]
        r2 = test_results['r2_scores'][idx]
        
        plt.plot(true_curve, label='True', color='blue', alpha=0.7)
        plt.plot(pred_curve, label='Predicted', color='red', linestyle='--', alpha=0.7)
        plt.title(f'Example Curve {i}\nR² = {r2:.4f}')
        plt.grid(True, alpha=0.3)
        if i == 1:
            plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()