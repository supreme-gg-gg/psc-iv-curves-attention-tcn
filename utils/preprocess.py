import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, QuantileTransformer
import matplotlib.pyplot as plt
import seaborn as sns

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
        X_data, y_data, test_size=test_size, random_state=42, shuffle=True)
    
    # Preprocessing input parameters with log transform
    epsilon = 1e-40
    X_train_log = np.log10(X_train_raw + epsilon)
    X_test_log = np.log10(X_test_raw + epsilon)
    
    print(f"\nAfter log transform (inputs):")
    print(f"  Train: [{X_train_log.min():.2f}, {X_train_log.max():.2f}]")
    print(f"  Test: [{X_test_log.min():.2f}, {X_test_log.max():.2f}]")
    
    # Use RobustScaler with quantile_range to handle outliers
    input_scaler = RobustScaler(quantile_range=(5, 95))

    scale_factor_I = 150.0 # Adjust this based on domain knowledge / data exploration

    # Apply arcsinh transformation for IV curves
    y_train_arcsinh = np.arcsinh(y_train_raw / scale_factor_I)
    y_test_arcsinh = np.arcsinh(y_test_raw / scale_factor_I)
    
    # Use QuantileTransformer for output normalization
    output_transformer = QuantileTransformer(output_distribution='normal',
                                          n_quantiles=min(len(y_train_raw), 1000))

    # Transform data
    X_train_scaled = input_scaler.fit_transform(X_train_log)
    X_test_scaled = input_scaler.transform(X_test_log)
    
    # Transform outputs
    y_train_scaled = output_transformer.fit_transform(y_train_arcsinh)
    y_test_scaled = output_transformer.transform(y_test_arcsinh)
    
    print(f"\nAfter scaling:")
    print(f"Inputs - Train: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
    print(f"Inputs - Test: [{X_test_scaled.min():.2f}, {X_test_scaled.max():.2f}]")
    print(f"IV Curves - Train: [{y_train_scaled.min():.2f}, {y_train_scaled.max():.2f}]")
    print(f"IV Curves - Test: [{y_test_scaled.min():.2f}, {y_test_scaled.max():.2f}]")
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)
    
    return {
        'train': (X_train_tensor, y_train_tensor),
        'test': (X_test_tensor, y_test_tensor),
        'scalers': (input_scaler, output_transformer),
        'original_test_data_y': y_test_raw  # Keep original test data for plotting
    }

if __name__ == "__main__":
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
    X_train_tensor, y_train_tensor = data['train']
    X_test_tensor, y_test_tensor = data['test']
    input_physical_scaler, output_iv_scaler = data['scalers']
    y_test_original_scale = data['original_test_data_y'] # For plotting during inference
    
    # Data validation and visualization
    
    # Basic statistics
    print("\nData Shape Summary:")
    print(f"X_train: {X_train_tensor.shape}")
    print(f"y_train: {y_train_tensor.shape}")
    print(f"X_test: {X_test_tensor.shape}")
    print(f"y_test: {y_test_tensor.shape}")
    
    # Create two separate figures for better visibility
    # Figure 1: Original diagnostic plots
    plt.figure(figsize=(15, 10))
    
    # Input distribution boxplot
    plt.subplot(2, 2, 1)
    sns.boxplot(data=X_train_tensor.numpy())
    plt.title('Distribution of Scaled Input Features (Train)')
    plt.xticks(rotation=45)
    
    # Sample IV curves (original)
    plt.subplot(2, 2, 2)
    for i in range(5):
        plt.plot(y_test_original_scale[i], alpha=0.7, label=f'Sample {i+1}')
    plt.title('Sample Original IV Curves (Test)')
    plt.legend()

    # Sample IV curves (scaled)
    plt.subplot(2, 2, 3)
    for i in range(5):
        plt.plot(y_test_tensor.numpy()[i], alpha=0.7, label=f'Sample {i+1}')
    plt.title('Sample Scaled IV Curves (Test)')

    # Scaling verification
    plt.subplot(2, 2, 4)
    sample_idx = np.random.randint(0, len(y_test_tensor))
    scaled_curve = y_test_tensor[3].numpy()
    unscaled_curve = output_iv_scaler.inverse_transform(scaled_curve.reshape(1, -1))[0]
    unscaled_curve = np.sinh(unscaled_curve) * 150.0
    original_curve = y_test_original_scale[3]
    plt.plot(original_curve, label='Original', linestyle='--')
    plt.plot(unscaled_curve, label='Unscaled from scaled', alpha=0.7)
    plt.title('Scaling Verification')
    plt.legend()
    plt.tight_layout()
    
    # Figure 2: Histogram distributions
    plt.figure(figsize=(15, 8))
    
    # Input features histograms
    plt.subplot(2, 1, 1)
    X_train_np = X_train_tensor.numpy()
    for i in range(X_train_np.shape[1]):
        plt.hist(X_train_np[:, i], bins=50, alpha=0.5, label=f'Feature {i+1}')
    plt.title('Distribution of Scaled Input Features')
    plt.xlabel('Value')
    plt.ylabel('Count')
    # plt.legend()
    
    # IV curves histogram (using a few selected points to avoid overcrowding)
    plt.subplot(2, 1, 2)
    y_train_np = y_train_tensor.numpy()
    selected_points = [0, y_train_np.shape[1]//2, -1]  # Start, middle, end points
    for idx in selected_points:
        plt.hist(y_train_np[:, idx], bins=50, alpha=0.5, 
                label=f'Current at point {idx}')
    plt.title('Distribution of Scaled IV Curves at Selected Points')
    plt.xlabel('Scaled Current')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Check for potential issues
    print("\nData Quality Checks:")
    print(f"Input range: [{X_train_tensor.min():.2f}, {X_train_tensor.max():.2f}]")
    print(f"Output range: [{y_train_tensor.min():.2f}, {y_train_tensor.max():.2f}]")
    print(f"Test input range: [{X_test_tensor.min():.2f}, {X_test_tensor.max():.2f}]")
    print(f"Test output range: [{y_test_tensor.min():.2f}, {y_test_tensor.max():.2f}]")
