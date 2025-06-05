import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, QuantileTransformer, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.utils.rnn import pad_sequence
import random

def load_data(file_paths):
    """Load and stack data from multiple files"""
    data = [np.loadtxt(p, delimiter=',') for p in file_paths]
    return np.vstack(data)

@DeprecationWarning
def prepare_data(input_paths, output_paths, test_size=0.2):
    """
    Load and preprocess data with robust validation
    THIS FUNCTION IS DEPRECATED.
    This is designed for preprocessing output when the entire curve, including the negative part,
    is used for training. It also does not consider the varying length of the positive IV curves.
    """
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

def preprocess_data_with_eos(input_paths, output_paths, test_size=0.2):
    """
    Filters out the negative part of each IV curve.
    Applies scaling to both input and output data.
    Pads IV curves to allow batching of variable-length sequences.
    Appends an EOS token to each sequence.
    
    Returns a dictionary with:
        - 'train': (X_train_tensor, padded_y_train_tensor, lengths_train, eos_targets_train)
        - 'test': (X_test_tensor, padded_y_test_tensor, lengths_test, eos_targets_test)
        - 'scalers': (input_scaler, output_scaler)
        - 'original_test_y': list of filtered test IV curves (unscaled, unpadded)
        - 'max_sequence_length': Maximum actual sequence length + 1 (for EOS)
    """
    epsilon = 1e-40

    print("\nSequential Data Loading and Preprocessing (with EOS):")
    # Load raw data
    X_data = load_data(input_paths)
    y_data = load_data(output_paths)
    print(f"Raw data ranges:")
    print(f"  Inputs: [{X_data.min():.2f}, {X_data.max():.2f}]")
    print(f"  IV Curves: [{y_data.min():.2f}, {y_data.max():.2f}]")

    # Split into train and test sets
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_data, y_data, test_size=test_size, random_state=42, shuffle=True)

    # Function to filter each IV curve up to its first negative value (if any)
    def filter_curve(curve):
        """Filter curve including the first negative value, then place EOS token"""
        neg_indices = np.where(curve < 0)[0]
        if neg_indices.size > 0:
            return curve[:neg_indices[0] + 1]  # Include the first negative value
        else:
            return curve

    # Process training IV curves
    filtered_train = [filter_curve(curve) for curve in y_train_raw]
    
    # Fit StandardScaler on all scaled values
    output_scaler = RobustScaler()
    all_train_values = np.concatenate(filtered_train)
    output_scaler.fit(all_train_values.reshape(-1, 1))
    
    # Apply StandardScaler and prepare EOS token targets
    scaled_train_std = []
    eos_targets_train = []
    lengths_train = []
    
    for curve in filtered_train:
        scaled_curve = output_scaler.transform(curve.reshape(-1, 1)).flatten()
        scaled_train_std.append(scaled_curve)
        
        # EOS token target: 0 for all data points, 1 for the 'virtual' EOS point
        # EOS target: Place EOS token after the sequence (including negative value)
        eos_target_for_curve = np.zeros(len(scaled_curve) + 1, dtype=np.float32)
        eos_target_for_curve[len(scaled_curve)] = 1.0  # EOS after final value
        eos_targets_train.append(eos_target_for_curve)
        lengths_train.append(len(scaled_curve)) # Store original length for MSE

    # Convert to tensors and pad
    tensor_train = [torch.tensor(curve, dtype=torch.float32) for curve in scaled_train_std]
    padded_y_train = pad_sequence(tensor_train, batch_first=True, padding_value=0.0)
    
    # Pad EOS targets
    max_len_train_actual = max(lengths_train)
    # The padded EOS target sequence needs to be max_len + 1 because EOS is predicted *after* the last element.
    padded_eos_targets_train = torch.zeros(len(eos_targets_train), max_len_train_actual + 1, dtype=torch.float32)
    for i, eos_target in enumerate(eos_targets_train):
        padded_eos_targets_train[i, :len(eos_target)] = torch.tensor(eos_target)

    # Process test IV curves similarly
    filtered_test = [filter_curve(curve) for curve in y_test_raw]
    scaled_test_std = []
    eos_targets_test = []
    lengths_test = []

    for curve in filtered_test:
        scaled_curve = output_scaler.transform(curve.reshape(-1, 1)).flatten()
        scaled_test_std.append(scaled_curve)
        
        eos_target_for_curve = np.zeros(len(scaled_curve) + 1, dtype=np.float32)
        eos_target_for_curve[len(scaled_curve)] = 1.0
        eos_targets_test.append(eos_target_for_curve)
        lengths_test.append(len(scaled_curve))

    tensor_test = [torch.tensor(curve, dtype=torch.float32) for curve in scaled_test_std]
    padded_y_test = pad_sequence(tensor_test, batch_first=True, padding_value=0.0)

    max_len_test_actual = max(lengths_test)
    padded_eos_targets_test = torch.zeros(len(eos_targets_test), max_len_test_actual + 1, dtype=torch.float32)
    for i, eos_target in enumerate(eos_targets_test):
        padded_eos_targets_test[i, :len(eos_target)] = torch.tensor(eos_target)

    # Create masks for padded sequences (train & test)
    mask_train = torch.zeros_like(padded_y_train)
    mask_test = torch.zeros_like(padded_y_test)
    for i, l in enumerate(lengths_train):
        mask_train[i, :l] = 1.0
    for i, l in enumerate(lengths_test):
        mask_test[i, :l] = 1.0

    # Preprocess input features: logarithm transform then RobustScaler
    X_train_log = np.log10(X_train_raw + epsilon)
    X_test_log = np.log10(X_test_raw + epsilon)
    input_scaler = RobustScaler(quantile_range=(5,95))
    X_train_scaled = input_scaler.fit_transform(X_train_log)
    X_test_scaled = input_scaler.transform(X_test_log)

    # Convert inputs to torch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    # Print diagnostic information
    print("\nSequence Length Statistics:")
    print(f"Train - min: {min(lengths_train)}, max: {max(lengths_train)}, mean: {np.mean(lengths_train):.1f}")
    print(f"Test  - min: {min(lengths_test)}, max: {max(lengths_test)}, mean: {np.mean(lengths_test):.1f}")
    print(f"\nTarget shapes after padding:")
    print(f"Train sequences: {padded_y_train.shape}")
    print(f"Train EOS targets: {padded_eos_targets_train.shape}")
    print(f"Test sequences: {padded_y_test.shape}")
    print(f"Test EOS targets: {padded_eos_targets_test.shape}")

    return {
        'train': (X_train_tensor, padded_y_train, mask_train, torch.tensor(lengths_train), padded_eos_targets_train),
        'test': (X_test_tensor, padded_y_test, mask_test, torch.tensor(lengths_test), padded_eos_targets_test),
        'scalers': (input_scaler, output_scaler),
        'original_test_y': filtered_test,
        'max_sequence_length': max(max_len_train_actual, max_len_test_actual) + 1  # +1 for EOS position
    }

def preprocess_data_no_eos(input_paths, output_paths, test_size=0.2, return_masks=True):
    """
    Preprocesses data for sequence modeling WITHOUT EOS tokens.
    Essentially, the first negative value acts like an EOS token added to the end of each sequence.
    Each IV curve is truncated at the first negative value (inclusive),
    and the model is trained to predict only up to that point.
    Returns padded tensors and masks/lengths for batching.

    Returns:
    - 'train': (X_train_tensor, padded_y_train, mask_train, lengths_train)
    - 'test': (X_test_tensor, padded_y_test, mask_test, lengths_test)
    - 'scalers': (input_scaler, output_scaler)
    - 'original_test_y': list of filtered test IV curves (unscaled, unpadded)
    """
    epsilon = 1e-40
    # Load data
    X_data = load_data(input_paths)
    y_data = load_data(output_paths)

    if test_size == 1.0:
        X_test_raw = X_data
        y_test_raw = y_data
        X_train_raw = np.empty((0, X_data.shape[1]))
        y_train_raw = np.empty((0, y_data.shape[1]))
    else:
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X_data, y_data, test_size=test_size, random_state=42, shuffle=True)

    def filter_curve(curve):
        neg_indices = np.where(curve < 0)[0]
        return curve[:neg_indices[0]+1] if neg_indices.size > 0 else curve

    # Process training data
    filtered_train = [filter_curve(curve) for curve in y_train_raw] if len(y_train_raw) > 0 else []
    lengths_train = [len(curve) for curve in filtered_train]
    # Process test data
    filtered_test = [filter_curve(curve) for curve in y_test_raw]
    lengths_test = [len(curve) for curve in filtered_test]

    # Fit scaler on all data when test_size=1.0, otherwise just on training data
    output_scaler = RobustScaler()
    if test_size == 1.0:
        all_values = np.concatenate(filtered_test)
    else:
        all_values = np.concatenate(filtered_train)
    output_scaler.fit(all_values.reshape(-1, 1))

    # Scale and pad training data (if any)
    if len(filtered_train) > 0:
        scaled_train = [output_scaler.transform(curve.reshape(-1, 1)).flatten() for curve in filtered_train]
        tensor_train = [torch.tensor(curve, dtype=torch.float32) for curve in scaled_train]
        padded_y_train = pad_sequence(tensor_train, batch_first=True, padding_value=0.0)
    else:
        padded_y_train = torch.empty((0,))

    # Scale and pad test data
    scaled_test = [output_scaler.transform(curve.reshape(-1, 1)).flatten() for curve in filtered_test]
    tensor_test = [torch.tensor(curve, dtype=torch.float32) for curve in scaled_test]
    padded_y_test = pad_sequence(tensor_test, batch_first=True, padding_value=0.0)

    # Create masks if requested
    if return_masks:
        mask_train = torch.zeros_like(padded_y_train) if len(filtered_train) > 0 else torch.empty((0,))
        mask_test = torch.zeros_like(padded_y_test)
        for i, length in enumerate(lengths_train):
            mask_train[i, :length] = 1.0
        for i, length in enumerate(lengths_test):
            mask_test[i, :length] = 1.0

    # Process input features
    input_scaler = RobustScaler(quantile_range=(5,95))
    if test_size == 1.0:
        X_test_log = np.log10(X_test_raw + epsilon)
        X_test_scaled = input_scaler.fit_transform(X_test_log)
        X_train_scaled = np.empty((0, X_test_scaled.shape[1]))
    else:
        X_train_log = np.log10(X_train_raw + epsilon)
        X_test_log = np.log10(X_test_raw + epsilon)
        X_train_scaled = input_scaler.fit_transform(X_train_log)
        X_test_scaled = input_scaler.transform(X_test_log)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    train_data = (X_train_tensor, padded_y_train, mask_train, lengths_train) if return_masks else (X_train_tensor, padded_y_train, lengths_train)
    test_data = (X_test_tensor, padded_y_test, mask_test, lengths_test) if return_masks else (X_test_tensor, padded_y_test, lengths_test)

    # compute the transformed zero threshold for the output scaler
    scaled_zero_threshold = output_scaler.transform(np.array([[0]]))[0, 0]

    return {
        'train': train_data,
        'test': test_data,
        'scalers': (input_scaler, output_scaler),
        'original_test_y': filtered_test,
        'threshold': scaled_zero_threshold,
    }

if __name__ == "__main__":
    train_input_paths = [
        "../../dataset/Data_10k_sets/Data_10k_rng1/LHS_parameters_m.txt",
        "../../dataset/Data_10k_sets/Data_10k_rng2/LHS_parameters_m.txt",
        "../../dataset/Data_10k_sets/Data_10k_rng3/LHS_parameters_m.txt"
    ]

    train_output_paths = [
        "../../dataset/Data_10k_sets/Data_10k_rng1/iV_m.txt",
        "../../dataset/Data_10k_sets/Data_10k_rng2/iV_m.txt",
        "../../dataset/Data_10k_sets/Data_10k_rng3/iV_m.txt",
    ]
    
    # Process data using the new sequential preprocessing
    data = preprocess_data_no_eos(train_input_paths, train_output_paths, return_masks=False)
    (X_train_tensor, padded_y_train, lengths_train) = data['train']
    (X_test_tensor, padded_y_test, lengths_test) = data['test']
    input_scaler, output_scaler = data['scalers']
    filtered_test = data['original_test_y']
    
    # Print basic statistics
    print("\nData Shape Summary:")
    print(f"X_train: {X_train_tensor.shape}")
    print(f"Padded y_train: {padded_y_train.shape}")
    print(f"X_test: {X_test_tensor.shape}")
    print(f"Padded y_test: {padded_y_test.shape}")
    print(f"\nSequence lengths:")
    print(f"Train - min: {min(lengths_train)}, max: {max(lengths_train)}")
    print(f"Test - min: {min(lengths_test)}, max: {max(lengths_test)}")
    
    # Create visualization plots
    plt.figure(figsize=(15, 10))
    
    # Input feature distributions
    plt.subplot(2, 2, 1)
    sns.boxplot(data=X_train_tensor.numpy())
    plt.title('Distribution of Scaled Input Features')
    plt.xticks(rotation=45)
    
    # Original IV curves (few samples)
    plt.subplot(2, 2, 2)
    for i in range(5):
        plt.plot(filtered_test[i], alpha=0.7, label=f'Sample {i+1}')
    plt.title('Sample Original IV Curves (Test)')
    plt.xlabel('Index')
    plt.ylabel('Current Density (A/m^2)')
    plt.legend()
    
    # Scaled and padded IV curves
    plt.subplot(2, 2, 3)
    for i in range(5):
        valid_length = lengths_test[i]
        curve = padded_y_test[i, :valid_length]
        plt.plot(curve.numpy(), alpha=0.7, label=f'Sample {i+1}')
    plt.title('Scaled & Padded IV Curves (Test)')
    plt.xlabel('Index')
    plt.ylabel('Scaled Current')
    plt.legend()
    
    # Sequence length distribution
    plt.subplot(2, 2, 4)
    plt.hist(lengths_train, bins=30, alpha=0.5, label='Train')
    plt.hist(lengths_test, bins=30, alpha=0.5, label='Test')
    plt.title('Distribution of Sequence Lengths')
    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print ranges for quality check
    print("\nData Quality Checks:")
    print(f"Input range: [{X_train_tensor.min():.2f}, {X_train_tensor.max():.2f}]")
    print(f"Scaled IV range (excluding padding): [{padded_y_train[padded_y_train != 0].min():.2f}, {padded_y_train[padded_y_train != 0].max():.2f}]")

    # plot original vs new IV curve and the point of EOS
    sample_idx = random.randint(0, len(filtered_test))
    plt.figure(figsize=(10, 6))
    original_curve = filtered_test[sample_idx]
    scaled_curve = padded_y_test[sample_idx, :lengths_test[sample_idx]].numpy()
    unscaled_curve = output_scaler.inverse_transform(scaled_curve.reshape(-1, 1)).flatten()

    plt.plot(original_curve, label='Original Curve', linestyle='--')
    plt.plot(unscaled_curve, label='New Curve (Unscaled)', alpha=0.7)
    plt.title(f'Original vs. New IV Curve (Sample {sample_idx + 1})')
    plt.xlabel('Index')
    plt.ylabel('Current Density (A/m^2)')
    plt.legend()
    plt.grid(True)
    plt.axvline(x=lengths_test[sample_idx] - 1, color='red', linestyle=':', label=f'Crop Point (Index {lengths_test[sample_idx] - 1})')
    plt.legend()
    plt.show()
