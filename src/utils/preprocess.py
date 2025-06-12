import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from scalers import GlobalISCScaler, GlobalVocScaler
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence


def load_data(file_paths):
    """Load and stack data from multiple files"""
    data = [np.loadtxt(p, delimiter=",") for p in file_paths]
    return np.vstack(data)


def preprocess_data_with_eos(
    input_paths, output_paths, test_size=0.2, isc_scaler_method="median"
):
    """
    Filters out negative values, applies scaling using GlobalISCScaler.
    Pads IV curves to allow batching of variable-length sequences.
    NOTE: We pad with -1.0 to differentiate from actual data.
    Constructs EOS targets for each sequence, marking the end of the actual data.
    NOTE: EOS sequence is the same length as the IV curve with 1.0 at the last position (inclusive)

    Returns a dictionary with:
        - 'train': (X_train_tensor, padded_y_train_tensor, mask_train, lengths_train, eos_targets_train)
        - 'test': (X_test_tensor, padded_y_test_tensor, mask_test, lengths_test, eos_targets_test)
        - 'scalers': (input_scaler, output_scaler)
        - 'original_test_y': list of filtered test IV curves (unscaled, unpadded)
        - 'max_sequence_length': Maximum actual sequence length + 1 (for EOS)
    """
    epsilon = 1e-40

    print("\nSequential Data Loading and Preprocessing (with EOS):")
    # Load raw data
    X_data = load_data(input_paths)
    y_data = load_data(output_paths)
    print("Raw data ranges:")
    print(f"  Inputs: [{X_data.min():.2f}, {X_data.max():.2f}]")
    print(f"  IV Curves: [{y_data.min():.2f}, {y_data.max():.2f}]")

    # Split into train and test sets
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_data, y_data, test_size=test_size, random_state=42, shuffle=True
    )

    # Function to filter each IV curve up to its first negative value (if any)
    def filter_curve(curve):
        """Filter curve including the first negative value, then place EOS token"""
        neg_indices = np.where(curve < 0)[0]
        if neg_indices.size > 0:
            return curve[: neg_indices[0] + 1]  # Include the first negative value
        else:
            return curve

    # Process training IV curves
    filtered_train = [filter_curve(curve) for curve in y_train_raw]

    # Fit GlobalISCScaler on all ISC values from training data
    output_scaler = GlobalISCScaler(method=isc_scaler_method)
    train_isc_values = np.array([curve[0] for curve in filtered_train])
    output_scaler.fit(train_isc_values)

    # Apply GlobalISCScaler and prepare EOS token targets
    scaled_train_std = []
    eos_targets_train = []
    lengths_train = []

    for curve in filtered_train:
        scaled_curve = output_scaler.transform(curve)
        scaled_train_std.append(scaled_curve)

        # EOS token target: mark EOS at the last actual data point (inclusive)
        eos_target_for_curve = np.zeros(len(scaled_curve), dtype=np.float32)
        eos_target_for_curve[-1] = 1.0  # EOS at final value point
        eos_targets_train.append(eos_target_for_curve)
        lengths_train.append(len(scaled_curve))  # Number of timesteps (sequence length)

    # Convert to tensors and pad
    tensor_train = [
        torch.tensor(curve, dtype=torch.float32) for curve in scaled_train_std
    ]
    padded_y_train = pad_sequence(tensor_train, batch_first=True, padding_value=-1.0)
    # Pad EOS targets to the same sequence length
    max_len_train_actual = max(lengths_train)
    padded_eos_targets_train = torch.zeros(
        len(eos_targets_train), max_len_train_actual, dtype=torch.float32
    )
    for i, eos_target in enumerate(eos_targets_train):
        padded_eos_targets_train[i, : len(eos_target)] = torch.tensor(eos_target)

    # Process test IV curves similarly
    filtered_test = [filter_curve(curve) for curve in y_test_raw]
    scaled_test_std = []
    eos_targets_test = []
    lengths_test = []

    for curve in filtered_test:
        scaled_curve = output_scaler.transform(curve)
        scaled_test_std.append(scaled_curve)

        # mark EOS at last data point
        eos_target_for_curve = np.zeros(len(scaled_curve), dtype=np.float32)
        eos_target_for_curve[-1] = 1.0
        eos_targets_test.append(eos_target_for_curve)
        lengths_test.append(len(scaled_curve))

    tensor_test = [
        torch.tensor(curve, dtype=torch.float32) for curve in scaled_test_std
    ]
    padded_y_test = pad_sequence(tensor_test, batch_first=True, padding_value=-1.0)

    max_len_test_actual = max(lengths_test)
    padded_eos_targets_test = torch.zeros(
        len(eos_targets_test), max_len_test_actual, dtype=torch.float32
    )
    for i, eos_target in enumerate(eos_targets_test):
        padded_eos_targets_test[i, : len(eos_target)] = torch.tensor(eos_target)

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
    input_scaler = RobustScaler(quantile_range=(5, 95))
    X_train_scaled = input_scaler.fit_transform(X_train_log)
    X_test_scaled = input_scaler.transform(X_test_log)

    # Convert inputs to torch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    # Print diagnostic information
    print("\nSequence Length Statistics:")
    print(
        f"Train - min: {min(lengths_train)}, max: {max(lengths_train)}, mean: {np.mean(lengths_train):.1f}"
    )
    print(
        f"Test  - min: {min(lengths_test)}, max: {max(lengths_test)}, mean: {np.mean(lengths_test):.1f}"
    )
    print("\nTarget shapes after padding:")
    print(f"Train sequences: {padded_y_train.shape}")
    print(f"Train EOS targets: {padded_eos_targets_train.shape}")
    print(f"Test sequences: {padded_y_test.shape}")
    print(f"Test EOS targets: {padded_eos_targets_test.shape}")
    print(f"Global ISC value: {output_scaler.get_isc():.4f}")

    return {
        "train": (
            X_train_tensor,
            padded_y_train,
            mask_train,
            torch.tensor(lengths_train),
            padded_eos_targets_train,
        ),
        "test": (
            X_test_tensor,
            padded_y_test,
            mask_test,
            torch.tensor(lengths_test),
            padded_eos_targets_test,
        ),
        "scalers": (input_scaler, output_scaler),
        "original_test_y": filtered_test,
        "original_train_y": filtered_train,
        "max_sequence_length": max(max_len_train_actual, max_len_test_actual),
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
            X_data, y_data, test_size=test_size, random_state=42, shuffle=True
        )

    def filter_curve(curve):
        neg_indices = np.where(curve < 0)[0]
        return curve[: neg_indices[0] + 1] if neg_indices.size > 0 else curve

    # Process training data
    filtered_train = (
        [filter_curve(curve) for curve in y_train_raw] if len(y_train_raw) > 0 else []
    )
    lengths_train = [len(curve) for curve in filtered_train]
    # Process test data
    filtered_test = [filter_curve(curve) for curve in y_test_raw]
    lengths_test = [len(curve) for curve in filtered_test]

    # Fit scaler on all data when test_size=1.0, otherwise just on training data
    output_scaler = MinMaxScaler()
    if test_size == 1.0:
        all_values = np.concatenate(filtered_test)
    else:
        all_values = np.concatenate(filtered_train)
    output_scaler.fit(all_values.reshape(-1, 1))

    # Scale and pad training data (if any)
    if len(filtered_train) > 0:
        scaled_train = [
            output_scaler.transform(curve.reshape(-1, 1)).flatten()
            for curve in filtered_train
        ]
        tensor_train = [
            torch.tensor(curve, dtype=torch.float32) for curve in scaled_train
        ]
        padded_y_train = pad_sequence(
            tensor_train, batch_first=True, padding_value=-1.0
        )
    else:
        padded_y_train = torch.empty((0,))

    # Scale and pad test data
    scaled_test = [
        output_scaler.transform(curve.reshape(-1, 1)).flatten()
        for curve in filtered_test
    ]
    tensor_test = [torch.tensor(curve, dtype=torch.float32) for curve in scaled_test]
    padded_y_test = pad_sequence(tensor_test, batch_first=True, padding_value=-1.0)

    # Create masks if requested
    if return_masks:
        mask_train = (
            torch.zeros_like(padded_y_train)
            if len(filtered_train) > 0
            else torch.empty((0,))
        )
        mask_test = torch.zeros_like(padded_y_test)
        for i, length in enumerate(lengths_train):
            mask_train[i, :length] = 1.0
        for i, length in enumerate(lengths_test):
            mask_test[i, :length] = 1.0

    # Process input features
    input_scaler = RobustScaler(quantile_range=(5, 95))
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

    train_data = (
        (X_train_tensor, padded_y_train, mask_train, lengths_train)
        if return_masks
        else (X_train_tensor, padded_y_train, lengths_train)
    )
    test_data = (
        (X_test_tensor, padded_y_test, mask_test, lengths_test)
        if return_masks
        else (X_test_tensor, padded_y_test, lengths_test)
    )

    return {
        "train": train_data,
        "test": test_data,
        "scalers": (input_scaler, output_scaler),
        "original_test_y": filtered_test,
    }


def preprocess_fixed_length_dual_output(
    input_paths,
    output_paths,
    voltage_raw=None,
    num_pre=5,
    num_post=6,
    high_res_points=10000,
    current_thresh=None,
    round_digits=None,
    test_size=0.2,
    isc_scaler_method="median",
    voc_scaler_method="median",
):
    """
    Standalone preprocessing for dual-head models.
    NOTE: THIS ASSUMES THAT YOU ARE TRAINING A MODEL WITH TWO OUTPUT HEADS FOR CURRENT AND VOLTAGE.
    If you want to only output current, consider preprocess_fixed_length_common_axis instead.
    Preprocessing with only fixed length doesn't make sense for either use case because its voltage axis is different and not scaled.
    Returns both current and voltage targets as separate tensors.
    """
    # Load and stack data
    X_data = load_data(input_paths)
    y_data = load_data(output_paths)

    # Default voltage grid if not provided
    if voltage_raw is None:
        voltage_raw = np.concatenate(
            (np.arange(0, 0.41, 0.1), np.arange(0.425, 1.401, 0.025))
        )

    # High-res axis
    hi_V = np.linspace(voltage_raw.min(), voltage_raw.max(), high_res_points)

    def _reduce(curve):
        # High-res interpolation
        y_hi = np.interp(hi_V, voltage_raw, curve)

        # Voc (zero-cross)
        neg = np.where(y_hi < 0)[0]
        if len(neg) > 0:
            i0 = neg[0]
            V1 = hi_V[i0 - 1]
            V2 = hi_V[i0]
            y1 = y_hi[i0 - 1]
            y2 = y_hi[i0]
            Voc = V1 - y1 * (V2 - V1) / (y2 - y1)
        else:
            Voc = hi_V[-1]

        # MPP
        P = hi_V * y_hi
        mpp = np.argmax(P)
        Vmp = hi_V[mpp]

        # Sample grids
        v_pre = np.linspace(0, Vmp, num_pre)
        v_post = np.linspace(Vmp, Voc, num_post)

        # Values
        y_pre = np.interp(v_pre, hi_V, y_hi)
        y_post = np.interp(v_post, hi_V, y_hi)

        # Drop duplicate Vmp
        v_red = np.concatenate((v_pre, v_post[1:]))
        y_red = np.concatenate((y_pre, y_post[1:]))
        return v_red, y_red

    v_list, y_list, idxs = [], [], []
    for idx, curve in enumerate(y_data):
        v_red, y_red = _reduce(curve)

        # Outlier filter
        if current_thresh is not None:
            if np.nanmax(y_red) > current_thresh or np.nanmin(y_red) < -1:
                continue

        # Rounding
        if round_digits is not None:
            y_red = np.round(y_red, round_digits)

        v_list.append(v_red)
        y_list.append(y_red)
        idxs.append(idx)

    if not y_list:
        raise ValueError("No curves left after filtering")

    X_clean = X_data[idxs]
    Y_current_reduced = np.vstack(y_list)  # Physical current values
    V_voltage_reduced = np.vstack(v_list)  # Physical voltage positions

    isc_vals = Y_current_reduced[:, 0]

    # Split indices first
    indices = np.arange(len(X_clean))
    idx_train, idx_test = train_test_split(
        indices, test_size=test_size, random_state=42
    )

    # Split all data
    X_train_raw, X_test_raw = X_clean[idx_train], X_clean[idx_test]
    Y_current_train, Y_current_test = (
        Y_current_reduced[idx_train],
        Y_current_reduced[idx_test],
    )
    V_voltage_train, V_voltage_test = (
        V_voltage_reduced[idx_train],
        V_voltage_reduced[idx_test],
    )
    isc_train, isc_test = isc_vals[idx_train], isc_vals[idx_test]

    # Scale current values
    isc_scaler = GlobalISCScaler(method=isc_scaler_method)
    Y_current_train_scaled = isc_scaler.fit_transform(Y_current_train)
    Y_current_test_scaled = isc_scaler.transform(Y_current_test)

    # Scale voltage values using GlobalVocScaler
    voltage_scaler = GlobalVocScaler(method=voc_scaler_method)
    V_voltage_train_scaled = voltage_scaler.fit_transform(V_voltage_train)
    V_voltage_test_scaled = voltage_scaler.transform(V_voltage_test)

    # Process input features
    epsilon = 1e-40
    X_train_log = np.log10(X_train_raw + epsilon)
    input_scaler = RobustScaler(quantile_range=(5, 95))
    X_train_scaled = input_scaler.fit_transform(X_train_log)
    X_test_log = np.log10(X_test_raw + epsilon)
    X_test_scaled = input_scaler.transform(X_test_log)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    Y_current_train_tensor = torch.tensor(Y_current_train_scaled, dtype=torch.float32)
    Y_current_test_tensor = torch.tensor(Y_current_test_scaled, dtype=torch.float32)
    V_voltage_train_tensor = torch.tensor(V_voltage_train_scaled, dtype=torch.float32)
    V_voltage_test_tensor = torch.tensor(V_voltage_test_scaled, dtype=torch.float32)

    return {
        "train": (
            X_train_tensor,
            Y_current_train_tensor,
            V_voltage_train_tensor,
            isc_train,
        ),
        "test": (X_test_tensor, Y_current_test_tensor, V_voltage_test_tensor, isc_test),
        "scalers": (isc_scaler, input_scaler, voltage_scaler),
        "original_test_y": Y_current_test,
        "original_test_v": V_voltage_test,
    }


def preprocess_fixed_length_common_axis(
    input_paths,
    output_paths,
    voltage_raw=None,
    num_pre=3,
    num_post=4,
    high_res_points=10000,
    current_thresh=None,
    round_digits=None,
    test_size=0.2,
    isc_scaler_method="median",
):
    """
    This function is identical to preprocess_fixed_length, but it uses a common voltage grid
    for all curves, allowing for direct comparison of IV curves across different samples.
    High res interpolation (upsampling) -> downsampling to fixed length -> interp to common grid.

    Returns:
        Dictionary with:
        - 'train': (X_train_tensor, Y_train_tensor, isc_train)
        - 'test': (X_test_tensor, Y_test_tensor, isc_test)
        - 'scalers': (isc_scaler, input_scaler)
        - 'v_common': common voltage grid for all curves
        - 'original_test_y': original test curves in physical units
    """
    # load and stack data
    X_data = load_data(input_paths)
    y_data = load_data(output_paths)

    # default voltage grid if not provided
    if voltage_raw is None:
        voltage_raw = np.concatenate(
            (np.arange(0, 0.41, 0.1), np.arange(0.425, 1.401, 0.025))
        )
    # high-res axis
    hi_V = np.linspace(voltage_raw.min(), voltage_raw.max(), high_res_points)
    # common fixed voltage grid for all curves
    v_common = np.linspace(voltage_raw.min(), voltage_raw.max(), num_pre + num_post - 1)

    def _reduce(curve):
        # high-res interpolation
        y_hi = np.interp(hi_V, voltage_raw, curve)
        # Voc (zero-cross)
        neg = np.where(y_hi < 0)[0]
        if len(neg) > 0:
            i0 = neg[0]
            V1 = hi_V[i0 - 1]
            V2 = hi_V[i0]
            y1 = y_hi[i0 - 1]
            y2 = y_hi[i0]
            Voc = V1 - y1 * (V2 - V1) / (y2 - y1)
        else:
            Voc = hi_V[-1]
        # MPP
        P = hi_V * y_hi
        mpp = np.argmax(P)
        Vmp = hi_V[mpp]
        # sample grids
        v_pre = np.linspace(0, Vmp, num_pre)
        v_post = np.linspace(Vmp, Voc, num_post)
        # values
        y_pre = np.interp(v_pre, hi_V, y_hi)
        y_post = np.interp(v_post, hi_V, y_hi)
        # drop duplicate Vmp
        v_red = np.concatenate((v_pre, v_post[1:]))
        y_red = np.concatenate((y_pre, y_post[1:]))
        return v_red, y_red

    y_list, idxs = [], []
    for idx, curve in enumerate(y_data):
        v_red, y_red = _reduce(curve)
        # outlier filter
        if current_thresh is not None:
            if np.nanmax(y_red) > current_thresh or np.nanmin(y_red) < -1:
                continue
        # rounding
        if round_digits is not None:
            y_red = np.round(y_red, round_digits)
        # resample onto common fixed grid
        y_fixed = np.interp(v_common, v_red, y_red)
        y_list.append(y_fixed)
        idxs.append(idx)

    if not y_list:
        raise ValueError("No curves left after filtering")
    X_clean = X_data[idxs]
    Y_reduced = np.vstack(y_list)  # physical current values

    isc_vals = Y_reduced[:, 0]

    # Split indices first
    indices = np.arange(len(X_clean))
    idx_train, idx_test = train_test_split(
        indices, test_size=test_size, random_state=42
    )

    # Split all data
    X_train_raw, X_test_raw = X_clean[idx_train], X_clean[idx_test]
    Y_train_original, Y_test_original = Y_reduced[idx_train], Y_reduced[idx_test]
    isc_train, isc_test = isc_vals[idx_train], isc_vals[idx_test]

    isc_scaler = GlobalISCScaler(method=isc_scaler_method)
    # Fit on training data only
    Y_train_scaled = isc_scaler.fit_transform(Y_train_original)
    Y_test_scaled = isc_scaler.transform(Y_test_original)

    # Process input features
    epsilon = 1e-40
    X_train_log = np.log10(X_train_raw + epsilon)
    input_scaler = RobustScaler(quantile_range=(5, 95))
    X_train_scaled = input_scaler.fit_transform(X_train_log)
    X_test_log = np.log10(X_test_raw + epsilon)
    X_test_scaled = input_scaler.transform(X_test_log)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test_scaled, dtype=torch.float32)

    return {
        "train": (X_train_tensor, Y_train_tensor, isc_train),
        "test": (X_test_tensor, Y_test_tensor, isc_test),
        "scalers": (isc_scaler, input_scaler),
        "v_common": v_common,
        "original_test_y": Y_test_original,  # physical curves
    }


# NOTE: This is very inefficient but we need to wait until the third major refactoring to improve this ig...
def test_and_visualize_preprocessing(
    preprocess_func, input_paths, output_paths, func_name, num_samples=5, **kwargs
):
    """
    Generic function to test and visualize any preprocessing function.

    Args:
        preprocess_func: Function pointer to the preprocessing function
        input_paths: List of input file paths
        output_paths: List of output file paths
        func_name: Name of the function for display purposes
        num_samples: Number of samples to plot
        **kwargs: Additional arguments to pass to the preprocessing function
    """
    print(f"\nTesting {func_name}:")

    # Call the preprocessing function
    data = preprocess_func(input_paths, output_paths, **kwargs)

    # Handle different return formats
    if func_name == "preprocess_data_with_eos":
        X_train, y_train, _, _, _ = data["train"]
        scalers = data["scalers"]
        original_data = data["original_train_y"]
        scaler = scalers[1]  # output scaler

        # Plot EOS data
        plt.figure(figsize=(12, 6))

        # Scaled curves
        plt.subplot(1, 2, 1)
        for i in range(min(num_samples, len(y_train))):
            length = len(original_data[i])
            scaled_curve = y_train[i, :length].numpy()
            plt.plot(scaled_curve, label=f"Sample {i + 1}")
        plt.title(f"{func_name} - Scaled Curves")
        plt.xlabel("Index")
        plt.ylabel("Scaled Current")
        plt.legend()
        plt.grid(True)

        # Original curves
        plt.subplot(1, 2, 2)
        for i in range(min(num_samples, len(original_data))):
            plt.plot(original_data[i], label=f"Sample {i + 1}")
        plt.title(f"{func_name} - Original Curves")
        plt.xlabel("Index")
        plt.ylabel("Current Density (A/m²)")
        plt.legend()
        plt.grid(True)

    elif func_name == "preprocess_fixed_length_common_axis":
        X_train, y_train, _ = data["train"]
        scalers = data["scalers"]
        v_common = data["v_common"]
        scaler = scalers[0]  # isc_scaler

        plt.figure(figsize=(12, 6))

        # Scaled curves
        plt.subplot(1, 2, 1)
        for i in range(min(num_samples, len(y_train))):
            scaled_curve = y_train[i].numpy()
            plt.plot(v_common, scaled_curve, label=f"Sample {i + 1}")
        plt.title(f"{func_name} - Scaled Curves")
        plt.xlabel("Voltage (V)")
        plt.ylabel("Scaled Current")
        plt.legend()
        plt.grid(True)

        # Original curves
        plt.subplot(1, 2, 2)
        for i in range(min(num_samples, len(y_train))):
            original_curve = scaler.inverse_transform(
                y_train[i].numpy().reshape(-1, 1)
            ).flatten()
            plt.plot(v_common, original_curve, label=f"Sample {i + 1}")
        plt.title(f"{func_name} - Original Curves")
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current Density (A/m²)")
        plt.legend()
        plt.grid(True)

    elif func_name == "preprocess_fixed_length_dual_output":
        X_train, y_current_train, v_voltage_train, _ = data["train"]
        isc_scaler, _, voltage_scaler = data["scalers"]

        plt.figure(figsize=(12, 6))

        # Scaled IV curves
        plt.subplot(1, 2, 1)
        for i in range(min(num_samples, len(y_current_train))):
            scaled_current = y_current_train[i].numpy()
            scaled_voltage = v_voltage_train[i].numpy()
            plt.plot(scaled_voltage, scaled_current, label=f"Sample {i + 1}")
        plt.title(f"{func_name} - Scaled IV Curves")
        plt.xlabel("Scaled Voltage [0,1]")
        plt.ylabel("Scaled Current [-1,1]")
        plt.legend()
        plt.grid(True)

        # Original IV curves
        plt.subplot(1, 2, 2)
        for i in range(min(num_samples, len(y_current_train))):
            current = isc_scaler.inverse_transform(
                y_current_train[i].numpy().reshape(1, -1)
            ).flatten()
            voltage = voltage_scaler.inverse_transform(
                v_voltage_train[i].numpy().reshape(1, -1)
            ).flatten()
            plt.plot(voltage, current, label=f"Sample {i + 1}")
        plt.title(f"{func_name} - Original IV Curves")
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current Density (A/m²)")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("Data shapes:")
    print(f"  X_train: {X_train.shape}")
    if func_name == "preprocess_fixed_length_dual_output":
        print(f"  Y_current_train: {y_current_train.shape}")
        print(f"  V_voltage_train: {v_voltage_train.shape}")
        print(f"  Global ISC: {isc_scaler.get_isc():.4f}")
        print(f"  Global Voc: {voltage_scaler.get_voc():.4f}")
    else:
        print(f"  Y_train: {y_train.shape}")
        if hasattr(scaler, "get_isc"):
            print(f"  Global ISC: {scaler.get_isc():.4f}")


if __name__ == "__main__":
    train_input_paths = ["dataset/Data_1k_sets/Data_1k_rng1/LHS_parameters_m.txt"]
    train_output_paths = ["dataset/Data_1k_sets/Data_1k_rng1/iV_m.txt"]

    # Test all preprocessing functions
    # test_and_visualize_preprocessing(
    #     preprocess_data_with_eos,
    #     train_input_paths,
    #     train_output_paths,
    #     "preprocess_data_with_eos",
    #     test_size=0.2,
    #     isc_scaler_method="median",
    # )

    # test_and_visualize_preprocessing(
    #     preprocess_fixed_length_common_axis,
    #     train_input_paths,
    #     train_output_paths,
    #     "preprocess_fixed_length_common_axis",
    #     num_pre=5,
    #     num_post=6,
    #     high_res_points=10000,
    #     round_digits=2,
    #     test_size=0.2,
    #     isc_scaler_method="median",
    # )

    test_and_visualize_preprocessing(
        preprocess_fixed_length_dual_output,
        train_input_paths,
        train_output_paths,
        "preprocess_fixed_length_dual_output",
        num_pre=5,
        num_post=6,
        high_res_points=10000,
        round_digits=2,
        test_size=0.2,
        isc_scaler_method="median",
        voc_scaler_method="median",
    )
