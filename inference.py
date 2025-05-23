import numpy as np
import torch
from models.jem import JointEmbeddingModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np

def load_trained_autoencoders(input_dim, curve_dim, emb_dim=5, device='cpu'):
    """Load pretrained autoencoders only"""
    model = JointEmbeddingModel(input_dim, curve_dim, emb_dim).to(device)
    model.load_autoencoders()
    return model

def load_trained_joint_model(device='cpu'):
    """Load full trained joint embedding model with scalers"""
    checkpoint = torch.load("checkpoints/jem/model.pth")
    config = checkpoint['model_config']
    
    model = JointEmbeddingModel(
        input_dim=config['input_dim'],
        curve_dim=config['curve_dim'],
        emb_dim=config['emb_dim']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['input_scaler'], checkpoint['output_scaler']

def generate_iv_curve(input_params, model, input_scaler, output_scaler):
    """Generate IV curve from input parameters (in original scale)"""
    epsilon = 1e-40
    
    # Preprocess input
    input_log = np.log10(input_params + epsilon)
    input_scaled = input_scaler.transform(input_log.reshape(1, -1))
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    
    # Generate curve
    generated_curve_scaled = model.generate_curve(input_tensor)
    
    # Inverse transform to original scale
    generated_curve = output_scaler.inverse_transform(generated_curve_scaled.numpy())
    return generated_curve[0]

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model performance on test set"""
    model.eval()
    all_generated_curves = []
    all_true_curves = []
    
    with torch.no_grad():
        for input_data, true_curve in test_loader:
            input_data = input_data.to(device)
            true_curve = true_curve.to(device)
            
            # Generate curve
            generated_curve = model.generate_curve(input_data)
            
            all_generated_curves.append(generated_curve.cpu().numpy())
            all_true_curves.append(true_curve.cpu().numpy())
    
    # Convert to numpy arrays
    all_generated_curves = np.vstack(all_generated_curves)
    all_true_curves = np.vstack(all_true_curves)
    
    # Calculate metrics
    mse = mean_squared_error(all_true_curves, all_generated_curves)
    
    # Calculate per-curve R² scores and analyze prediction quality
    r2_scores = []
    rel_errors = []
    neg_predictions = []
    max_deviations = []
    
    for true, pred in zip(all_true_curves, all_generated_curves):
        r2_scores.append(r2_score(true, pred))
        
        # Calculate relative error, handling division by small values
        rel_error = np.abs(true - pred) / (np.abs(true) + 1e-6)
        rel_errors.append(np.mean(rel_error))
        
        # Track negative predictions and their magnitude
        neg_mask = pred < 0
        if neg_mask.any():
            neg_predictions.append(np.min(pred))
        
        # Track maximum deviation from true values
        max_deviation = np.max(np.abs(true - pred))
        max_deviations.append(max_deviation)
    
    # Calculate mean relative error and R² statistics
    # Calculate statistics
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)
    median_r2 = np.median(r2_scores)
    mean_rel_error = np.mean(rel_errors)
    max_deviation = np.max(max_deviations)
    
    # Analyze prediction quality
    n_negative_curves = len(neg_predictions)
    worst_negative = min(neg_predictions) if neg_predictions else 0
    
    return {
        'mse': mse,
        'r2_mean': mean_r2,
        'r2_std': std_r2,
        'r2_median': median_r2,
        'relative_error': mean_rel_error,
        'per_curve_r2': r2_scores,
        'per_curve_rel_error': rel_errors,
        'predictions': all_generated_curves,
        'true_values': all_true_curves,
        'max_deviation': max_deviation,
        'n_negative_curves': n_negative_curves,
        'worst_negative': worst_negative
    }

def visualize_jem_embeddings(model, dataloader, device='cpu', n_samples=500, use_tsne=True, perplexity_val=30):
    """
    Visualizes the aligned embeddings from the Joint Embedding Model.
    Args:
        model (JointEmbeddingModel): The trained JEM model.
        dataloader (DataLoader): DataLoader for input and curve data.
        device (str): Device to run the model on ('cpu' or 'cuda').
        n_samples (int): Approximate number of samples to visualize.
        use_tsne (bool): If True, use t-SNE. Otherwise, use PCA.
        perplexity_val (int): Perplexity value for t-SNE.
    """
    model.eval()
    all_paired_emb_input = []
    all_paired_emb_curve = []
    
    count = 0
    with torch.no_grad():
        for input_data_batch, curve_data_batch in dataloader:
            if count >= n_samples:
                break
            
            input_data_batch = input_data_batch.to(device)
            curve_data_batch = curve_data_batch.to(device)
            
            # Get embeddings from the full JEM forward pass
            outputs = model(input_data_batch, curve_data_batch) 
            
            samples_to_take_this_batch = min(input_data_batch.size(0), n_samples - count)
            
            all_paired_emb_input.append(outputs['emb_input'][:samples_to_take_this_batch].cpu().numpy())
            all_paired_emb_curve.append(outputs['emb_curve'][:samples_to_take_this_batch].cpu().numpy())
            
            count += samples_to_take_this_batch

    if not all_paired_emb_input or not all_paired_emb_curve:
        print("No embeddings collected. Check dataloader or n_samples.")
        return

    all_paired_emb_input = np.concatenate(all_paired_emb_input, axis=0)
    all_paired_emb_curve = np.concatenate(all_paired_emb_curve, axis=0)
    
    actual_samples = all_paired_emb_input.shape[0]
    
    if actual_samples <= 1:
        print(f"Not enough samples ({actual_samples}) to visualize embeddings.")
        return

    # Combine for joint visualization to see alignment
    combined_embeddings = np.vstack((all_paired_emb_input, all_paired_emb_curve))
    # Create labels: 0 for input embeddings, 1 for curve embeddings
    labels = np.array([0] * actual_samples + [1] * actual_samples) 

    reducer_name = ''
    if use_tsne:
        # Adjust perplexity if the number of samples is too low
        # t-SNE perplexity should be less than the number of samples, and usually > 5 for good results.
        effective_perplexity = min(perplexity_val, (actual_samples * 2) - 1) if (actual_samples * 2) > 1 else 5
        if effective_perplexity <=0 : effective_perplexity = 1 # Failsafe for extremely few samples.
        
        reducer = TSNE(n_components=2, random_state=42, perplexity=effective_perplexity, n_iter=300, init='pca', learning_rate='auto')
        reducer_name = 't-SNE'
    else:
        reducer = PCA(n_components=2, random_state=42)
        reducer_name = 'PCA'
        
    print(f"Running {reducer_name} on {actual_samples*2} combined embedding samples (from {actual_samples} pairs)...")
    combined_embeddings_2d = reducer.fit_transform(combined_embeddings)
    
    emb_input_2d_from_combined = combined_embeddings_2d[:actual_samples]
    emb_curve_2d_from_combined = combined_embeddings_2d[actual_samples:]

    plt.figure(figsize=(10, 8))
    scatter_input = plt.scatter(emb_input_2d_from_combined[:, 0], emb_input_2d_from_combined[:, 1], label='Input Embeddings (aligned)', alpha=0.6, s=15, c='blue')
    scatter_curve = plt.scatter(emb_curve_2d_from_combined[:, 0], emb_curve_2d_from_combined[:, 1], label='Curve Embeddings (aligned)', alpha=0.6, s=15, c='red')
    plt.title(f'{reducer_name} of Aligned Input and Curve Embeddings')
    plt.xlabel(f'{reducer_name} Dimension 1')
    plt.ylabel(f'{reducer_name} Dimension 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path_combined = f'{reducer_name.lower()}_aligned_embeddings_visualization.png'
    plt.savefig(save_path_combined)
    print(f"Saved combined embedding visualization to {save_path_combined}")
    plt.show()

def main():
    # Example: Load full model and evaluate
    print("Loading trained model...")
    model, input_scaler, output_scaler = load_trained_joint_model()
    
    # Load some test data
    test_input_path = "dataset/Data_1k_sets/Data_1k_rng1/LHS_parameters_m.txt"
    test_output_path = "dataset/Data_1k_sets/Data_1k_rng1/iV_m.txt"
    
    X_test = np.loadtxt(test_input_path, delimiter=',')
    y_test = np.loadtxt(test_output_path, delimiter=',')
    
    # Preprocess test data
    epsilon = 1e-40
    X_test_log = np.log10(X_test + epsilon)
    X_test_scaled = input_scaler.transform(X_test_log)
    y_test_scaled = output_scaler.transform(y_test)
    
    # Create test loader
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    # Evaluate
    print("\nEvaluating model performance...")
    results = evaluate_model(model, test_loader)
    
    print(f"\nTest Set Performance:")
    print(f"MSE: {results['mse']:.6f}")
    print(f"R² scores:")
    print(f"  Mean: {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
    print(f"  Median: {results['r2_median']:.4f}")
    print(f"Mean Relative Error: {results['relative_error']:.4f}")
    print(f"\nPrediction Quality Metrics:")
    print(f"  Maximum deviation from true values: {results['max_deviation']:.4f}")
    print(f"  Number of curves with negative predictions: {results['n_negative_curves']}")
    if results['n_negative_curves'] > 0:
        print(f"  Most negative prediction: {results['worst_negative']:.4f}")
    
    # Display histogram of R² scores
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(results['per_curve_r2'], bins=50)
    plt.title('Distribution of R² Scores Across Test Curves')
    plt.xlabel('R² Score')
    plt.ylabel('Count')
    plt.savefig('r2_distribution.png')
    plt.close()

    # Visualize random sample of curves
    true_curves = results['true_values']
    pred_curves = results['predictions']
    n_curves = len(true_curves)
    
    # Randomly select 3 curves
    np.random.seed(42)  # for reproducibility
    sample_indices = np.random.choice(n_curves, size=3, replace=False)
    
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(sample_indices, 1):
        plt.subplot(1, 3, i)
        
        # Get scaled values and inverse transform them
        true_curve = output_scaler.inverse_transform(true_curves[idx].reshape(1, -1))[0]
        pred_curve = output_scaler.inverse_transform(pred_curves[idx].reshape(1, -1))[0]

        # don't scale
        # true_curve = true_curves[idx]
        # pred_curve = pred_curves[idx]

        r2 = results['per_curve_r2'][idx]
        rel_err = results['per_curve_rel_error'][idx]
        
        # Plot
        plt.plot(true_curve, label='True', color='blue', alpha=0.7)
        plt.plot(pred_curve, label='Predicted', color='red', linestyle='--', alpha=0.7)
        plt.title(f'Curve {idx}\nR²: {r2:.4f}, RelErr: {rel_err:.4f}')
        plt.grid(True, alpha=0.3)
        if i == 1:  # Only show legend for first plot
            plt.legend()
    
    plt.tight_layout()
    plt.show()

    print("\nVisualizing embeddings...")
    if test_loader is not None and model is not None:
        visualize_jem_embeddings(model, test_loader, n_samples=500, use_tsne=True)
    else:
        print("Skipping embedding visualization as model or test_loader is not available.")
    
if __name__ == '__main__':
    main()
