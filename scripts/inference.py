import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.config import load_yaml_config, get_device
from models.cvae import CVAE
from models.seq_iv_model import SeqIVModel
from data.base import IVCurveDataModule

def parse_args():
    parser = argparse.ArgumentParser(description='Generate IV curves using trained models')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to model configuration file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Directory to save generated curves'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of samples to generate per input'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'mps'],
        help='Device to run inference on'
    )
    return parser.parse_args()

def load_model_and_data(args, config):
    """Load model, configuration and prepare data."""
    device = get_device(args.device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = config['model']
    
    # Initialize appropriate model
    if model_config['type'] == 'cvae':
        model = CVAE(
            physical_dim=model_config['physical_dim'],
            va_sweep_dim=model_config['va_sweep_dim'],
            latent_dim=model_config['latent_dim'],
            output_iv_dim=model_config['va_sweep_dim']
        )
    else:  # sequence
        model = SeqIVModel(
            physical_dim=model_config['physical_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config.get('num_layers', 1),
            dropout=model_config.get('dropout', 0.2)
        )
    
    # Load state dict and move to device
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Prepare data
    data_config = config['data']
    datamodule = IVCurveDataModule(
        input_paths=data_config['input_paths'],
        output_paths=data_config['output_paths'],
        mode='fixed' if model_config['type'] == 'cvae' else 'variable',
        test_size=data_config['test_size'],
        batch_size=1  # Generate one at a time
    )
    data = datamodule.setup()
    
    return model, data, device

def generate_curves(model, physical_params, num_samples, device, model_type='cvae'):
    """Generate multiple IV curves for given physical parameters."""
    physical_params = physical_params.to(device)
    
    with torch.no_grad():
        if model_type == 'cvae':
            # For CVAE, sample from latent space multiple times
            all_curves = []
            for _ in range(num_samples):
                curves, _, _ = model(physical_params, None)  # No target during inference
                all_curves.append(curves)
            generated = torch.stack(all_curves, dim=1)  # (batch, num_samples, points)
        else:
            # For sequence model, generate multiple sequences
            all_curves = []
            for _ in range(num_samples):
                curves = model(physical_params, teacher_forcing_ratio=0.0)
                all_curves.append(curves)
            generated = torch.stack(all_curves, dim=1)
    
    return generated

def plot_generated_curves(true_curves, generated_curves, output_path, voltage_points=None):
    """Plot and save comparison of true and generated curves."""
    num_samples = generated_curves.shape[1]
    
    plt.figure(figsize=(12, 8))
    
    if voltage_points is None:
        voltage_points = np.arange(len(true_curves[0]))
    
    # Plot true curve
    plt.plot(voltage_points, true_curves[0], 'k-', label='True', linewidth=2)
    
    # Plot generated curves
    for i in range(num_samples):
        plt.plot(voltage_points, generated_curves[0, i], '--', 
                label=f'Generated {i+1}', alpha=0.6)
    
    plt.xlabel('Applied Voltage (V)')
    plt.ylabel('Current Density (A/m²)')
    plt.title('True vs Generated IV Curves')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(output_path)
    plt.close()

def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    model, data, device = load_model_and_data(args, config)
    test_loader = data['test_loader']
    input_scaler, output_scaler = data['scalers']
    
    # Generate and plot curves for a few test samples
    for i, batch in enumerate(test_loader):
        if i >= 5:  # Generate for first 5 test samples
            break
            
        if len(batch) == 2:  # CVAE data format
            physical_params, true_curves = batch
            masks = None
        else:  # Sequence model data format
            physical_params, true_curves, masks = batch
        
        # Generate curves
        generated = generate_curves(
            model, physical_params, args.num_samples, device, 
            model_type=config['model']['type']
        )
        
        # Convert to numpy and inverse transform
        true_np = output_scaler.inverse_transform(
            true_curves[0].cpu().numpy().reshape(-1, 1)
        ).flatten()
        
        gen_np = output_scaler.inverse_transform(
            generated[0].cpu().numpy().reshape(-1, 1)
        ).reshape(args.num_samples, -1)
        
        # Plot and save
        plot_generated_curves(
            true_np[None, :],  # Add batch dimension
            gen_np[None, :, :],  # Add batch dimension
            output_dir / f'sample_{i+1}.png'
        )
        
        print(f"Generated curves for sample {i+1}")
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()