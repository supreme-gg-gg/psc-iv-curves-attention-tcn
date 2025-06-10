"""
Training script for CVAE model using the unified ModelTrainer.
Demonstrates the new unified training interface.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# Import unified trainer and loss functions
from src.utils.iv_model_trainer import IVModelTrainer
from src.models.loss_functions import cvae_loss_function
from src.models.conditional_vae import CVAEModel
from src.utils.preprocess import preprocess_data_with_eos

def main():
    """Main training function for CVAE using unified trainer."""
    Va_np = np.concatenate((np.arange(0, 0.41, 0.1), np.arange(0.425, 1.401, 0.025)))
    num_voltage_points = len(Va_np)
    num_physical_qualities = 31

    # Hyperparameters
    latent_space_dim = 20
    adam_learning_rate = 2e-4
    training_batch_size = 64
    training_epochs = 50
    kl_beta_target = 0.5
    warmup_epochs = 5

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and prepare data
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

    # Use EOS-based preprocessing for variable-length sequences
    data = preprocess_data_with_eos(train_input_paths, train_output_paths)
    X_train_tensor, y_train_tensor, train_mask_tensor, train_length, eos_targets_train = data['train']
    X_test_tensor, y_test_tensor, test_mask_tensor, test_length, eos_targets_test = data['test']
    input_physical_scaler, output_scaler = data['scalers']
    y_test_original_scale = data['original_test_y']

    # Create DataLoader instances
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, train_mask_tensor, train_length, eos_targets_train)
    train_loader = DataLoader(train_dataset, batch_size=training_batch_size, shuffle=True, num_workers=2,
                            pin_memory=(device.type == 'cuda'))

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, test_mask_tensor, test_length, eos_targets_test)
    test_loader = DataLoader(test_dataset, batch_size=training_batch_size, shuffle=False, num_workers=2,
                           pin_memory=(device.type == 'cuda'))

    # Initialize model
    print("\n--- Initializing Model ---")
    cvae_model = CVAEModel(
        physical_dim=num_physical_qualities, 
        va_sweep_dim=num_voltage_points,
        latent_dim=latent_space_dim, 
        output_iv_dim=num_voltage_points
    ).to(device)
    
    optimizer = optim.Adam(cvae_model.parameters(), lr=adam_learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=training_epochs - warmup_epochs)
    
    print(cvae_model)
    total_trainable_params = sum(p.numel() for p in cvae_model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_trainable_params:,}")

    # Initialize unified trainer with CVAE loss function
    trainer = IVModelTrainer(
        model=cvae_model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        loss_function=cvae_loss_function,
        loss_params={
            'kl_beta': kl_beta_target,
            'eos_weight': 1.0,
            'physics_loss_weight': 0.5,
            'monotonicity_weight': 0.5,
            'smoothness_weight': 0.5
        }
    )

    # Training loop
    print("\n--- Starting Training ---")
    epoch_train_losses, epoch_test_losses = [], []
    
    # Smooth KL annealing parameters
    annealing_epochs = training_epochs // 3
    
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
            kl_beta_current = kl_beta_target / (1 + np.exp(-10 * (progress - 0.5)))
        else:
            kl_beta_current = kl_beta_target
            
        print(f'\nEpoch {epoch}, LR: {optimizer.param_groups[0]["lr"]:.6f}, KL Beta: {kl_beta_current:.4f}')
        
        # Train and validate using unified trainer
        avg_train_loss = trainer.train_one_epoch(
            train_loader, 
            kl_beta=kl_beta_current  # Pass current KL beta to override default
        )
        avg_test_loss = trainer.validate_one_epoch(
            test_loader, 
            kl_beta=kl_beta_current
        )
        
        epoch_train_losses.append(avg_train_loss)
        epoch_test_losses.append(avg_test_loss)
        
        print(f'====> Epoch: {epoch} Average train loss: {avg_train_loss:.4f}')
        print(f'====> Epoch: {epoch} Average test loss: {avg_test_loss:.4f}')
    
    print("--- Training Finished ---")

    # Save trained model
    model_save_path = 'checkpoints/cvae_model_unified.pth'
    params = {
        'physical_dim': num_physical_qualities,
        'va_sweep_dim': num_voltage_points,
        'latent_dim': latent_space_dim,
        'output_iv_dim': num_voltage_points
    }
    trainer.save_model(model_save_path, (input_physical_scaler, output_scaler), params)
    print(f"Model saved to {model_save_path}")

    # Evaluate using unified trainer
    print("\n--- Running Model Evaluation ---")
    mean_r2, sample_data = trainer.evaluate(
        test_loader=test_loader,
        scalers=(input_physical_scaler, output_scaler),
        include_plots=True
    )
    
    print(f"Final R² Score: {mean_r2:.4f}")

if __name__ == "__main__":
    main()