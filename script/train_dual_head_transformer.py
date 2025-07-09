#!/usr/bin/env python3
"""
Training script for dual-head transformer IV curve model.
This script demonstrates how to use the dual output approach.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from src.utils.preprocess import preprocess_fixed_length_dual_output
from src.models.duo_transformer import TransformerDualOutputIVModel
from src.utils.iv_model_trainer import DualOutputIVModelTrainer


def main():
    # Configuration
    config = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "batch_size": 32,
        "learning_rate": 1e-4,
        "num_epochs": 100,
        "patience": 15,
        # Model parameters
        "physical_dim": 15,  # Adjust based on your input features
        "d_model": 128,
        "nhead": 8,
        "num_decoder_layers": 4,
        "max_sequence_length": 10,  # num_pre + num_post - 1 from preprocessing
        "dropout": 0.1,
        # Loss weights
        "current_weight": 0.5,
        "voltage_weight": 0.3,
        "monotonicity_weight": 0.1,
        "smoothness_weight": 0.05,
        "endpoint_weight": 0.05,
        "use_enhanced_loss": True,  # Set to False for simpler loss
        # Data paths
        "input_paths": [
            "dataset/Data_100k/LHS_parameters_m.txt",
        ],
        "output_paths": [
            "dataset/Data_100k/iV_m.txt",
        ],
    }

    print(f"Using device: {config['device']}")

    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_dict = preprocess_fixed_length_dual_output(
        input_paths=config["input_paths"],
        output_paths=config["output_paths"],
        num_pre=5,
        num_post=6,  # This gives max_sequence_length = 5 + 6 - 1 = 10
        current_thresh=1.5,  # Filter outliers
        round_digits=6,
        test_size=0.2,
    )

    # Extract data
    X_train, Y_current_train, V_voltage_train, isc_train = data_dict["train"]
    X_test, Y_current_test, V_voltage_test, isc_test = data_dict["test"]
    scalers = data_dict["scalers"]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Sequence length: {Y_current_train.shape[1]}")
    print(f"Input features: {X_train.shape[1]}")

    # Update config with actual dimensions
    config["physical_dim"] = X_train.shape[1]
    config["max_sequence_length"] = Y_current_train.shape[1]

    # Create datasets
    train_dataset = TensorDataset(X_train, Y_current_train, V_voltage_train)
    test_dataset = TensorDataset(X_test, Y_current_test, V_voltage_test)

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # Initialize model
    print("Initializing model...")
    model = TransformerDualOutputIVModel(
        physical_dim=config["physical_dim"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_decoder_layers=config["num_decoder_layers"],
        max_sequence_length=config["max_sequence_length"],
        dropout=config["dropout"],
    ).to(config["device"])

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=5, min_lr=1e-6
    )

    # Initialize trainer
    trainer = DualOutputIVModelTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config["device"],
        current_loss_weight=config["current_weight"],
        voltage_loss_weight=config["voltage_weight"],
        monotonicity_loss_weight=config["monotonicity_weight"],
        smoothness_loss_weight=config["smoothness_weight"],
        endpoint_loss_weight=config["endpoint_weight"],
        use_enhanced_loss=config["use_enhanced_loss"],
    )

    # Training loop
    print("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config["num_epochs"]):
        # Train
        train_losses = trainer.train_one_epoch(train_loader)

        # Validate
        val_loss = trainer.validate_one_epoch(test_loader)

        # Update scheduler
        scheduler.step(val_loss)

        # Print progress
        if epoch % 10 == 0 or epoch < 10:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_losses['total']:.6f} "
                f"(C: {train_losses['current']:.4f}, "
                f"V: {train_losses['voltage']:.4f}, "
                f"P: {train_losses['physics']:.4f}) | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": config,
                    "scalers": scalers,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                "checkpoints/best_dual_head_transformer.pth",
            )
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final evaluation
    print("\nFinal evaluation...")
    model.load_state_dict(
        torch.load("checkpoints/best_dual_head_transformer.pth")["model_state_dict"]
    )

    mean_r2, sample_data = trainer.evaluate(test_loader, scalers)
    print(f"Final R² Score: {mean_r2:.4f}")

    print("Training completed!")


if __name__ == "__main__":
    main()
