#!/usr/bin/env python3
"""
train_seq_model_physics.py

Training script for sequence IV models (RNN or Transformer) using the physics-informed sequence loss.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.utils.preprocess import preprocess_data_with_eos
from src.models.rnn_seq_model import RNNIVModel
from src.utils.loss_functions import (
    physics_informed_sequence_loss,
)
from src.utils.iv_model_trainer import IVModelTrainer

# ========== User settings ==========
MODEL_TYPE = "RNN"  # 'RNN' or 'TRANSFORMER'
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-4
TEACHER_FORCING_RATIO = 1.0
WARMUP_EPOCHS = 5
MIN_TF_RATIO = 0.1

# EOS loss weight and physics loss weights
EOS_LOSS_WEIGHT = 2.0

LOSS_WEIGHTS = {
    "mse": 1.0,
    "mse_knee_factor": 2.0,
    "monotonicity": 0.01,
    "curvature": 0.01,
    "jsc": 0.2,
    "voc": 0.3,
}
KNEE_WINDOW = 5
GAMMA = 2.0

# RNN parameters
PHYSICAL_DIM = 31
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT_RNN = 0.2
MAX_SEQ_LEN = 50


# Data paths (example)
INPUT_PARAMS = [
    "dataset/Data_10k_sets/Data_10k_rng1/LHS_parameters_m.txt",
    "dataset/Data_10k_sets/Data_10k_rng2/LHS_parameters_m.txt",
    "dataset/Data_10k_sets/Data_10k_rng3/LHS_parameters_m.txt",
]
INPUT_IV = [
    "dataset/Data_10k_sets/Data_10k_rng1/iV_m.txt",
    "dataset/Data_10k_sets/Data_10k_rng2/iV_m.txt",
    "dataset/Data_10k_sets/Data_10k_rng3/iV_m.txt",
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load and preprocess with EOS
    data = preprocess_data_with_eos(INPUT_PARAMS, INPUT_IV)
    (X_train, y_train, mask_train, len_train, eos_train) = data["train"]
    (X_val, y_val, mask_val, len_val, eos_val) = data["test"]

    train_ds = TensorDataset(X_train, y_train, mask_train, len_train, eos_train)
    val_ds = TensorDataset(X_val, y_val, mask_val, len_val, eos_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = RNNIVModel(
        physical_dim=PHYSICAL_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT_RNN,
        max_sequence_length=MAX_SEQ_LEN,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Trainer setup
    trainer = IVModelTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        loss_function=physics_informed_sequence_loss,
        loss_params={
            "eos_loss_weight": EOS_LOSS_WEIGHT,
            "loss_weights": LOSS_WEIGHTS,
            "knee_window": KNEE_WINDOW,
            "gamma": GAMMA,
        },
    )

    # training loop with scheduled sampling
    for epoch in range(1, EPOCHS + 1):
        if epoch <= WARMUP_EPOCHS:
            tf_ratio = 1.0
        else:
            decay = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
            tf_ratio = max(MIN_TF_RATIO, 1.0 - decay * (1.0 - MIN_TF_RATIO))

        train_loss = trainer.train_one_epoch(
            train_loader, teacher_forcing_ratio=tf_ratio
        )
        val_loss = trainer.validate_one_epoch(val_loader)
        print(
            f"Epoch {epoch}/{EPOCHS} TF={tf_ratio:.2f}  Train Loss={train_loss:.4f}  Val Loss={val_loss:.4f}"
        )

    # Save via trainer
    trainer.save_model(
        "checkpoints/seq_physics.pth",
        data["scalers"],
        params={
            "model_type": MODEL_TYPE,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT_RNN,
        },
    )
    print("Model saved to checkpoints/seq_physics.pth")


if __name__ == "__main__":
    main()
