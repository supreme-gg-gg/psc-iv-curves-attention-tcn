"""
Lightweight training script for sequence models.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.utils.preprocess import preprocess_data_no_eos
from src.models.seq_model_trainer import SeqModelTrainer
from src.models.rnn_seq_model import SeqIVModel
from src.models.transformer_model import TransformerIVModel

# Data paths
INPUT_PATHS = [
    "dataset/Data_10k_sets/Data_10k_rng1/LHS_parameters_m.txt",
    "dataset/Data_10k_sets/Data_10k_rng2/LHS_parameters_m.txt",
    "dataset/Data_10k_sets/Data_10k_rng3/LHS_parameters_m.txt"
]
OUTPUT_PATHS = [
    "dataset/Data_10k_sets/Data_10k_rng1/iV_m.txt",
    "dataset/Data_10k_sets/Data_10k_rng2/iV_m.txt",
    "dataset/Data_10k_sets/Data_10k_rng3/iV_m.txt"
]

# Model Selection
MODEL_TYPE = "TRANSFORMER"  # or "RNN"

# Shared hyperparameters
PHYSICAL_DIM = 31 
BATCH_SIZE = 64  # Reduced from 128
EPOCHS = 50
LR = 1e-4
TEACHER_FORCING_RATIO = 0.5

# Transformer hyperparameters - lightweight version
D_MODEL = 32      # Reduced from 128
NHEAD = 4         # Reduced from 8 
NUM_DECODER_LAYERS = 2  # Keep small number of layers
DROPOUT_TRANSFORMER = 0.1
MAX_SEQ_LEN = 50

# RNN hyperparameters
HIDDEN_DIM = 64   # Reduced from 128
NUM_LAYERS = 2
DROPOUT_RNN = 0.2

# Save path
SAVE_PATH = "checkpoints/seq_iv_model.pth"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data
    data = preprocess_data_no_eos(INPUT_PATHS, OUTPUT_PATHS)
    X_train, padded_y_train, mask_train, lengths_train = data['train']
    X_test, padded_y_test, mask_test, lengths_test = data['test']
    scaled_zero_threshold = data['threshold']

    train_dataset = TensorDataset(X_train, padded_y_train, mask_train, torch.tensor(lengths_train))
    test_dataset = TensorDataset(X_test, padded_y_test, mask_test, torch.tensor(lengths_test))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    if MODEL_TYPE.upper() == "TRANSFORMER":
        model = TransformerIVModel(
            physical_dim=PHYSICAL_DIM,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_decoder_layers=NUM_DECODER_LAYERS,
            dropout=DROPOUT_TRANSFORMER,
            max_sequence_length=MAX_SEQ_LEN,
            scaled_zero_threshold=scaled_zero_threshold
        ).to(device)
    else:
        model = SeqIVModel(
            physical_dim=PHYSICAL_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT_RNN,
            max_sequence_length=MAX_SEQ_LEN,
            scaled_zero_threshold=scaled_zero_threshold
        ).to(device)

    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, min_lr=1e-5
    )
    trainer = SeqModelTrainer(model, optimizer, scheduler, device)

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        train_loss = trainer.train_one_epoch(train_loader)
        val_loss = trainer.validate_one_epoch(test_loader)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)

    # Save the model
    if MODEL_TYPE == "TRANSFORMER":
        params = {
            "physical_dim": PHYSICAL_DIM,
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_decoder_layers": NUM_DECODER_LAYERS,
            "dropout": DROPOUT_TRANSFORMER,
            "max_sequence_length": MAX_SEQ_LEN,
            "scaled_zero_threshold": scaled_zero_threshold
        }
    else:
        params = {
            "physical_dim": PHYSICAL_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT_RNN,
            "max_sequence_length": MAX_SEQ_LEN,
            "scaled_zero_threshold": scaled_zero_threshold
        }
    
    trainer.save_model(SAVE_PATH, data['scalers'], params)
     
    # Evaluate
    trainer.evaluate(test_loader, data['scalers'], include_plots=True)

if __name__ == "__main__":
    main()