"""
Lightweight training script for sequence models.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.utils.preprocess import preprocess_data_with_eos
from src.utils.iv_model_trainer import IVModelTrainer
from src.models.loss_functions import sequence_loss_with_eos
from src.models.rnn_seq_model import RNNIVModel
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
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-4
TEACHER_FORCING_RATIO = 1.0
MAX_SEQ_LEN = 50
MIN_TF_RATIO = 0.1
EOS_LOSS_WEIGHT = 2.0  # increase emphasis on EOS loss
WARMUP_EPOCHS = 5  # epochs at full teacher forcing

# Transformer hyperparameters - lightweight version
D_MODEL = 32
NHEAD = 2
NUM_DECODER_LAYERS = 3
DROPOUT_TRANSFORMER = 0.1
MAX_SEQ_LEN = 50

# RNN hyperparameters
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT_RNN = 0.2

# Save path
SAVE_PATH = "checkpoints/seq_iv_model.pth"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data with EOS targets
    data = preprocess_data_with_eos(INPUT_PATHS, OUTPUT_PATHS)
    X_train, padded_y_train, mask_train, lengths_train, eos_train = data['train']
    X_test, padded_y_test, mask_test, lengths_test, eos_test = data['test']

    # include EOS targets for training
    train_dataset = TensorDataset(X_train, padded_y_train, mask_train, lengths_train, eos_train)
    # test does not require EOS targets
    test_dataset = TensorDataset(X_test, padded_y_test, mask_test, lengths_test, eos_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    if MODEL_TYPE.upper() == "TRANSFORMER":
        model = TransformerIVModel(
            physical_dim=PHYSICAL_DIM,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_decoder_layers=NUM_DECODER_LAYERS,
            dropout=DROPOUT_TRANSFORMER,
            max_sequence_length=MAX_SEQ_LEN,
            decoder_mask_ratio=0.15  # 15% denoising mask
        ).to(device)
    else:
        model = RNNIVModel(
            physical_dim=PHYSICAL_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT_RNN,
            max_sequence_length=MAX_SEQ_LEN
        ).to(device)

    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, min_lr=1e-5
    )
    # Trainer with unified interface and sequence loss function
    trainer = IVModelTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        loss_function=sequence_loss_with_eos,
        loss_params={
            'eos_loss_weight': EOS_LOSS_WEIGHT
        }
    )

    # Training loop with scheduled sampling and efficiency enhancements
    for epoch in range(1, EPOCHS + 1):
        # Curriculum: full TF for warmup, then linear decay
        if epoch <= WARMUP_EPOCHS:
            current_tf_ratio = 1.0
        else:
            decay_epochs = EPOCHS - WARMUP_EPOCHS
            decayed = (epoch - WARMUP_EPOCHS) / decay_epochs
            current_tf_ratio = max(MIN_TF_RATIO, 1.0 - decayed * (1.0 - MIN_TF_RATIO))

        train_loss = trainer.train_one_epoch(train_loader, teacher_forcing_ratio=current_tf_ratio)
        val_loss = trainer.validate_one_epoch(test_loader)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, TF Ratio = {current_tf_ratio:.4f}")

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
        }
    else:
        params = {
            "physical_dim": PHYSICAL_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT_RNN,
        }
    
    trainer.save_model(SAVE_PATH, data['scalers'], params)
     
    # Evaluate
    mean_r2, samples = trainer.evaluate(test_loader, data['scalers'], include_plots=True)

if __name__ == "__main__":
    main()