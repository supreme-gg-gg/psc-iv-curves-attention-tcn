# IV Curve Modeling

This repository contains implementations of deep learning models for modeling and generating photovoltaic (PV) IV curves. The codebase has been refactored to provide a clean, modular structure that supports multiple model architectures.

## Features

- Multiple model architectures:
  - Conditional Variational Autoencoder (CVAE) with physics-informed loss
  - Sequence-based model using LSTM with teacher forcing
- Modular design with base classes for easy extension
- Configuration-driven training
- Standardized data preprocessing
- Comprehensive evaluation metrics
- Training visualization and logging

## Project Structure

```
.
├── src/
│   ├── data/            # Data loading and preprocessing
│   ├── models/          # Model architectures
│   ├── trainers/        # Training logic
│   ├── configs/         # Configuration files
│   └── utils/           # Utility functions
├── notebooks/           # Analysis notebooks
├── dataset/            # Data files
├── checkpoints/        # Saved models
└── assets/            # Generated figures
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/psc-iv-curves.git
cd psc-iv-curves

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training a Model

1. Choose or modify a configuration file in `src/configs/`:
   - `cvae.yaml` for the CVAE model
   - `sequence.yaml` for the sequence model

2. Run training:
```bash
python src/train.py --config src/configs/cvae.yaml
# or
python src/train.py --config src/configs/sequence.yaml
```

Optional arguments:
- `--device`: Specify device ('cuda', 'cpu', or 'mps')

### Configuration

The YAML configuration files contain sections for:
- Data loading and preprocessing
- Model architecture and parameters
- Training hyperparameters
- Logging and checkpointing

Example (CVAE):
```yaml
data:
  input_paths:
    - dataset/Data_10k_sets/Data_10k_rng1/LHS_parameters_m.txt
  output_paths:
    - dataset/Data_10k_sets/Data_10k_rng1/iV_m.txt
  test_size: 0.2
  batch_size: 64

model:
  type: cvae
  physical_dim: 31
  latent_dim: 20
  kld_weight: 1.0
  physics_weight: 0.1

training:
  epochs: 50
  learning_rate: 2.0e-4
  max_grad_norm: 1.0
```

## Models

### CVAE
- Conditional Variational Autoencoder for IV curve generation
- Physics-informed loss combining:
  - Reconstruction loss
  - KL divergence
  - Monotonicity constraint
  - Smoothness constraint
- KL annealing during training

### Sequence Model
- LSTM-based autoregressive model
- Teacher forcing with scheduled decay
- Handles variable-length sequences
- Masked loss computation

## Development

To add a new model:

1. Create model class in `src/models/` inheriting from `BaseIVModel`
2. Create trainer class in `src/trainers/` inheriting from `BaseTrainer`
3. Add configuration file in `src/configs/`
4. Update `src/train.py` to handle the new model type

Example:
```python
class NewModel(BaseIVModel):
    def __init__(self, physical_dim, **kwargs):
        super().__init__(physical_dim)
        # Model-specific initialization
        
    def forward(self, physical_params, **kwargs):
        # Model-specific forward pass
        pass
```

## Evaluation

Models are evaluated using:
- R² scores for prediction accuracy
- Physics-based metrics for curve validity
- Visual comparison of generated curves

Results are saved in:
- Training curves: `checkpoints/<model_type>/training_curves.png`
- Model checkpoints: `checkpoints/<model_type>/`
- Logs: `checkpoints/<model_type>/<model_type>_training.log`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
