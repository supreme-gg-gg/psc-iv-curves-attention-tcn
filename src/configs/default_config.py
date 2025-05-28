from dataclasses import dataclass
from typing import List, Optional, Union
from pathlib import Path

@dataclass
class DataConfig:
    """Data configuration."""
    input_paths: List[str] = None
    output_paths: List[str] = None
    test_size: float = 0.2
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 50
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0
    save_dir: str = './checkpoints'
    save_interval: int = 10
    log_interval: int = 10
    early_stopping_patience: int = 10
    scheduler_config: Optional[dict] = None

@dataclass
class ModelConfig:
    """Base model configuration."""
    physical_dim: int = 31
    model_type: str = None  # 'cvae' or 'sequence'
    device: str = 'cuda'  # 'cuda', 'cpu', or 'mps'

@dataclass
class CVAEConfig(ModelConfig):
    """CVAE-specific configuration."""
    va_sweep_dim: int = 41
    latent_dim: int = 20
    kld_weight: float = 1.0
    physics_weight: float = 0.1
    monotonicity_weight: float = 0.7
    smoothness_weight: float = 0.3
    model_type: str = 'cvae'

@dataclass
class SequenceConfig(ModelConfig):
    """Sequence model-specific configuration."""
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    teacher_forcing_ratio: float = 0.5
    model_type: str = 'sequence'

# Default configurations
default_cvae_config = {
    'data': {
        'input_paths': [
            "dataset/Data_10k_sets/Data_10k_rng1/LHS_parameters_m.txt",
            "dataset/Data_10k_sets/Data_10k_rng2/LHS_parameters_m.txt",
            "dataset/Data_10k_sets/Data_10k_rng3/LHS_parameters_m.txt"
        ],
        'output_paths': [
            "dataset/Data_10k_sets/Data_10k_rng1/iV_m.txt",
            "dataset/Data_10k_sets/Data_10k_rng2/iV_m.txt",
            "dataset/Data_10k_sets/Data_10k_rng3/iV_m.txt"
        ],
        'test_size': 0.2,
        'batch_size': 64,
        'num_workers': 4,
        'pin_memory': True
    },
    'training': {
        'epochs': 50,
        'learning_rate': 2e-4,
        'max_grad_norm': 1.0,
        'save_dir': './checkpoints/cvae',
        'scheduler_config': {
            'name': 'cosine',
            'T_max': 50,
            'eta_min': 1e-6
        }
    },
    'model': {
        'physical_dim': 31,
        'va_sweep_dim': 41,
        'latent_dim': 20,
        'kld_weight': 1.0,
        'physics_weight': 0.1,
        'monotonicity_weight': 0.7,
        'smoothness_weight': 0.3,
        'model_type': 'cvae'
    }
}

default_sequence_config = {
    'data': {
        'input_paths': [
            "dataset/Data_10k_sets/Data_10k_rng1/LHS_parameters_m.txt",
            "dataset/Data_10k_sets/Data_10k_rng2/LHS_parameters_m.txt",
            "dataset/Data_10k_sets/Data_10k_rng3/LHS_parameters_m.txt"
        ],
        'output_paths': [
            "dataset/Data_10k_sets/Data_10k_rng1/iV_m.txt",
            "dataset/Data_10k_sets/Data_10k_rng2/iV_m.txt",
            "dataset/Data_10k_sets/Data_10k_rng3/iV_m.txt"
        ],
        'test_size': 0.2,
        'batch_size': 32,
        'num_workers': 4,
        'pin_memory': True
    },
    'training': {
        'epochs': 20,
        'learning_rate': 1e-3,
        'max_grad_norm': 1.0,
        'save_dir': './checkpoints/sequence'
    },
    'model': {
        'physical_dim': 31,
        'hidden_dim': 64,
        'num_layers': 1,
        'dropout': 0.2,
        'teacher_forcing_ratio': 0.5,
        'model_type': 'sequence'
    }
}

def load_config(model_type: str) -> dict:
    """Load default configuration for a specific model type."""
    if model_type == 'cvae':
        return default_cvae_config
    elif model_type == 'sequence':
        return default_sequence_config
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def update_config(base_config: dict, updates: dict) -> dict:
    """Update configuration with new values."""
    import copy
    config = copy.deepcopy(base_config)
    
    for section, values in updates.items():
        if section in config:
            config[section].update(values)
        else:
            config[section] = values
            
    return config