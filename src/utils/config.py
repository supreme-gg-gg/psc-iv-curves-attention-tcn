import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Union, Optional
import logging

def setup_logger(name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """Set up logger for the application."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_dir is provided
    if log_dir:
        log_path = Path(log_dir) / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_device(device_name: Optional[str] = None) -> torch.device:
    """
    Get the appropriate torch device.
    
    Args:
        device_name: Optional device specification ('cuda', 'cpu', or 'mps')
        
    Returns:
        torch.device: The device to use
    """
    if device_name is None:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    return torch.device(device_name)

def setup_model_directory(config: Dict[str, Any]) -> Path:
    """
    Create and return model directory based on configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Path: Path to model directory
    """
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = save_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return save_dir

def validate_config(config: Dict[str, Any], required_keys: Dict[str, type]) -> None:
    """
    Validate configuration dictionary against required keys and types.
    
    Args:
        config: Configuration dictionary to validate
        required_keys: Dictionary mapping required keys to their expected types
        
    Raises:
        ValueError: If validation fails
    """
    for key, expected_type in required_keys.items():
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
        
        value = config[key]
        if not isinstance(value, expected_type):
            raise ValueError(
                f"Invalid type for {key}. Expected {expected_type}, got {type(value)}"
            )

def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Dict: Merged configuration
    """
    merged = base.copy()
    
    for key, value in override.items():
        if (
            key in merged and 
            isinstance(merged[key], dict) and 
            isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
            
    return merged

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass

def create_optimizer(
    model: torch.nn.Module,
    config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: The model whose parameters will be optimized
        config: Configuration dictionary containing optimizer settings
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    optimizer_config = config['training'].get('optimizer', {})
    optimizer_name = optimizer_config.get('name', 'adam').lower()
    lr = config['training']['learning_rate']
    
    if optimizer_name == 'adam':
        beta1 = optimizer_config.get('beta1', 0.9)
        beta2 = optimizer_config.get('beta2', 0.999)
        eps = optimizer_config.get('eps', 1e-8)
        weight_decay = optimizer_config.get('weight_decay', 0)
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd':
        momentum = optimizer_config.get('momentum', 0.9)
        weight_decay = optimizer_config.get('weight_decay', 0)
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ConfigurationError(f"Unsupported optimizer: {optimizer_name}")

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any]
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: The optimizer
        config: Configuration dictionary containing scheduler settings
        
    Returns:
        Optional[torch.optim.lr_scheduler._LRScheduler]: Configured scheduler
    """
    scheduler_config = config['training'].get('scheduler_config')
    if not scheduler_config:
        return None
        
    scheduler_name = scheduler_config['name'].lower()
    
    if scheduler_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config['T_max'],
            eta_min=scheduler_config.get('eta_min', 0)
        )
    elif scheduler_name == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config['step_size'],
            gamma=scheduler_config['gamma']
        )
    elif scheduler_name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.1),
            patience=scheduler_config.get('patience', 10)
        )
    else:
        raise ConfigurationError(f"Unsupported scheduler: {scheduler_name}")