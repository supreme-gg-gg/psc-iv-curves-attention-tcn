import argparse
from pathlib import Path
import torch

from utils.config import (
    load_yaml_config, 
    get_device, 
    setup_model_directory,
    setup_logger,
    create_optimizer,
    create_scheduler
)
from data.base import IVCurveDataModule
from models.cvae import CVAE
from models.seq_iv_model import SeqIVModel
from trainers.cvae_trainer import CVAETrainer
from trainers.seq_trainer import SeqTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train IV curve models')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'mps'],
        help='Device to train on (default: auto-detect)'
    )
    return parser.parse_args()

def main():
    # Parse arguments and load config
    args = parse_args()
    config = load_yaml_config(args.config)
    
    # Setup logging and device
    logger = setup_logger(
        name=f"{config['model']['type']}_training",
        log_dir=config['training']['save_dir']
    )
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create model directory and save config
    save_dir = setup_model_directory(config)
    logger.info(f"Saving results to: {save_dir}")
    
    # Setup data
    data_config = config['data']
    datamodule = IVCurveDataModule(
        input_paths=data_config['input_paths'],
        output_paths=data_config['output_paths'],
        mode='fixed' if config['model']['type'] == 'cvae' else 'variable',
        test_size=data_config['test_size'],
        batch_size=data_config['batch_size']
    )
    data = datamodule.setup()
    logger.info("Data loading complete")
    
    # Create model
    model_config = config['model']
    if model_config['type'] == 'cvae':
        model = CVAE(
            physical_dim=model_config['physical_dim'],
            va_sweep_dim=model_config['va_sweep_dim'],
            latent_dim=model_config['latent_dim'],
            output_iv_dim=model_config['va_sweep_dim']
        )
    else:  # sequence model
        model = SeqIVModel(
            physical_dim=model_config['physical_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config.get('num_layers', 1),
            dropout=model_config.get('dropout', 0.2)
        )
    model = model.to(device)
    logger.info(f"Created {model_config['type']} model")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Create trainer
    if model_config['type'] == 'cvae':
        trainer = CVAETrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            config=config,
            scheduler=scheduler
        )
    else:
        trainer = SeqTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            config=config,
            scheduler=scheduler
        )
    
    # Train model
    logger.info("Starting training")
    train_loader, test_loader = data['train_loader'], data['test_loader']
    history = trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=config['training']['epochs']
    )
    
    # Save final results
    logger.info("Training complete")
    logger.info(f"Best test loss: {trainer.best_test_loss:.4f}")
    
    # Final evaluation
    results = model.evaluate(
        test_loader,
        scalers=data['scalers'],
        device=device
    )
    logger.info("Final evaluation metrics:")
    for metric, value in results.items():
        logger.info(f"{metric}: {value}")

if __name__ == "__main__":
    main()