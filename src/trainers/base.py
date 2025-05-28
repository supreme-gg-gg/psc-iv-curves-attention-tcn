import torch
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from torch.optim.lr_scheduler import _LRScheduler

class BaseTrainer(ABC):
    """Base trainer class for IV curve models."""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 config: Dict[str, Any],
                 scheduler: Optional[_LRScheduler] = None):
        """
        Initialize trainer.
        
        Args:
            model: The model to train
            optimizer: Optimizer instance
            device: Device to train on
            config: Training configuration
            scheduler: Optional learning rate scheduler
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.scheduler = scheduler
        
        # Training history
        self.train_losses: List[float] = []
        self.test_losses: List[float] = []
        self.learning_rates: List[float] = []
        self.best_test_loss: float = float('inf')
        
        # Setup save directory
        self.save_dir = Path(config.get('save_dir', './checkpoints'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def compute_loss(self, batch: tuple, **kwargs) -> torch.Tensor:
        """
        Compute loss for a batch of data.
        
        Args:
            batch: Tuple of tensors from dataloader
            **kwargs: Additional arguments for loss computation
            
        Returns:
            Loss tensor
        """
        pass

    def train_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            loss = self.compute_loss(batch)
            loss.backward()
            
            # Optional gradient clipping
            if 'max_grad_norm' in self.config:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['max_grad_norm']
                )
            
            self.optimizer.step()
            total_loss += loss.item()
            
            # Progress logging
            if batch_idx % self.config.get('log_interval', 10) == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(batch[0])}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        return avg_loss

    def test_epoch(self, test_loader: torch.utils.data.DataLoader) -> float:
        """
        Evaluate on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Average test loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = len(test_loader)
        
        with torch.no_grad():
            for batch in test_loader:
                loss = self.compute_loss(batch)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.test_losses.append(avg_loss)
        
        # Save best model
        if avg_loss < self.best_test_loss:
            self.best_test_loss = avg_loss
            self.save_checkpoint('best_model.pth')
        
        return avg_loss

    def train(self, train_loader: torch.utils.data.DataLoader, 
              test_loader: torch.utils.data.DataLoader,
              num_epochs: int) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            num_epochs: Number of epochs to train
            
        Returns:
            Dictionary containing training history
        """
        for epoch in range(1, num_epochs + 1):
            print(f'\nEpoch {epoch}/{num_epochs}')
            
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            print(f'Average train loss: {train_loss:.4f}')
            
            # Testing
            test_loss = self.test_epoch(test_loader)
            print(f'Average test loss: {test_loss:.4f}')
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Save checkpoint
            if epoch % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        
        # Plot training curves
        self.plot_training_curves()
        
        return {
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'learning_rates': self.learning_rates
        }

    def save_checkpoint(self, filename: str) -> None:
        """
        Save a checkpoint of the current training state.
        
        Args:
            filename: Name of the checkpoint file
        """
        checkpoint = {
            'epoch': len(self.train_losses),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'learning_rates': self.learning_rates,
            'best_test_loss': self.best_test_loss,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        print(f'Checkpoint saved to {save_path}')

    def plot_training_curves(self) -> None:
        """Plot and save training curves."""
        plt.figure(figsize=(12, 4))
        
        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()
        plt.grid(True)
        
        # Learning rate curve
        plt.subplot(1, 2, 2)
        plt.plot(self.learning_rates)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png')
        plt.close()

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a checkpoint to resume training.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.train_losses = checkpoint['train_losses']
        self.test_losses = checkpoint['test_losses']
        self.learning_rates = checkpoint['learning_rates']
        self.best_test_loss = checkpoint['best_test_loss']
        
        print(f'Loaded checkpoint from epoch {len(self.train_losses)}')