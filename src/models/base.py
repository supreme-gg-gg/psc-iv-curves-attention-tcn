import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import r2_score

class BaseIVModel(nn.Module, ABC):
    """Base class for IV curve models."""
    
    def __init__(self, physical_dim: int, **kwargs):
        """
        Initialize the base model.
        
        Args:
            physical_dim: Dimension of physical parameter input
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        self.physical_dim = physical_dim
        self.model_type = self.__class__.__name__

    @abstractmethod
    def forward(self, physical_params: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            physical_params: Physical parameters tensor
            **kwargs: Additional arguments specific to each model type
            
        Returns:
            Model output (specific format depends on model type)
        """
        pass

    def save_checkpoint(self, path: str, scalers: Optional[Tuple] = None, **metadata) -> None:
        """
        Save model checkpoint with metadata.
        
        Args:
            path: Path to save the checkpoint
            scalers: Optional tuple of (input_scaler, output_scaler)
            **metadata: Additional metadata to save
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_type': self.model_type,
            'model_config': self._get_config(),
        }
        
        if scalers is not None:
            save_dict['scalers'] = scalers
            
        save_dict.update(metadata)
        torch.save(save_dict, path)
        
    @abstractmethod
    def _get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for saving/loading.
        
        Returns:
            Dictionary containing model configuration
        """
        return {
            'physical_dim': self.physical_dim
        }

    @classmethod
    def load_checkpoint(cls, path: str, device: torch.device) -> Tuple['BaseIVModel', Dict[str, Any]]:
        """
        Load model from checkpoint.
        
        Args:
            path: Path to the checkpoint file
            device: Device to load the model onto
            
        Returns:
            Tuple of (loaded_model, checkpoint_data)
        """
        data = torch.load(path, map_location=device)
        model = cls(**data['model_config'])
        model.load_state_dict(data['model_state_dict'])
        model.to(device)
        return model, data

    def evaluate(self, dataloader: torch.utils.data.DataLoader, 
                scalers: Tuple, device: torch.device) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            dataloader: DataLoader containing test data
            scalers: Tuple of (input_scaler, output_scaler)
            device: Device to run evaluation on
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.eval()
        r2_scores = []
        total_loss = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle different batch formats
                if len(batch) == 2:  # Fixed length (CVAE)
                    physical_params, true_curves = batch
                    mask = None
                else:  # Variable length with mask
                    physical_params, true_curves, mask = batch
                
                # Move to device
                physical_params = physical_params.to(device)
                true_curves = true_curves.to(device)
                if mask is not None:
                    mask = mask.to(device)
                
                # Model forward pass
                outputs = self(physical_params, target_seq=true_curves if hasattr(self, 'training') else None)
                
                # Handle different output formats
                if isinstance(outputs, tuple):  # CVAE returns multiple values
                    predicted_curves = outputs[0]
                else:
                    predicted_curves = outputs
                
                # Compute metrics
                _, output_scaler = scalers
                for i in range(len(physical_params)):
                    true_curve = true_curves[i]
                    pred_curve = predicted_curves[i]
                    
                    if mask is not None:
                        # For variable length, only evaluate on valid positions
                        valid_mask = mask[i].bool()
                        true_curve = true_curve[valid_mask]
                        pred_curve = pred_curve[valid_mask]
                    
                    # Convert to original scale
                    true_unscaled = output_scaler.inverse_transform(true_curve.cpu().numpy().reshape(-1, 1)).flatten()
                    pred_unscaled = output_scaler.inverse_transform(pred_curve.cpu().numpy().reshape(-1, 1)).flatten()
                    
                    # Compute R² score
                    r2 = r2_score(true_unscaled, pred_unscaled)
                    r2_scores.append(r2)
                
                # Accumulate samples
                num_samples += len(physical_params)
        
        # Compute aggregate metrics
        results = {
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'r2_median': np.median(r2_scores),
            'num_samples': num_samples
        }
        
        return results