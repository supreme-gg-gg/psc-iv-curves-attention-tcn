import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from typing import List, Tuple, Dict, Optional
from torch.nn.utils.rnn import pad_sequence

class IVCurveDataModule:
    """Base data module for IV curve datasets."""
    
    def __init__(self, 
                 input_paths: List[str],
                 output_paths: List[str],
                 mode: str = 'fixed',
                 test_size: float = 0.2,
                 batch_size: int = 32,
                 **kwargs):
        """
        Initialize the data module.
        
        Args:
            input_paths: List of paths to input parameter files
            output_paths: List of paths to IV curve files
            mode: 'fixed' for CVAE or 'variable' for sequence models
            test_size: Fraction of data to use for testing
            batch_size: Batch size for dataloaders
        """
        self.input_paths = input_paths
        self.output_paths = output_paths
        self.mode = mode
        self.test_size = test_size
        self.batch_size = batch_size
        self.scalers = None
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input and output path pairs."""
        if len(self.input_paths) != len(self.output_paths):
            raise ValueError("Number of input and output paths must match")
        for in_path, out_path in zip(self.input_paths, self.output_paths):
            if not all(path.endswith('.txt') for path in [in_path, out_path]):
                raise ValueError("All paths must point to .txt files")

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and stack data from multiple files."""
        input_data = [np.loadtxt(p, delimiter=',') for p in self.input_paths]
        output_data = [np.loadtxt(p, delimiter=',') for p in self.output_paths]
        return np.vstack(input_data), np.vstack(output_data)

    def _preprocess_fixed_length(self, 
                               X_data: np.ndarray, 
                               y_data: np.ndarray) -> Dict[str, torch.Tensor]:
        """Preprocess data for fixed-length models (e.g., CVAE)."""
        # Log transform inputs with small epsilon to handle zeros
        epsilon = 1e-40
        X_log = np.log10(X_data + epsilon)
        
        # Use RobustScaler for input features
        input_scaler = RobustScaler(quantile_range=(5, 95))
        X_scaled = input_scaler.fit_transform(X_log)
        
        # Use StandardScaler for outputs
        output_scaler = StandardScaler()
        y_scaled = output_scaler.fit_transform(y_data)
        
        self.scalers = (input_scaler, output_scaler)
        
        # Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
        
        return X_tensor, y_tensor

    def _preprocess_variable_length(self, 
                                  X_data: np.ndarray, 
                                  y_data: np.ndarray) -> Dict[str, torch.Tensor]:
        """Preprocess data for variable-length models (e.g., LSTM)."""
        # Process input features similarly to fixed length
        epsilon = 1e-40
        X_log = np.log10(X_data + epsilon)
        input_scaler = RobustScaler(quantile_range=(5, 95))
        X_scaled = input_scaler.fit_transform(X_log)
        
        # Filter IV curves and get lengths
        def filter_curve(curve):
            neg_indices = np.where(curve < 0)[0]
            return curve[:neg_indices[0]+1] if neg_indices.size > 0 else curve
            
        filtered_curves = [filter_curve(curve) for curve in y_data]
        lengths = [len(curve) for curve in filtered_curves]
        
        # Scale IV curves
        output_scaler = StandardScaler()
        all_values = np.concatenate(filtered_curves)
        output_scaler.fit(all_values.reshape(-1, 1))
        
        scaled_curves = [output_scaler.transform(curve.reshape(-1, 1)).flatten() 
                        for curve in filtered_curves]
        
        # Convert to tensors and pad
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensors = [torch.tensor(curve, dtype=torch.float32) for curve in scaled_curves]
        padded_y = pad_sequence(y_tensors, batch_first=True, padding_value=0.0)
        
        # Create masks for padding
        masks = torch.zeros_like(padded_y)
        for i, length in enumerate(lengths):
            masks[i, :length] = 1.0
            
        self.scalers = (input_scaler, output_scaler)
        
        return X_tensor, padded_y, masks, lengths

    def setup(self) -> Dict:
        """
        Set up the data module by loading and preprocessing data.
        
        Returns:
            Dictionary containing train/test splits and relevant metadata
        """
        # Load raw data
        X_data, y_data = self._load_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=self.test_size, random_state=42)
        
        # Process based on mode
        if self.mode == 'fixed':
            X_train_tensor, y_train_tensor = self._preprocess_fixed_length(X_train, y_train)
            X_test_tensor, y_test_tensor = self._preprocess_fixed_length(X_test, y_test)
            
            train_data = TensorDataset(X_train_tensor, y_train_tensor)
            test_data = TensorDataset(X_test_tensor, y_test_tensor)
            
        else:  # variable length
            X_train_tensor, y_train_tensor, train_masks, train_lengths = \
                self._preprocess_variable_length(X_train, y_train)
            X_test_tensor, y_test_tensor, test_masks, test_lengths = \
                self._preprocess_variable_length(X_test, y_test)
            
            train_data = TensorDataset(X_train_tensor, y_train_tensor, train_masks)
            test_data = TensorDataset(X_test_tensor, y_test_tensor, test_masks)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(
            test_data, batch_size=self.batch_size, shuffle=False)
        
        return {
            'train_loader': self.train_loader,
            'test_loader': self.test_loader,
            'scalers': self.scalers,
            'original_test_y': y_test  # Keep original test data for evaluation
        }

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Get train and test dataloaders."""
        if not hasattr(self, 'train_loader') or not hasattr(self, 'test_loader'):
            raise RuntimeError("Call setup() before accessing dataloaders")
        return self.train_loader, self.test_loader

    def get_scalers(self) -> Tuple[RobustScaler, StandardScaler]:
        """Get the fitted scalers."""
        if self.scalers is None:
            raise RuntimeError("Call setup() before accessing scalers")
        return self.scalers