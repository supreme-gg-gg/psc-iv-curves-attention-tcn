#!/usr/bin/env python3
"""
Based on Memo's basic_models.py in tensorflow. Refactored and ported to PyTorch.

Custom implementation of temporal causal dilated convolution ref: https://github.com/pytorch/pytorch/issues/1333
This is not supported natively in PyTorch Conv1d, but if you use Keras you can just set padding='causal'
A very similar alternative approach is on PyTorch Forum: https://discuss.pytorch.org/t/causal-convolution/3456/4
However, we here use the official original implementation from the TCN paper authors (Bai et. al.)
from this open-source repo: https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
"""

import numpy as np
import torch
from torch import nn
from utils.config import INPUT_FILE_PARAMS, INPUT_FILE_IV, RANDOM_SEED, TCNConfig
from utils.common import FourierFeatures, print_model_info, Chomp1d
from utils.preprocessor import DataPreprocessor
from utils.trainer import ModelTrainer


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
    ):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout1d(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        if self.downsample is not None:
            nn.init.xavier_uniform_(self.downsample.weight)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PhysicsInformedTCN(nn.Module):
    def __init__(self, num_params: int, config: TCNConfig):
        super().__init__()
        DENSE_UNITS_PARAMS = config.DENSE_UNITS_PARAMS
        DROPOUT_RATE = config.DROPOUT_RATE
        TCN_FILTERS = config.FILTERS
        TCN_KERNEL_SIZE = config.KERNEL_SIZE
        FOURIER_NUM_BANDS = config.FOURIER_NUM_BANDS
        # N_LAYERS = len(TCN_FILTERS)

        self.param_mlp = nn.Sequential(
            nn.Linear(num_params, DENSE_UNITS_PARAMS[0]),
            nn.BatchNorm1d(DENSE_UNITS_PARAMS[0]),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(DENSE_UNITS_PARAMS[0], DENSE_UNITS_PARAMS[1]),
            nn.BatchNorm1d(DENSE_UNITS_PARAMS[1]),
            nn.ReLU(),
            nn.Linear(DENSE_UNITS_PARAMS[1], DENSE_UNITS_PARAMS[2]),
            nn.BatchNorm1d(DENSE_UNITS_PARAMS[2]),
            nn.ReLU(),
        )
        self.fourier_features = FourierFeatures(FOURIER_NUM_BANDS)
        tcn_in_dim = DENSE_UNITS_PARAMS[2] + (2 * FOURIER_NUM_BANDS)
        # True causal TCN stack
        self.tcn = TemporalConvNet(
            num_inputs=tcn_in_dim,
            num_channels=TCN_FILTERS,
            kernel_size=TCN_KERNEL_SIZE,
            dropout=DROPOUT_RATE,
        )
        self.final_projection = nn.Conv1d(TCN_FILTERS[-1], 1, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x_params: torch.Tensor, v_grid: torch.Tensor) -> torch.Tensor:
        param_embedding = self.param_mlp(x_params)
        voltage_embedding = self.fourier_features(v_grid)
        param_tiled = param_embedding.unsqueeze(1).expand(-1, v_grid.shape[1], -1)
        combined_features = torch.cat([param_tiled, voltage_embedding], dim=-1)
        tcn_input = combined_features.permute(0, 2, 1)
        tcn_output = self.tcn(tcn_input)
        return self.final_projection(tcn_output).squeeze(1)


if __name__ == "__main__":
    config = TCNConfig()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    preprocessor = DataPreprocessor(INPUT_FILE_PARAMS, INPUT_FILE_IV)
    preprocessor.load_and_prepare(truncation_threshold_pct=0.01)
    if preprocessor.X_clean is None:
        print("Preprocessor did not produce X_clean. Aborting.")
        raise ValueError("Preprocessor did not produce X_clean. Aborting.")

    dataloaders = preprocessor.get_dataloaders(config.TRAIN_CONFIG.BATCH_SIZE)

    model = PhysicsInformedTCN(num_params=preprocessor.X_clean.shape[1], config=config)
    print_model_info(model)

    trainer = ModelTrainer(model, config)

    trainer.train(dataloaders, epochs=config.TRAIN_CONFIG.EPOCHS)

    trainer.evaluate(dataloaders["test"])
    trainer.plot_results(dataloaders["test"])

    print("Experiment finished successfully.")
