#!/usr/bin/env python3
"""
Combining causal dilated neighbour attention with a TCN.
Adapted from https://github.com/alexmehta/NAC-TCN-TCNs-with-Causal-NA/tree/main

You need to first install natten: pip install natten
"""

import numpy as np
import torch
from torch import nn
from utils.preprocessor import DataPreprocessor
from utils.trainer import ModelTrainer
from utils.config import DINAConfig, INPUT_FILE_PARAMS, INPUT_FILE_IV, RANDOM_SEED
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from utils.common import FourierFeatures, print_model_info, Chomp1d

from natten import NeighborhoodAttention1D


class TemporalBlock(nn.Module):
    """A direct, line-by-line translation of the provided TemporalBlock."""

    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
        local=14,
    ):
        super(TemporalBlock, self).__init__()
        self.k = kernel_size
        self.dilation = dilation

        # Net 1: Causal Convolution
        self.conv1 = weight_norm(
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
        self.net_1 = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)

        # Net 2: Neighborhood Attention Layer
        # Using 4 heads for better capacity instead of the original 1
        self.conv2 = NeighborhoodAttention1D(
            n_outputs, kernel_size=kernel_size, dilation=dilation, num_heads=1
        )
        self.net_2 = nn.Sequential(self.conv2)

        # Net 3: Post-processing for the attention layer
        # This chomp is necessary to remove the manual padding added before attention
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout1d(dropout)
        self.net_3 = nn.Sequential(self.chomp2, self.relu2, self.dropout2)

        # Residual Connection
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # Apply the first causal convolution network
        out = self.net_1(x)

        # The author's trick to make centered attention causal:
        # 1. Manually pad the sequence on the left.
        # The amount of padding is `(kernel_size-1)*dilation`, which is equal to the `padding` parameter.
        pad_amount = (self.k - 1) * self.dilation
        out = F.pad(out, (pad_amount, 0))

        # 2. Permute for attention layer, apply attention, permute back.
        out = out.permute(0, 2, 1)
        out = self.net_2(out)
        out = out.permute(0, 2, 1)

        # 3. Apply post-processing, which includes chomping the manual padding.
        out = self.net_3(out)

        # Apply residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """A direct, line-by-line translation of the provided TemporalConvNet."""

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
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


class PhysicsInformedNATCN(nn.Module):
    """The main model that integrates the faithfully translated modules."""

    def __init__(self, num_params: int, config: DINAConfig):
        super().__init__()

        DENSE_UNITS_PARAMS = config.DENSE_UNITS_PARAMS
        DROPOUT_RATE = config.DROPOUT_RATE
        TCN_N_LAYERS = config.N_LAYERS
        TCN_KERNEL_SIZE = config.KERNEL_SIZE
        FOURIER_NUM_BANDS = config.FOURIER_NUM_BANDS

        # 1. Input Encoder
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
        self.fourier_features = FourierFeatures(
            FOURIER_NUM_BANDS, mode=config.FOURIER_MODE
        )

        # 2. Main TCN Stack using the translated modules
        tcn_input_channels = DENSE_UNITS_PARAMS[2] + (2 * FOURIER_NUM_BANDS)
        tcn_hidden_channels = [tcn_input_channels] * TCN_N_LAYERS

        self.tcn = TemporalConvNet(
            num_inputs=tcn_input_channels,
            num_channels=tcn_hidden_channels,
            kernel_size=TCN_KERNEL_SIZE,
            dropout=DROPOUT_RATE,
        )

        # 3. Output Decoder
        self.decoder = nn.Conv1d(tcn_hidden_channels[-1], 1, 1)

    def forward(self, x_params, v_grid):
        # 1. Encode inputs
        param_embedding = self.param_mlp(x_params)
        voltage_embedding = self.fourier_features(v_grid)
        param_tiled = param_embedding.unsqueeze(1).expand(-1, v_grid.shape[1], -1)
        combined_features = torch.cat([param_tiled, voltage_embedding], dim=-1)

        tcn_input = combined_features.permute(0, 2, 1)

        # 2. Process with TCN
        tcn_output = self.tcn(tcn_input)

        # 3. Decode to final prediction
        predictions = self.decoder(tcn_output)

        return predictions.squeeze(1)


if __name__ == "__main__":
    if not (INPUT_FILE_PARAMS.exists() and INPUT_FILE_IV.exists()):
        print("Input data files not found!")
    else:
        config = DINAConfig()
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        preprocessor = DataPreprocessor(INPUT_FILE_PARAMS, INPUT_FILE_IV)
        preprocessor.load_and_prepare(truncation_threshold_pct=0.01)
        dataloaders = preprocessor.get_dataloaders(config.TRAIN_CONFIG.BATCH_SIZE)

        if preprocessor.X_clean is None:
            print("Preprocessor did not produce X_clean. Aborting.")
            raise ValueError("Preprocessor did not produce X_clean. Aborting.")

        model = PhysicsInformedNATCN(
            num_params=preprocessor.X_clean.shape[1], config=config
        )
        print_model_info(model)

        trainer = ModelTrainer(
            model,
            config=config,
        )
        trainer.train(dataloaders, epochs=config.TRAIN_CONFIG.EPOCHS)
        trainer.evaluate(dataloaders["test"])
        trainer.plot_results(dataloaders["test"])
        print("Experiment finished successfully.")
