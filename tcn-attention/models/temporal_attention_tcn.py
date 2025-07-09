#!/usr/bin/env python3
"""
Temporal attention (i.e. masked multihead attention) with a TCN.
NOTE: Adapted from https://github.com/haohy/TCAN/tree/master
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils.preprocessor import DataPreprocessor
from utils.trainer import ModelTrainer
from utils.config import TCANConfig, INPUT_FILE_PARAMS, INPUT_FILE_IV, RANDOM_SEED
from utils.common import FourierFeatures, print_model_info, Chomp1d


class CausalAttentionBlock(nn.Module):
    def __init__(self, n_inputs, n_heads, key_size, dropout):
        super(CausalAttentionBlock, self).__init__()
        self.n_heads = n_heads
        self.key_size = key_size
        self.n_inputs = n_inputs

        self.linear_q = nn.Linear(n_inputs, n_heads * key_size)
        self.linear_k = nn.Linear(n_inputs, n_heads * key_size)
        self.linear_v = nn.Linear(n_inputs, n_heads * n_inputs)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(n_heads * n_inputs, n_inputs)

        # Using a registered buffer for the mask is more efficient
        self.register_buffer(
            "bias", torch.tril(torch.ones(200, 200)).view(1, 1, 200, 200)
        )

    def forward(self, x):
        B, T, C = x.size()  # Batch, SeqLen, Channels

        q = self.linear_q(x).view(B, T, self.n_heads, self.key_size).transpose(1, 2)
        k = self.linear_k(x).view(B, T, self.n_heads, self.key_size).transpose(1, 2)
        v = self.linear_v(x).view(B, T, self.n_heads, self.n_inputs).transpose(1, 2)

        att_scores = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        att_scores = att_scores.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = self.dropout(att_weights)

        y = att_weights @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.n_inputs)

        return self.proj(y), att_weights


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, n_heads, key_size, dropout, dilation
    ):
        super(TemporalBlock, self).__init__()

        self.attn = CausalAttentionBlock(n_inputs, n_heads, key_size, dropout)

        padding = (kernel_size - 1) * dilation
        self.conv_net = nn.Sequential(
            nn.Conv1d(
                n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input x has shape (Batch, Channels, SeqLen)
        x_permuted = x.permute(0, 2, 1)  # (B, T, C) for attention
        attn_out, attn_weights = self.attn(x_permuted)
        attn_out = attn_out.permute(0, 2, 1)  # (B, C, T) back

        # In the TCAN paper, attention is not residual, it feeds the conv layer
        conv_out = self.conv_net(attn_out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(conv_out + res), attn_weights


class TemporalConvNet(nn.Module):
    def __init__(
        self, num_inputs, num_channels, kernel_size, n_heads, key_size, dropout
    ):
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
                    n_heads,
                    key_size,
                    dropout,
                    dilation_size,
                )
            )
        self.network = nn.ModuleList(layers)

    def forward(self, x):
        attention_weights_list = []
        for layer in self.network:
            x, attn_weights = layer(x)
            attention_weights_list.append(attn_weights)
        return x, attention_weights_list


class PhysicsInformedTCANet(nn.Module):
    def __init__(self, num_params: int, config: TCANConfig):
        super().__init__()
        # 1. "Encoder" part: Processes the input parameters and voltage grid
        self.param_mlp = nn.Sequential(
            nn.Linear(num_params, config.DENSE_UNITS_PARAMS[0]),
            nn.BatchNorm1d(config.DENSE_UNITS_PARAMS[0]),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.DENSE_UNITS_PARAMS[0], config.DENSE_UNITS_PARAMS[1]),
            nn.BatchNorm1d(config.DENSE_UNITS_PARAMS[1]),
            nn.ReLU(),
            nn.Linear(config.DENSE_UNITS_PARAMS[1], config.DENSE_UNITS_PARAMS[2]),
            nn.BatchNorm1d(config.DENSE_UNITS_PARAMS[2]),
            nn.ReLU(),
        )
        self.fourier_features = FourierFeatures(
            config.FOURIER_NUM_BANDS, mode=config.FOURIER_MODE
        )

        # 2. Core sequence processing part
        tcan_input_channels = config.DENSE_UNITS_PARAMS[2] + (
            2 * config.FOURIER_NUM_BANDS
        )
        tcan_hidden_channels = [tcan_input_channels] * config.N_LAYERS

        self.tcan = TemporalConvNet(
            num_inputs=tcan_input_channels,
            num_channels=tcan_hidden_channels,
            kernel_size=config.KERNEL_SIZE,
            n_heads=config.N_HEADS,
            key_size=tcan_input_channels // config.N_HEADS,
            dropout=config.DROPOUT_RATE,
        )

        # 3. "Decoder" part: Projects final hidden states to the output sequence
        self.decoder = nn.Conv1d(tcan_hidden_channels[-1], 1, 1)

    def _encode_inputs(self, x_params, v_grid):
        param_embedding = self.param_mlp(x_params)
        voltage_embedding = self.fourier_features(v_grid)
        param_tiled = param_embedding.unsqueeze(1).expand(-1, v_grid.shape[1], -1)
        combined_features = torch.cat([param_tiled, voltage_embedding], dim=-1)
        return combined_features.permute(0, 2, 1)  # (B, C, T)

    def forward(self, x_params, v_grid):
        encoded_seq = self._encode_inputs(x_params, v_grid)
        tcan_out, attn_weights = self.tcan(encoded_seq)
        decoded_seq = self.decoder(tcan_out)
        return decoded_seq.squeeze(1), attn_weights


if __name__ == "__main__":
    if not (INPUT_FILE_PARAMS.exists() and INPUT_FILE_IV.exists()):
        print("Input data files not found!")
        raise FileNotFoundError("Input data files not found!")

    config = TCANConfig()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    preprocessor = DataPreprocessor(INPUT_FILE_PARAMS, INPUT_FILE_IV)
    preprocessor.load_and_prepare(truncation_threshold_pct=0.01)
    dataloaders = preprocessor.get_dataloaders(config.TRAIN_CONFIG.BATCH_SIZE)

    if preprocessor.X_clean is None:
        print("Preprocessor did not produce X_clean. Aborting.")
        raise ValueError("Preprocessor did not produce X_clean. Aborting.")

    model = PhysicsInformedTCANet(
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
    trainer.plot_attention_maps(dataloaders["test"])

    print("Experiment finished successfully.")
