from pathlib import Path
import numpy as np
from dataclasses import dataclass, field

# NOTE: Feel free to modify these paths
INPUT_FILE_PARAMS = Path(
    "/content/drive/MyDrive/PSC_IV_CURVES/Data_100k/LHS_parameters_m.txt"
)
INPUT_FILE_IV = Path("/content/drive/MyDrive/PSC_IV_CURVES/Data_100k/iV_m.txt")

V1 = np.arange(0.0, 0.4 + 1e-8, 0.1)
V2 = np.arange(0.425, 1.4 + 1e-8, 0.025)
FULL_VOLTAGE_GRID = np.concatenate([V1, V2]).astype(np.float32)
MIN_LEN_FOR_PROCESSING = 5
RANDOM_SEED = 42
PARAM_COLNAMES = [
    "Eg",
    "NCv",
    "NCc",
    "mu_e",
    "mu_h",
    "eps",
    "A",
    "Cn",
    "Cp",
    "Nt",
    "Et",
    "nD",
    "nA",
    "thickness",
    "T",
    "Sn",
    "Sp",
    "Rs",
    "Rsh",
    "G",
    "light_intensity",
    "Voc_ref",
    "Jsc_ref",
    "FF_ref",
    "PCE_ref",
    "Qe_loss",
    "R_loss",
    "SRH_loss",
    "series_loss",
    "shunt_loss",
    "other_loss",
]

COLNAMES_INTERP_MODEL = [
    "lH",
    "lP",
    "lE",
    "muHh",
    "muPh",
    "muPe",
    "muEe",
    "NvH",
    "NcH",
    "NvE",
    "NcE",
    "NvP",
    "NcP",
    "chiHh",
    "chiHe",
    "chiPh",
    "chiPe",
    "chiEh",
    "chiEe",
    "Wlm",
    "Whm",
    "epsH",
    "epsP",
    "epsE",
    "Gavg",
    "Aug",
    "Brad",
    "Taue",
    "Tauh",
    "vII",
    "vIII",
]


@dataclass
class LossWeights:
    mse: float = 0.776
    monotonicity: float = 0.0008
    curvature: float = 0.003
    jsc: float = 0.1
    voc: float = 0.16
    knee_window_size: int = 4
    knee_weight_factor: float = 2.13


@dataclass
class TrainConfig:
    LEARNING_RATE: float = 0.002
    EPOCHS: int = 30
    BATCH_SIZE: int = 128


@dataclass
class TCNConfig:
    DENSE_UNITS_PARAMS: tuple = (256, 128, 128)
    FILTERS: tuple = (128, 128)
    FOURIER_NUM_BANDS: int = 16
    FOURIER_MODE: str = "gaussian"
    N_LAYERS: int = 2  # Does not have N_HEADS because this does not use attention
    KERNEL_SIZE: int = 5
    DROPOUT_RATE: float = 0.28
    LOSS_WEIGHTS: LossWeights = field(default_factory=LossWeights)
    TRAIN_CONFIG: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class TCANConfig:
    DENSE_UNITS_PARAMS: tuple = (256, 128, 128)
    FOURIER_NUM_BANDS: int = 16
    FOURIER_MODE: str = "gaussian"
    N_LAYERS: int = 2
    N_HEADS: int = 4
    KERNEL_SIZE: int = 5
    DROPOUT_RATE: float = 0.28
    LOSS_WEIGHTS: LossWeights = field(default_factory=LossWeights)
    TRAIN_CONFIG: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class DINAConfig:
    DENSE_UNITS_PARAMS: tuple = (256, 128, 128)
    FOURIER_NUM_BANDS: int = 16
    FOURIER_MODE: str = "gaussian"
    N_LAYERS: int = 2
    N_HEADS: int = 4
    KERNEL_SIZE: int = 5
    DROPOUT_RATE: float = 0.28
    LOSS_WEIGHTS: LossWeights = field(default_factory=LossWeights)
    TRAIN_CONFIG: TrainConfig = field(default_factory=TrainConfig)
