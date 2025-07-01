from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# NOTE: These imports are basically constants instead of user config for models
# Therefore, this will never need to be changed
from config import (
    PARAM_COLNAMES,
    MIN_LEN_FOR_PROCESSING,
    FULL_VOLTAGE_GRID,
    RANDOM_SEED,
)


class DataPreprocessor:
    """Orchestrates loading, cleaning, and transforming data for the model."""

    def __init__(self, params_path: Path, iv_path: Path):
        self.params_path = params_path
        self.iv_path = iv_path
        self.param_transformer, self.scalar_scaler = None, None
        (
            self.X_clean,
            self.y_padded_scaled,
            self.v_padded,
            self.masks,
            self.orig_lengths,
            self.per_curve_isc,
        ) = [None] * 6

    def load_and_prepare(self, truncation_threshold_pct: float = 0.01):
        print("=== Starting Data Preprocessing Pipeline ===")
        params_df = pd.read_csv(self.params_path, header=None, names=PARAM_COLNAMES)
        iv_data_np = np.loadtxt(self.iv_path, delimiter=",", dtype=np.float32)
        raw_currents, raw_voltages, valid_indices = [], [], []
        for i in range(iv_data_np.shape[0]):
            v_trunc, c_trunc = self._truncate_iv_curve(
                iv_data_np[i], truncation_threshold_pct
            )
            if v_trunc is not None:
                raw_voltages.append(v_trunc)
                raw_currents.append(c_trunc)
                valid_indices.append(i)
        params_df_valid = params_df.iloc[valid_indices].reset_index(drop=True)
        per_curve_isc_np, scaled_curves = zip(
            *[self._normalize_and_scale_by_isc(c) for c in raw_currents]
        )
        y_padded, v_padded, masks, lengths = self._pad_and_create_mask(
            list(scaled_curves), raw_voltages
        )
        self.y_padded_scaled = torch.from_numpy(y_padded)
        self.v_padded = torch.from_numpy(v_padded)
        self.masks = torch.from_numpy(masks)
        self.orig_lengths = torch.from_numpy(lengths)
        self.per_curve_isc = torch.from_numpy(np.array(per_curve_isc_np))
        scalar_df = pd.DataFrame(
            {
                "Vknee_raw": [v[-1] for v in raw_voltages],
                "Imax_raw": [np.max(c) for c in raw_currents],
                "Imin_raw": [np.min(c) for c in raw_currents],
                "Imean_raw": [np.mean(c) for c in raw_currents],
            }
        )
        X_params_processed, _ = self._preprocess_input_parameters(params_df_valid)
        X_scalar_processed, _ = self._preprocess_scalar_features(scalar_df)
        X_clean_np = np.concatenate(
            [X_params_processed, X_scalar_processed], axis=1
        ).astype(np.float32)
        if np.any(np.isnan(X_clean_np)) or np.any(np.isinf(X_clean_np)):
            print("Warning: NaN/inf detected in final input data. Clamping values.")
            X_clean_np = np.nan_to_num(X_clean_np, nan=0.0, posinf=1e6, neginf=-1e6)
        self.X_clean = torch.from_numpy(X_clean_np)
        print(
            f"Preprocessing complete. Final shapes: X={self.X_clean.shape}, y={self.y_padded_scaled.shape}"
        )

    def get_dataloaders(self, batch_size: int):
        dataset = TensorDataset(
            self.X_clean,
            self.y_padded_scaled,
            self.v_padded,
            self.masks,
            self.orig_lengths,
            self.per_curve_isc,
        )
        train_val_idx, test_idx = train_test_split(
            range(len(dataset)), test_size=0.2, random_state=RANDOM_SEED
        )
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.15, random_state=RANDOM_SEED
        )
        dataloaders = {}
        for name, indices in [
            ("train", train_idx),
            ("val", val_idx),
            ("test", test_idx),
        ]:
            subset = torch.utils.data.Subset(dataset, indices)
            shuffle = True if name == "train" else False
            dataloaders[name] = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=True,
                num_workers=2,
            )
        return dataloaders

    def _truncate_iv_curve(self, curve, threshold_pct):
        if curve.size == 0 or (isc_val := curve[0]) <= 0:
            return None, None
        threshold = threshold_pct * isc_val
        indices = np.where(curve < threshold)[0]
        trunc_idx = indices[0] if indices.size > 0 else curve.shape[0]
        if trunc_idx < MIN_LEN_FOR_PROCESSING:
            return None, None
        return FULL_VOLTAGE_GRID[:trunc_idx].copy(), curve[:trunc_idx].copy()

    def _normalize_and_scale_by_isc(self, curve):
        return curve[0], (2.0 * (curve / curve[0]) - 1.0).astype(np.float32)

    def _pad_and_create_mask(self, scaled_curves, volt_curves):
        lengths = np.array([c.size for c in scaled_curves])
        max_len = lengths.max()
        y_padded = np.zeros((len(scaled_curves), max_len), dtype=np.float32)
        v_padded, mask = np.copy(y_padded), np.copy(y_padded)
        for i, (c, v, l) in enumerate(zip(scaled_curves, volt_curves, lengths)):
            y_padded[i, :l], v_padded[i, :l], mask[i, :l] = c, v, 1.0
            if l < max_len:
                y_padded[i, l:], v_padded[i, l:] = c[-1], v[-1]
        return y_padded, v_padded, mask, lengths.astype(np.int32)

    def _preprocess_input_parameters(self, params_df):
        const_cols = [c for c in params_df.columns if params_df[c].std(ddof=0) <= 1e-10]
        if const_cols:
            params_df = params_df.drop(columns=const_cols)
        param_defs = {
            "material": ["Eg", "NCv", "NCc", "mu_e", "mu_h", "eps"],
            "device": [
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
            ],
            "operating": ["G", "light_intensity"],
            "reference": ["Voc_ref", "Jsc_ref", "FF_ref", "PCE_ref"],
            "loss": [
                "Qe_loss",
                "R_loss",
                "SRH_loss",
                "series_loss",
                "shunt_loss",
                "other_loss",
            ],
        }
        all_group_cols = [c for cols in param_defs.values() for c in cols]
        params_df = params_df[[c for c in params_df.columns if c in all_group_cols]]
        transformers = [
            (
                group,
                Pipeline(
                    [("log1p", FunctionTransformer(np.log1p))]
                    if group == "material"
                    else []
                    + [
                        ("robust", RobustScaler()),
                        ("minmax", MinMaxScaler(feature_range=(-1, 1))),
                    ]
                ),
                [c for c in cols if c in params_df.columns],
            )
            for group, cols in param_defs.items()
            if any(c in params_df.columns for c in cols)
        ]
        self.param_transformer = ColumnTransformer(
            transformers, remainder="passthrough"
        )
        X_params = self.param_transformer.fit_transform(params_df)
        extreme_mask = np.abs(X_params) > 1e10
        if np.any(extreme_mask):
            col_names = self.param_transformer.get_feature_names_out()
            rows, cols = np.where(extreme_mask)
            for r, c in zip(rows, cols):
                print(
                    f"[WARNING] Extreme value detected in input features: row {r}, column '{col_names[c]}', value {X_params[r, c]}"
                )
        return X_params, self.param_transformer

    def _preprocess_scalar_features(self, scalar_df):
        scaler = Pipeline(
            [
                ("robust", RobustScaler()),
                ("minmax", MinMaxScaler(feature_range=(-1, 1))),
            ]
        )
        return scaler.fit_transform(scalar_df), scaler
