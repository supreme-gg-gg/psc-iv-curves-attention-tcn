from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    RobustScaler,
    MinMaxScaler,
    FunctionTransformer,
    StandardScaler,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import logging
import typing
import joblib
from tqdm import tqdm
from scipy.interpolate import PchipInterpolator

# NOTE: These imports are basically constants instead of user config for models
# Therefore, this will never need to be changed
from config import (
    PARAM_COLNAMES,
    MIN_LEN_FOR_PROCESSING,
    FULL_VOLTAGE_GRID,
    RANDOM_SEED,
    COLNAMES_INTERP_MODEL,
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


log = logging.getLogger(__name__)


def process_iv_with_pchip(
    iv_raw: np.ndarray,
    full_v_grid: np.ndarray,
    n_pre: int,
    n_post: int,
    v_max: float,
    n_fine: int,
) -> typing.Optional[tuple]:
    """Identical logic to the TF version, but with improved type hints."""
    seq_len = n_pre + 1 + n_post
    try:
        pi = PchipInterpolator(full_v_grid, iv_raw, extrapolate=False)
        v_fine = np.linspace(0, v_max, n_fine)
        i_fine = pi(v_fine)
        valid_mask = ~np.isnan(i_fine)
        v_fine, i_fine = v_fine[valid_mask], i_fine[valid_mask]
        if v_fine.size < 2:
            return None
        zero_cross_idx = np.where(i_fine <= 0)[0]
        voc_v = v_fine[zero_cross_idx[0]] if len(zero_cross_idx) > 0 else v_fine[-1]
        v_search_mask = v_fine <= voc_v
        v_search, i_search = v_fine[v_search_mask], i_fine[v_search_mask]
        if v_search.size == 0:
            return None
        power = v_search * i_search
        mpp_idx = np.argmax(power)
        v_mpp = v_search[mpp_idx]
        v_pre_mpp = np.linspace(v_search[0], v_mpp, n_pre + 2, endpoint=True)[:-1]
        v_post_mpp = np.linspace(v_mpp, v_search[-1], n_post + 2, endpoint=True)[1:]
        v_mpp_grid = np.unique(np.concatenate([v_pre_mpp, v_post_mpp]))
        v_slice = np.interp(
            np.linspace(0, 1, seq_len), np.linspace(0, 1, len(v_mpp_grid)), v_mpp_grid
        )
        i_slice = pi(v_slice)
        if np.any(np.isnan(i_slice)) or i_slice.shape[0] != seq_len:
            return None
        return (
            v_slice.astype(np.float32),
            i_slice.astype(np.float32),
            (v_fine.astype(np.float32), i_fine.astype(np.float32)),
        )
    except (ValueError, IndexError):
        return None


def normalize_and_scale_by_isc(curve: np.ndarray) -> tuple[float, np.ndarray]:
    """Scales curve to [-1, 1] and returns the Isc value."""
    isc_val = float(curve[0])
    return isc_val, (2.0 * (curve / isc_val) - 1.0).astype(np.float32)


def compute_curvature_weights(
    y_curves: np.ndarray, alpha: float, power: float
) -> np.ndarray:
    """Computes sample weights based on curvature of the scaled curves."""
    padded = np.pad(y_curves, ((0, 0), (1, 1)), mode="edge")
    kappa = np.abs(padded[:, 2:] - 2 * padded[:, 1:-1] + padded[:, :-2])
    max_kappa = np.max(kappa, axis=1, keepdims=True)
    max_kappa[max_kappa < 1e-9] = 1.0
    return (1.0 + alpha * np.power(kappa / max_kappa, power)).astype(np.float32)


def get_param_transformer(colnames: list[str]) -> ColumnTransformer:
    """Builds the sklearn ColumnTransformer for input parameters."""
    param_defs = {
        "layer_thickness": ["lH", "lP", "lE"],
        "material_properties": [
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
            "epsH",
            "epsP",
            "epsE",
        ],
        "contacts": ["Wlm", "Whm"],
        "recombination_gen": ["Gavg", "Aug", "Brad", "Taue", "Tauh", "vII", "vIII"],
    }
    transformers = []
    for group, cols in param_defs.items():
        actual_cols = [c for c in cols if c in colnames]
        if not actual_cols:
            continue
        steps = [
            ("robust", RobustScaler()),
            ("minmax", MinMaxScaler(feature_range=(-1, 1))),
        ]
        if group == "material_properties":
            steps.insert(0, ("log1p", FunctionTransformer(func=np.log1p)))
        transformers.append((group, Pipeline(steps), actual_cols))
    return ColumnTransformer(transformers, remainder="passthrough")


class InterpModelDataset(Dataset):
    """Dataset for fixed-length I-V curves"""

    def __init__(self, cfg: dict, split: str, param_tf, scalar_tf):
        self.cfg = cfg
        self.split = split
        data = np.load(cfg["dataset"]["paths"]["preprocessed_npz"], allow_pickle=True)

        # FIXED INDEXING. Now it should be the same as the MATLAB code (MATLAB would be one index ahead)
        split_labels = data["split_labels"]
        indices = np.where(split_labels == split)[0]

        self.v_slices = torch.from_numpy(data["v_slices"][indices])
        self.i_slices_scaled = torch.from_numpy(data["i_slices_scaled"][indices])
        self.sample_weights = torch.from_numpy(data["sample_weights"][indices])

        params_df = pd.read_csv(
            cfg["dataset"]["paths"]["params_csv"],
            header=None,
            names=COLNAMES_INTERP_MODEL,
        )
        params_df_valid = params_df.iloc[data["valid_indices"]].reset_index(drop=True)
        scalar_df = pd.DataFrame(
            {
                "I_ref": data["i_slices"][:, 0],
                "V_mpp": data["v_slices"][:, 3],
                "I_mpp": data["i_slices"][:, 3],
            }
        )

        X_params_full = param_tf.transform(params_df_valid).astype(np.float32)
        X_scalar_full = scalar_tf.transform(scalar_df).astype(np.float32)
        X_combined = np.concatenate([X_params_full, X_scalar_full], axis=1)
        self.X = torch.from_numpy(X_combined[indices])

        # Physical values for evaluation
        self.isc_vals = torch.from_numpy(data["isc_vals"][indices])

    def __len__(self):
        return len(self.v_slices)

    def __getitem__(self, idx):
        return {
            "X_combined": self.X[idx],
            "voltage": self.v_slices[idx],
            "current_scaled": self.i_slices_scaled[idx],
            "sample_w": self.sample_weights[idx],
            "isc": self.isc_vals[idx],
        }


class InterpModelDataPreprocessor:
    """Data preprocessor for fixed-length sequences"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.param_tf = None
        self.scalar_tf = None

    def prepare_data(self):
        """This method is called once per node. Good for downloads, etc."""
        if not Path(self.cfg["dataset"]["paths"]["preprocessed_npz"]).exists():
            log.info("Preprocessed data not found. Running preprocessing...")
            self._preprocess_and_save()
        else:
            log.info("Found preprocessed data. Skipping preprocessing.")

    def setup(self, stage: str | None = None):
        """Setup datasets."""
        if self.param_tf is None:
            self.param_tf = joblib.load(
                self.cfg["dataset"]["paths"]["param_transformer"]
            )
            self.scalar_tf = joblib.load(
                self.cfg["dataset"]["paths"]["scalar_transformer"]
            )

        if stage == "fit" or stage is None:
            self.train_dataset = InterpModelDataset(
                self.cfg, "train", self.param_tf, self.scalar_tf
            )
            self.val_dataset = InterpModelDataset(
                self.cfg, "val", self.param_tf, self.scalar_tf
            )
        if stage == "test" or stage is None:
            self.test_dataset = InterpModelDataset(
                self.cfg, "test", self.param_tf, self.scalar_tf
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, **self.cfg["dataset"]["dataloader"], shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.cfg["dataset"]["dataloader"])

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.cfg["dataset"]["dataloader"])

    def _preprocess_and_save(self):
        """PCHIP preprocessing pipeline (from another_model.py)."""
        log.info("--- Starting Data Preprocessing ---")
        cfg = self.cfg
        paths = cfg["dataset"]["paths"]
        params_df = pd.read_csv(
            paths["params_csv"], header=None, names=COLNAMES_INTERP_MODEL
        )
        iv_data_raw = np.loadtxt(paths["iv_raw_txt"], delimiter=",")
        full_v_grid = np.concatenate(
            [np.arange(0, 0.4 + 1e-8, 0.1), np.arange(0.425, 1.4 + 1e-8, 0.025)]
        ).astype(np.float32)

        pchip_cfg = cfg["dataset"]["pchip"]
        pchip_args = (
            full_v_grid,
            pchip_cfg["n_pre_mpp"],
            pchip_cfg["n_post_mpp"],
            pchip_cfg["v_max"],
            pchip_cfg["n_fine"],
        )
        results = [
            process_iv_with_pchip(iv_data_raw[i], *pchip_args)
            for i in tqdm(range(len(iv_data_raw)), desc="PCHIP")
        ]

        valid_indices, v_slices, i_slices, fine_curves_tuples = [], [], [], []
        for i, res in enumerate(results):
            if (
                res is not None and res[1][0] > 1e-9
            ):  # Filter out zero/negative Isc curves
                valid_indices.append(i)
                v_slices.append(res[0])
                i_slices.append(res[1])
                fine_curves_tuples.append(res[2])

        log.info(
            f"Retained {len(valid_indices)} / {len(iv_data_raw)} valid curves after PCHIP & Isc filtering."
        )
        v_slices = np.array(v_slices)
        i_slices = np.array(i_slices)
        valid_indices = np.array(valid_indices)

        # Normalize and compute weights
        isc_vals, i_slices_scaled = zip(
            *[normalize_and_scale_by_isc(c) for c in i_slices]
        )
        isc_vals = np.array(isc_vals)
        i_slices_scaled = np.array(i_slices_scaled)
        sample_weights = compute_curvature_weights(
            i_slices_scaled, **cfg["dataset"]["curvature_weighting"]
        )

        # Feature Engineering & Scaling
        params_df_valid = params_df.iloc[valid_indices].reset_index(drop=True)
        param_transformer = get_param_transformer(COLNAMES_INTERP_MODEL)
        param_transformer.fit(params_df_valid)
        joblib.dump(param_transformer, paths["param_transformer"])
        scalar_df = pd.DataFrame(
            {
                "I_ref": i_slices[:, 0],
                "V_mpp": v_slices[:, pchip_cfg["n_pre_mpp"]],
                "I_mpp": i_slices[:, pchip_cfg["n_pre_mpp"]],
            }
        )
        scalar_transformer = Pipeline([("scaler", StandardScaler())])
        scalar_transformer.fit(scalar_df)
        joblib.dump(scalar_transformer, paths["scalar_transformer"])

        # Calculate and store the parameter dimensions
        param_dim = param_transformer.transform(params_df_valid).shape[1]
        scalar_dim = scalar_transformer.transform(scalar_df).shape[1]
        self.cfg["model"]["param_dim"] = param_dim + scalar_dim
        log.info(
            f"Total parameter dimension calculated: {self.cfg['model']['param_dim']} ({param_dim} params + {scalar_dim} scalars)"
        )

        # Create data splits using a labels array
        all_indices = np.arange(len(valid_indices))
        train_val_idx, test_idx = train_test_split(
            all_indices, test_size=0.2, random_state=cfg["train"]["seed"]
        )
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.15, random_state=cfg["train"]["seed"]
        )
        split_labels = np.array([""] * len(all_indices), dtype=object)
        split_labels[train_idx] = "train"
        split_labels[val_idx] = "val"
        split_labels[test_idx] = "test"

        # Don't store fine curves in object array, they will be in dense padded arrays
        max_len = max(len(v) for v, i in fine_curves_tuples)
        v_fine_padded = np.full(
            (len(fine_curves_tuples), max_len), np.nan, dtype=np.float32
        )
        i_fine_padded = np.full(
            (len(fine_curves_tuples), max_len), np.nan, dtype=np.float32
        )
        for i, (v, c) in enumerate(fine_curves_tuples):
            v_fine_padded[i, : len(v)] = v
            i_fine_padded[i, : len(c)] = c

        np.savez(
            paths["preprocessed_npz"],
            v_slices=v_slices,
            i_slices=i_slices,
            i_slices_scaled=i_slices_scaled,
            sample_weights=sample_weights,
            isc_vals=isc_vals,
            valid_indices=valid_indices,
            split_labels=split_labels,
            v_fine_padded=v_fine_padded,
            i_fine_padded=i_fine_padded,
        )
        log.info(
            f"Saved all preprocessed data and transformers to {paths['output_dir']}"
        )
