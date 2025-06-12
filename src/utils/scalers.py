import numpy as np


class GlobalISCScaler:
    """
    A robust scaler that normalizes IV curves by a global ISC value and maps to [-1, 1].
    Acts like a proper sklearn scaler with complete state management.

    Transformation pipeline:
    1. Normalize by global ISC: y_norm = y / global_isc
    2. Map to [-1, 1]: y_scaled = 2.0 * y_norm - 1.0

    Inverse transformation:
    1. Map back to [0, 1]: y_norm = (y_scaled + 1.0) / 2.0
    2. Denormalize: y = y_norm * global_isc
    """

    def __init__(self, method="median"):
        self.method = method
        self.global_isc = None
        self.is_fitted = False

    def fit(self, y_data):
        """
        Fit the scaler on training data.

        Args:
            y_data: 2D array (n_samples, n_features) where first column is ISC values
                   OR 1D array of ISC values
        """
        if y_data.ndim == 2:
            isc_values = y_data[:, 0]  # First column is ISC
        else:
            isc_values = y_data

        if self.method == "median":
            self.global_isc = float(np.median(isc_values))
        elif self.method == "mean":
            self.global_isc = float(np.mean(isc_values))
        else:
            raise ValueError("method must be 'mean' or 'median'")

        if self.global_isc <= 0:
            raise ValueError(f"Global ISC must be positive, got {self.global_isc}")

        self.is_fitted = True
        return self

    def transform(self, y):
        """
        Transform data to [-1, 1] range using global ISC normalization.

        Args:
            y: Array of IV curve data to transform

        Returns:
            y_scaled: Transformed data in [-1, 1] range
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming")

        y = np.asarray(y)

        # Step 1: Normalize by global ISC
        y_norm = y / self.global_isc

        # Step 2: Map to [-1, 1]
        y_scaled = 2.0 * y_norm - 1.0

        return y_scaled

    def inverse_transform(self, y_scaled):
        """
        Inverse transform data from [-1, 1] back to original scale.

        Args:
            y_scaled: Scaled data in [-1, 1] range

        Returns:
            y: Data in original physical units
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transforming")

        y_scaled = np.asarray(y_scaled)

        # Step 1: Map back from [-1, 1] to [0, 1]
        y_norm = (y_scaled + 1.0) / 2.0

        # Step 2: Denormalize by global ISC
        y = y_norm * self.global_isc

        return y

    def fit_transform(self, y):
        """
        Fit the scaler and transform the data in one step.
        """
        return self.fit(y).transform(y)

    def get_isc(self):
        """Get the fitted global ISC value."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted first")
        return self.global_isc

    def get_params(self, deep=True):
        """Get parameters for this scaler (sklearn compatibility)."""
        return {"method": self.method}

    def set_params(self, **params):
        """Set parameters for this scaler (sklearn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self):
        return f"GlobalISCScaler(method='{self.method}', fitted={self.is_fitted})"


class GlobalVocScaler:
    """
    A robust scaler that normalizes voltage curves by a global Voc value and maps to [0, 1].
    Acts like a proper sklearn scaler with complete state management.

    Transformation pipeline:
    1. Normalize by global Voc: v_norm = v / global_voc
    2. Map to [0, 1]: v_scaled = v_norm (already in [0, 1] since v <= voc)

    Inverse transformation:
    1. Denormalize: v = v_scaled * global_voc
    """

    def __init__(self, method="median"):
        self.method = method
        self.global_voc = None
        self.is_fitted = False

    def fit(self, v_data):
        """
        Fit the scaler on training data.

        Args:
            v_data: 2D array (n_samples, n_features) where last column is Voc values
                   OR 1D array of Voc values
        """
        if v_data.ndim == 2:
            voc_values = v_data[:, -1]  # Last column is Voc
        else:
            voc_values = v_data

        if self.method == "median":
            self.global_voc = float(np.median(voc_values))
        elif self.method == "mean":
            self.global_voc = float(np.mean(voc_values))
        else:
            raise ValueError("method must be 'mean' or 'median'")

        if self.global_voc <= 0:
            raise ValueError(f"Global Voc must be positive, got {self.global_voc}")

        self.is_fitted = True
        return self

    def transform(self, v):
        """
        Transform data to [0, 1] range using global Voc normalization.

        Args:
            v: Array of voltage data to transform

        Returns:
            v_scaled: Transformed data in [0, 1] range
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming")

        v = np.asarray(v)

        # Normalize by global Voc
        v_scaled = v / self.global_voc

        return v_scaled

    def inverse_transform(self, v_scaled):
        """
        Inverse transform data from [0, 1] back to original scale.

        Args:
            v_scaled: Scaled data in [0, 1] range

        Returns:
            v: Data in original physical units
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transforming")

        v_scaled = np.asarray(v_scaled)

        # Denormalize by global Voc
        v = v_scaled * self.global_voc

        return v

    def fit_transform(self, v):
        """
        Fit the scaler and transform the data in one step.
        """
        return self.fit(v).transform(v)

    def get_voc(self):
        """Get the fitted global Voc value."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted first")
        return self.global_voc

    def get_params(self, deep=True):
        """Get parameters for this scaler (sklearn compatibility)."""
        return {"method": self.method}

    def set_params(self, **params):
        """Set parameters for this scaler (sklearn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self):
        return f"GlobalVocScaler(method='{self.method}', fitted={self.is_fitted})"
