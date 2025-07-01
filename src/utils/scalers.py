import numpy as np


class GlobalValueScaler:
    """
    A robust scaler that normalizes data by a global reference value and maps to [0, 1].
    Acts like a proper sklearn scaler with complete state management.

    Can be used for ISC (first column) or Voc (last column) values.

    Transformation pipeline:
    1. Normalize by global reference: v_norm = v / global_reference
    2. Map to [0, 1]: v_scaled = v_norm (already in [0, 1] since v <= reference)

    Inverse transformation:
    1. Denormalize: v = v_scaled * global_reference
    """

    def __init__(self, method="median", value_type="voc"):
        """
        Initialize the scaler.

        Args:
            method: "median" or "mean" for computing global reference
            value_type: "voc", "isc", or "auto" - determines which column to use for reference
        """
        self.method = method
        self.value_type = value_type.lower()
        self.global_reference = None
        self.is_fitted = False

    def fit(self, data):
        """
        Fit the scaler on training data.

        Args:
            data: 2D array (n_samples, n_features) or 1D array of reference values
                 For 2D:
                   - if value_type="isc": uses first column (ISC values)
                   - if value_type="voc": uses last column (Voc values)
                   - if value_type="auto": uses max value per sample
                 For 1D: uses the values directly
        """
        data = np.asarray(data)

        if data.ndim == 2:
            if self.value_type == "isc":
                reference_values = data[:, 0]  # First column is ISC
            elif self.value_type == "voc":
                reference_values = data[:, -1]  # Last column is Voc
            elif self.value_type == "auto":
                reference_values = np.max(data, axis=1)  # Max value per sample
            else:
                raise ValueError("value_type must be 'isc', 'voc', or 'auto'")
        else:
            reference_values = data

        if self.method == "median":
            self.global_reference = float(np.median(reference_values))
        elif self.method == "mean":
            self.global_reference = float(np.mean(reference_values))
        elif self.method == "max":
            self.global_reference = float(np.max(reference_values))
        else:
            raise ValueError("method must be 'mean' or 'median' or 'max'")

        if self.global_reference <= 0:
            raise ValueError(
                f"Global reference must be positive, got {self.global_reference}"
            )

        self.is_fitted = True
        return self

    def transform(self, data):
        """
        Transform data to [0, 1] range using global reference normalization.

        Args:
            data: Array of data to transform

        Returns:
            data_scaled: Transformed data in [0, 1] range
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming")

        data = np.asarray(data)

        # Normalize by global reference
        data_scaled = data / self.global_reference

        return data_scaled

    def inverse_transform(self, data_scaled):
        """
        Inverse transform data from [0, 1] back to original scale.

        Args:
            data_scaled: Scaled data in [0, 1] range

        Returns:
            data: Data in original physical units
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transforming")

        data_scaled = np.asarray(data_scaled)

        # Denormalize by global reference
        data = data_scaled * self.global_reference

        return data

    def fit_transform(self, data):
        """
        Fit the scaler and transform the data in one step.
        """
        return self.fit(data).transform(data)

    def get_reference(self):
        """Get the fitted global reference value."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted first")
        return self.global_reference

    # Backward compatibility methods
    def get_isc(self):
        """Get the fitted global ISC value (backward compatibility)."""
        if self.value_type != "isc":
            raise ValueError("This scaler was not fitted for ISC values")
        return self.get_reference()

    def get_voc(self):
        """Get the fitted global Voc value (backward compatibility)."""
        if self.value_type != "voc":
            raise ValueError("This scaler was not fitted for Voc values")
        return self.get_reference()

    def get_params(self, deep=True):
        """Get parameters for this scaler (sklearn compatibility)."""
        return {"method": self.method, "value_type": self.value_type}

    def set_params(self, **params):
        """Set parameters for this scaler (sklearn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self):
        return (
            f"GlobalValueScaler(method='{self.method}', "
            f"value_type='{self.value_type}', fitted={self.is_fitted})"
        )