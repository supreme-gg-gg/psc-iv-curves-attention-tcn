import numpy as np
from scipy.optimize import curve_fit, fsolve
import os

from multiprocessing import Pool, cpu_count

# Directories and files
DATA_DIRS = [
    "dataset/Data_10k_sets/Data_10k_rng1",
    "dataset/Data_10k_sets/Data_10k_rng2",
    "dataset/Data_10k_sets/Data_10k_rng3"
]
INPUT_FILENAME = "LHS_parameters_m.txt"
IV_FILENAME = "iV_m.txt"
OUTPUT_FILENAME = "fitted_parameters.txt"

# Voltage array as in fitting.ipynb
Va = np.concatenate((np.arange(0, 0.41, 0.1), np.arange(0.425, 1.401, 0.025)))
k_B = 1.380649e-23
q_e = 1.602176634e-19
T_cell_degC = 25
T_cell_K = T_cell_degC + 273.15
V_t = k_B * T_cell_K / q_e

class SingleDiodeModel:
    def __init__(self, thermal_voltage, experimental_I_for_guesses=None, param_bounds=None):
        self.V_t = thermal_voltage
        self.experimental_I_for_guesses = experimental_I_for_guesses
        self.V_oc_approx = 0.6 # for edge cases
        self.param_lower_bounds = param_bounds[0] if param_bounds else None

    def _diode_equation_solver(self, I, V, I_ph, I_0, n, R_s, R_sh):
        return I_ph - I_0 * (np.exp((V + I * R_s) / (n * self.V_t)) - 1) - (V + I * R_s) / R_sh - I

    def __call__(self, V_array, I_ph, I_0, n, R_s, R_sh):
        I_calculated = np.zeros_like(V_array, dtype=float)
        for i, V in enumerate(V_array):
            if self.experimental_I_for_guesses is not None and i < len(self.experimental_I_for_guesses):
                I_guess = self.experimental_I_for_guesses[i]
            else:
                I_guess = I_ph * max(0.001, (1.0 - V / self.V_oc_approx))
            try:
                sol = fsolve(
                    self._diode_equation_solver,
                    I_guess,
                    args=(V, I_ph, I_0, n, R_s, R_sh),
                    xtol=1.49012e-08,
                    maxfev=100
                )
                I_calculated[i] = sol[0]
            except Exception:
                I_calculated[i] = np.nan
        return I_calculated

def extract_params_for_curve(I_exp, V_exp):
    # Original curve_fit-based extraction (5 model parameters)
    I_sc_approx = I_exp[0]
    bounds_for_fitting = (
        [0.5 * I_sc_approx, 1e-12, 0.8, 1e-4, 10],
        [1.5 * I_sc_approx, 1e-6, 2.5, 0.5, 5000]
    )
    p0 = [
        I_sc_approx,
        2e-9,
        1.4,
        0.05,
        300
    ]
    for i in range(len(p0)):
        p0[i] = max(p0[i], bounds_for_fitting[0][i])
        p0[i] = min(p0[i], bounds_for_fitting[1][i])
    model_to_fit = SingleDiodeModel(V_t, experimental_I_for_guesses=I_exp, param_bounds=bounds_for_fitting)
    V_oc_idx = np.where(I_exp < 0.01 * I_sc_approx)[0]
    V_oc_approx = V_exp[V_oc_idx[0]] if len(V_oc_idx) > 0 else 0.62
    model_to_fit.V_oc_approx = V_oc_approx
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                model_to_fit, V_exp, I_exp, p0=p0, bounds=bounds_for_fitting, method='trf', max_nfev=1000,
            )
        return popt  # [I_ph, I_0, n, R_s, R_sh]
    except Exception:
        return [np.nan] * 5

def _fit_sdm_single(I_exp, V_exp):
    # Analytical guesses for Isc, Voc, Rs, Rsh
    I_sc_approx = I_exp[0]
    idx = np.where(I_exp <= 0)[0]
    Voc_approx = V_exp[idx[0]] if len(idx) > 0 else V_exp[-1]

    # Rs: slope at Isc
    Rs_guess = - (V_exp[1] - V_exp[0]) / (I_exp[1] - I_exp[0] + 1e-12) if len(I_exp) > 1 else 0.05

    # Rsh: slope at Voc
    Rsh_guess = - (V_exp[-1] - V_exp[-2]) / (I_exp[-1] - I_exp[-2] + 1e-12) if len(I_exp) > 1 else 300
    
    p0 = [
        I_sc_approx,
        2e-9,
        1.2,
        Rs_guess,
        Rsh_guess
    ]

    bounds_for_fitting = (
        [0.5 * I_sc_approx, 1e-12, 0.8, 1e-4, 10],
        [1.5 * I_sc_approx, 1e-6, 2.5, 0.5, 5000]
    )
    for i in range(len(p0)):
        p0[i] = max(p0[i], bounds_for_fitting[0][i])
        p0[i] = min(p0[i], bounds_for_fitting[1][i])
    model_to_fit = SingleDiodeModel(V_t, experimental_I_for_guesses=I_exp, param_bounds=bounds_for_fitting)
    model_to_fit.V_oc_approx = Voc_approx
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                model_to_fit, V_exp, I_exp, p0=p0, bounds=bounds_for_fitting, method='trf', max_nfev=400,
            )
        return popt
    except Exception:
        return [np.nan] * 5

def extract_params_parallel_optimized(Y, V_exp, n_jobs=None):
    # Y: [n_samples, n_points], V_exp: [n_points]
    # n_jobs: number of parallel processes (default: all cores)
    if n_jobs is None:
        n_jobs = cpu_count()
    args = [(Y[i, :], V_exp) for i in range(Y.shape[0])]
    with Pool(n_jobs) as pool:
        results = pool.map(_fit_sdm_single, args)
    return np.array(results)

def main():
    for data_dir in DATA_DIRS:
        input_path = os.path.join(data_dir, INPUT_FILENAME)
        iv_path = os.path.join(data_dir, IV_FILENAME)
        output_path = os.path.join(data_dir, OUTPUT_FILENAME)
        print(f"Processing {data_dir} ...")
        X = np.loadtxt(input_path, delimiter=",")
        Y = np.loadtxt(iv_path, delimiter=",")
        # Use optimized parallel SDM extraction
        print("  Starting parallel SDM parameter extraction...", flush=True)
        params_arr = extract_params_parallel_optimized(Y, Va, subsample=3)
        np.savetxt(output_path, params_arr, delimiter=",")
        print(f"Saved extracted parameters to {output_path}")

if __name__ == "__main__":
    main()