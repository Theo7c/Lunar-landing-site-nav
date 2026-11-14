import os
import json
import numpy as np
import pandas as pd

BASE_SITES_CSV = "processed_data/landing_sites.csv"
CUSTOM_SITES_CSV = "processed_data/user_inspected_sites.csv"
SSIS_CSV = "processed_data/ssis.csv"
CALIBRATION_JSON = "processed_data/score_calibration.json"


def load_sites_and_ssis(
    base_csv: str = BASE_SITES_CSV,
    custom_csv: str = CUSTOM_SITES_CSV,
    ssis_csv: str = SSIS_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    site_dfs = []
    if os.path.exists(base_csv):
        site_dfs.append(pd.read_csv(base_csv))
    if os.path.exists(custom_csv):
        site_dfs.append(pd.read_csv(custom_csv))
    sites_df = pd.concat(site_dfs, ignore_index=True) if site_dfs else pd.DataFrame()
    ssis_df = pd.read_csv(ssis_csv) if os.path.exists(ssis_csv) else pd.DataFrame()
    return sites_df, ssis_df


def compute_raw_safety(
    max_slope_deg: np.ndarray,
    mean_slope_deg: np.ndarray,
    roughness_m: np.ndarray,
    max_slope_limit_deg: float = 15.0,
    mean_slope_limit_deg: float = 10.0,
    roughness_limit_m: float = 25.0,
    slope_weight: float = 0.60,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (feasible_mask, raw_safety_values)."""
    alpha = float(np.clip(slope_weight, 0.0, 1.0))
    max_limit = max(1e-6, float(max_slope_limit_deg))
    rough_limit = max(1e-6, float(roughness_limit_m))

    feasible = (max_slope_deg <= max_limit) & (mean_slope_deg <= mean_slope_limit_deg)
    slope_term = np.clip(1.0 - (max_slope_deg / max_limit), 0.0, 1.0)
    rough_term = np.clip(1.0 - (roughness_m / rough_limit), 0.0, 1.0)

    raw_safety = np.where(feasible, alpha * slope_term + (1.0 - alpha) * rough_term, 0.0)
    return feasible, raw_safety


def fit_distribution(values: np.ndarray, method: str = "uniform") -> dict:
    """Fit either 201-knot ECDF quantiles ('uniform') or mean/std ('sigmoid') on positive values."""
    valid = values[np.isfinite(values) & (values > 0.0)]
    if len(valid) == 0:
        return {"method": method, "empty": True}

    if method == "sigmoid":
        mean_val = float(np.mean(valid))
        std_val = float(np.std(valid))
        return {
            "method": "sigmoid",
            "empty": False,
            "mean": mean_val,
            "std": max(1e-6, std_val),
        }

    # Default: 'uniform' via 201 quantile knots (0.0%, 0.5%, ..., 100.0%)
    probs = np.linspace(0.0, 1.0, 201)
    knots = np.quantile(valid, probs).tolist()
    return {
        "method": "uniform",
        "empty": False,
        "knots": knots,
        "probs": probs.tolist(),
    }


def apply_distribution(values: np.ndarray, dist_params: dict) -> np.ndarray:
    """Transform raw values into [0.0, 1.0] using pre-fitted grid parameters."""
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(arr, dtype=np.float64)
    pos = np.isfinite(arr) & (arr > 0.0)

    if not np.any(pos) or dist_params.get("empty", True):
        return out

    if dist_params["method"] == "sigmoid":
        z = (arr[pos] - dist_params["mean"]) / dist_params["std"]
        out[pos] = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
    else:
        # Uniform ECDF interpolation across the grid distribution
        out[pos] = np.interp(arr[pos], dist_params["knots"], dist_params["probs"])

    return np.clip(out, 0.0, 1.0)


def calculate_all_scores(
    sites_df: pd.DataFrame | None = None,
    ssis_df: pd.DataFrame | None = None,
    max_slope_limit_deg: float = 15.0,
    mean_slope_limit_deg: float = 10.0,
    roughness_limit_m: float = 25.0,
    slope_weight: float = 0.60,
    r_max_km: float = 12.0,
    decay_rate: float = 0.15,
    method: str = "uniform",
    filter_infeasible: bool = True,
    save_calibration: bool = True,
    chunk_size: int = 5_000,
) -> tuple[pd.DataFrame, dict]:
    """
    Compute grid-standardized s_safety and s_science in [0.0, 1.0] across all sites,
    and save the fitted standardization parameters so single-point inspection uses
    the exact same scale.
    """
    if sites_df is None or ssis_df is None:
        loaded_sites, loaded_ssis = load_sites_and_ssis()
        sites_df = loaded_sites if sites_df is None else sites_df
        ssis_df = loaded_ssis if ssis_df is None else ssis_df

    df = sites_df.copy()
    if df.empty:
        return df, {}

    # 1. Raw Safety & Feasibility
    feasible, raw_safety = compute_raw_safety(
        max_slope_deg=df["max_slope_deg"].to_numpy(dtype=np.float64),
        mean_slope_deg=df["mean_slope_deg"].to_numpy(dtype=np.float64),
        roughness_m=df["roughness_m"].to_numpy(dtype=np.float64),
        max_slope_limit_deg=max_slope_limit_deg,
        mean_slope_limit_deg=mean_slope_limit_deg,
        roughness_limit_m=roughness_limit_m,
        slope_weight=slope_weight,
    )
    df["feasible"] = feasible
    df["raw_safety"] = raw_safety

    if filter_infeasible:
        df = df[df["feasible"]].reset_index(drop=True)

    # 2. Raw Science via chunked 3D Euclidean distances
    n_sites = len(df)
    reachable_counts = np.zeros(n_sites, dtype=np.int32)
    raw_science = np.zeros(n_sites, dtype=np.float64)

    if not ssis_df.empty and r_max_km > 0.0 and n_sites > 0:
        site_xyz = df[["x_km", "y_km", "z_km"]].to_numpy(dtype=np.float32)
        ssi_xyz = ssis_df[["x_km", "y_km", "z_km"]].to_numpy(dtype=np.float32)
        ssi_vi = ssis_df["v_i"].to_numpy(dtype=np.float32)
        r_max_sq = float(r_max_km) ** 2

        for start in range(0, n_sites, chunk_size):
            end = min(start + chunk_size, n_sites)
            diff = site_xyz[start:end, None, :] - ssi_xyz[None, :, :]
            dist_sq = np.einsum("ijk,ijk->ij", diff, diff)

            in_range = dist_sq <= r_max_sq
            dists = np.sqrt(np.where(in_range, dist_sq, 0.0))
            weights = np.where(in_range, ssi_vi[None, :] / (1.0 + float(decay_rate) * dists), 0.0)

            reachable_counts[start:end] = np.sum(in_range, axis=1)
            raw_science[start:end] = np.sum(weights, axis=1)

    df["reachable_ssi_count"] = reachable_counts
    df["raw_science"] = raw_science

    # 3. Fit grid distributions & standardize into [0.0, 1.0]
    safety_dist = fit_distribution(df["raw_safety"].to_numpy(), method=method)
    science_dist = fit_distribution(df["raw_science"].to_numpy(), method=method)

    df["s_safety"] = np.round(apply_distribution(df["raw_safety"].to_numpy(), safety_dist), 4)
    df["s_science"] = np.round(apply_distribution(df["raw_science"].to_numpy(), science_dist), 4)

    calibration = {
        "params": {
            "max_slope_limit_deg": max_slope_limit_deg,
            "mean_slope_limit_deg": mean_slope_limit_deg,
            "roughness_limit_m": roughness_limit_m,
            "slope_weight": slope_weight,
            "r_max_km": r_max_km,
            "decay_rate": decay_rate,
            "method": method,
        },
        "safety_dist": safety_dist,
        "science_dist": science_dist,
    }

    if save_calibration:
        os.makedirs(os.path.dirname(CALIBRATION_JSON), exist_ok=True)
        with open(CALIBRATION_JSON, "w", encoding="utf-8") as f:
            json.dump(calibration, f, indent=2)

    return df, calibration


if __name__ == "__main__":
    ranked_df, cal = calculate_all_scores(method="uniform")
    print(f"Standardized {len(ranked_df):,} feasible sites and saved calibration to {CALIBRATION_JSON}.")
    print(ranked_df[["site_id", "tier", "s_safety", "s_science"]].describe().round(3).to_string())