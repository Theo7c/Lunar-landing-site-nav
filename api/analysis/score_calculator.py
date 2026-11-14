import os
import json
import numpy as np
import pandas as pd

from api.analysis.scores_calculator import (
    SSIS_CSV,
    CALIBRATION_JSON,
    compute_raw_safety,
    apply_distribution,
    calculate_all_scores,
)

# 12-column row list order:
# [0] site_id, [1] tier, [2] lat, [3] lon, [4] x_km, [5] y_km, [6] z_km,
# [7] area_radius_m, [8] elevation_m, [9] max_slope_deg, [10] mean_slope_deg, [11] roughness_m

_CACHED_SSIS: tuple[np.ndarray, np.ndarray] | None = None
_CACHED_CALIBRATION: dict | None = None


def _get_ssi_arrays(ssis_df: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
    global _CACHED_SSIS
    if ssis_df is not None:
        if ssis_df.empty:
            return np.empty((0, 3)), np.empty((0,))
        return ssis_df[["x_km", "y_km", "z_km"]].to_numpy(dtype=np.float64), ssis_df["v_i"].to_numpy(dtype=np.float64)

    if _CACHED_SSIS is None:
        if not os.path.exists(SSIS_CSV):
            return np.empty((0, 3)), np.empty((0,))
        df = pd.read_csv(SSIS_CSV)
        _CACHED_SSIS = (
            df[["x_km", "y_km", "z_km"]].to_numpy(dtype=np.float64),
            df["v_i"].to_numpy(dtype=np.float64),
        )
    return _CACHED_SSIS


def get_calibration(calibration: dict | None = None, reload: bool = False) -> dict:
    """Load the grid standardization parameters from disk, or compute them once if missing."""
    global _CACHED_CALIBRATION
    if calibration is not None:
        return calibration

    if _CACHED_CALIBRATION is None or reload:
        if os.path.exists(CALIBRATION_JSON):
            with open(CALIBRATION_JSON, "r", encoding="utf-8") as f:
                _CACHED_CALIBRATION = json.load(f)
        else:
            _, _CACHED_CALIBRATION = calculate_all_scores(save_calibration=True)

    return _CACHED_CALIBRATION


def calculate_safety_score(site_row: list, calibration: dict | None = None) -> float:
    """
    Take a 12-element site row list and return its grid-standardized safety score in [0.0, 1.0].
    """
    if not site_row or len(site_row) < 12:
        return 0.0

    cal = get_calibration(calibration)
    if not cal:
        return 0.0

    p = cal["params"]
    _, raw_safety = compute_raw_safety(
        max_slope_deg=np.array([float(site_row[9])]),
        mean_slope_deg=np.array([float(site_row[10])]),
        roughness_m=np.array([float(site_row[11])]),
        max_slope_limit_deg=p["max_slope_limit_deg"],
        mean_slope_limit_deg=p["mean_slope_limit_deg"],
        roughness_limit_m=p["roughness_limit_m"],
        slope_weight=p["slope_weight"],
    )

    score = float(apply_distribution(raw_safety, cal["safety_dist"])[0])
    return round(score, 4)


def calculate_ssi_score(
    site_row: list,
    calibration: dict | None = None,
    ssis_df: pd.DataFrame | None = None,
) -> float:
    """
    Take a 12-element site row list and return its grid-standardized science score in [0.0, 1.0].
    """
    if not site_row or len(site_row) < 7:
        return 0.0

    cal = get_calibration(calibration)
    if not cal:
        return 0.0

    p = cal["params"]
    r_max_km = float(p["r_max_km"])
    decay_rate = float(p["decay_rate"])
    if r_max_km <= 0.0:
        return 0.0

    ssi_xyz, ssi_vi = _get_ssi_arrays(ssis_df=ssis_df)
    if len(ssi_vi) == 0:
        return 0.0

    site_xyz = np.array([float(site_row[4]), float(site_row[5]), float(site_row[6])], dtype=np.float64)
    dists_km = np.linalg.norm(ssi_xyz - site_xyz, axis=1)

    in_range = dists_km <= r_max_km
    if not np.any(in_range):
        return 0.0

    raw_science = np.array([np.sum(ssi_vi[in_range] / (1.0 + decay_rate * dists_km[in_range]))])
    score = float(apply_distribution(raw_science, cal["science_dist"])[0])
    return round(score, 4)


if __name__ == "__main__":
    sample_row = [
        "ARTEMIS_SHACKLETON_RIM", "south_pole", -89.68, -130.0,
        -6.238, -7.434, -1737.373, 250, 1850.0, 6.2, 3.1, 14.5,
    ]
    print("Grid-standardized Safety Score :", calculate_safety_score(sample_row))
    print("Grid-standardized Science Score:", calculate_ssi_score(sample_row))