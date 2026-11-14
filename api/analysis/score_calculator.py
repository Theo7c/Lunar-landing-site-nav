import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from api.analysis.scores_calculator import SSIS_CSV, solve_best_rover_route

# 12-column row list order:
# [0] site_id, [1] tier, [2] lat, [3] lon, [4] x_km, [5] y_km, [6] z_km,
# [7] area_radius_m, [8] elevation_m, [9] max_slope_deg, [10] mean_slope_deg, [11] roughness_m

_CACHED_SSI_TREE: tuple[cKDTree, np.ndarray, np.ndarray] | None = None


def _get_ssi_tree(ssis_df: pd.DataFrame | None = None) -> tuple[cKDTree | None, np.ndarray, np.ndarray]:
    global _CACHED_SSI_TREE
    if ssis_df is not None:
        if ssis_df.empty:
            return None, np.empty((0, 3)), np.empty((0,))
        xyz = ssis_df[["x_km", "y_km", "z_km"]].to_numpy(dtype=np.float64)
        vi = ssis_df["v_i"].to_numpy(dtype=np.float64)
        return cKDTree(xyz), xyz, vi

    if _CACHED_SSI_TREE is None:
        if not os.path.exists(SSIS_CSV):
            return None, np.empty((0, 3)), np.empty((0,))
        df = pd.read_csv(SSIS_CSV)
        xyz = df[["x_km", "y_km", "z_km"]].to_numpy(dtype=np.float64)
        vi = df["v_i"].to_numpy(dtype=np.float64)
        _CACHED_SSI_TREE = (cKDTree(xyz), xyz, vi)

    return _CACHED_SSI_TREE


def calculate_safety_score(
    site_row: list,
    max_slope_limit_deg: float = 15.0,
    mean_slope_limit_deg: float = 10.0,
    roughness_limit_m: float = 25.0,
    slope_weight: float = 0.60,
) -> float:
    """Compute site-independent safety score in [0.0, 1.0] from a 12-element site row list."""
    if not site_row or len(site_row) < 12:
        return 0.0

    max_slp = float(site_row[9])
    mean_slp = float(site_row[10])
    rough = float(site_row[11])

    max_lim = max(1e-6, float(max_slope_limit_deg))
    rough_lim = max(1e-6, float(roughness_limit_m))
    if max_slp > max_lim or mean_slp > mean_slope_limit_deg:
        return 0.0

    alpha = float(np.clip(slope_weight, 0.0, 1.0))
    t_slope = float(np.clip(1.0 - (max_slp / max_lim), 0.0, 1.0))
    t_rough = float(np.clip(1.0 - (rough / rough_lim), 0.0, 1.0))

    s_safety = alpha * t_slope + (1.0 - alpha) * t_rough
    return round(float(np.clip(s_safety, 0.0, 1.0)), 4)


def calculate_ssi_score(
    site_row: list,
    ssis_df: pd.DataFrame | None = None,
    r_max_km: float = 12.0,
    decay_rate: float = 0.15,
    science_half_sat: float = 5.0,
    return_to_lander: bool = False,
) -> float:
    """
    Find all SSIs within r_max_km of the 12-element site row, solve for the
    maximum-weight route within the r_max_km traverse budget, and return s_science in [0.0, 1.0].
    """
    if not site_row or len(site_row) < 7 or r_max_km <= 0.0:
        return 0.0

    tree, ssi_xyz, ssi_vi = _get_ssi_tree(ssis_df=ssis_df)
    if tree is None or len(ssi_vi) == 0:
        return 0.0

    pt_xyz = np.array([float(site_row[4]), float(site_row[5]), float(site_row[6])], dtype=np.float64)
    idx_list = tree.query_ball_point(pt_xyz, r=float(r_max_km))
    if not idx_list:
        return 0.0

    matched_idx = np.asarray(idx_list, dtype=np.int32)
    raw_science, _ = solve_best_rover_route(
        site_xyz=pt_xyz,
        cand_xyz=ssi_xyz[matched_idx],
        cand_vi=ssi_vi[matched_idx],
        r_max_km=float(r_max_km),
        decay_rate=float(decay_rate),
        return_to_lander=return_to_lander,
    )

    half_sat = max(1e-6, float(science_half_sat))
    s_science = raw_science / (raw_science + half_sat)
    return round(float(np.clip(s_science, 0.0, 1.0)), 4)


def calculate_score(
    site_row: list,
    max_slope_limit_deg: float = 15.0,
    mean_slope_limit_deg: float = 10.0,
    roughness_limit_m: float = 25.0,
    slope_weight: float = 0.60,
    r_max_km: float = 12.0,
    decay_rate: float = 0.15,
    science_half_sat: float = 5.0,
    return_to_lander: bool = False,
    ssis_df: pd.DataFrame | None = None,
) -> tuple[float, float]:
    """Return (s_safety, s_science), both in [0.0, 1.0], for a single 12-element site row list."""
    s_safety = calculate_safety_score(
        site_row=site_row,
        max_slope_limit_deg=max_slope_limit_deg,
        mean_slope_limit_deg=mean_slope_limit_deg,
        roughness_limit_m=roughness_limit_m,
        slope_weight=slope_weight,
    )
    s_science = calculate_ssi_score(
        site_row=site_row,
        ssis_df=ssis_df,
        r_max_km=r_max_km,
        decay_rate=decay_rate,
        science_half_sat=science_half_sat,
        return_to_lander=return_to_lander,
    )
    return s_safety, s_science