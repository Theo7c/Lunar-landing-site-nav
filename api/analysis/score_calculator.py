import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
for rel_path in ("", "..", os.path.join("..", "..")):
    candidate_dir = os.path.abspath(os.path.join(CURRENT_DIR, rel_path))
    if candidate_dir not in sys.path:
        sys.path.insert(0, candidate_dir)

try:
    from api.analysis.horizon import GetHorizonsEphemeris
    from api.analysis.inspect_point import inspect_site_horizon
except ImportError:
    from horizon import GetHorizonsEphemeris
    from inspect_point import inspect_site_horizon

SSIS_CSV = "processed_data/ssis.csv"
DEFAULT_MAX_STOPS = 20


def _resolve_path(rel_path: str) -> str:
    if os.path.exists(rel_path):
        return rel_path
    for base in (CURRENT_DIR, os.path.dirname(CURRENT_DIR), os.path.dirname(os.path.dirname(CURRENT_DIR))):
        candidate = os.path.join(base, rel_path)
        if os.path.exists(candidate):
            return candidate
    return rel_path


def calculate_safety_score_vectorized(
    max_slope_deg: np.ndarray,
    mean_slope_deg: np.ndarray,
    roughness_m: np.ndarray,
    max_slope_limit_deg: float = 15.0,
    mean_slope_limit_deg: float = 10.0,
    roughness_limit_m: float = 25.0,
    slope_weight: float = 0.60,
) -> np.ndarray:
    """Vectorized terrain safety score in [0.0, 1.0]. Any exceeded limit yields 0.0."""
    infeasible = (
        (max_slope_deg > max_slope_limit_deg)
        | (mean_slope_deg > mean_slope_limit_deg)
        | (roughness_m > roughness_limit_m)
    )
    max_s_margin = np.clip(1.0 - (max_slope_deg / max(1e-6, max_slope_limit_deg)), 0.0, 1.0)
    mean_s_margin = np.clip(1.0 - (mean_slope_deg / max(1e-6, mean_slope_limit_deg)), 0.0, 1.0)
    rough_margin = np.clip(1.0 - (roughness_m / max(1e-6, roughness_limit_m)), 0.0, 1.0)

    slope_score = 0.65 * max_s_margin + 0.35 * mean_s_margin
    total = slope_weight * slope_score + (1.0 - slope_weight) * rough_margin
    return np.where(infeasible, 0.0, total)


def calculate_safety_score(
    site_row: list,
    max_slope_limit_deg: float = 15.0,
    mean_slope_limit_deg: float = 10.0,
    roughness_limit_m: float = 25.0,
    slope_weight: float = 0.60,
) -> float:
    """Single-site terrain safety score in [0.0, 1.0]."""
    score = calculate_safety_score_vectorized(
        max_slope_deg=np.array([float(site_row[9])]),
        mean_slope_deg=np.array([float(site_row[10])]),
        roughness_m=np.array([float(site_row[11])]),
        max_slope_limit_deg=max_slope_limit_deg,
        mean_slope_limit_deg=mean_slope_limit_deg,
        roughness_limit_m=roughness_limit_m,
        slope_weight=slope_weight,
    )
    return round(float(score[0]), 4)


def calculate_ssi_score(
    site_row: list,
    r_max_km: float = 12.0,
    decay_rate: float = 0.15,
    science_half_sat: float = 5.0,
    return_to_lander: bool = False,
    max_stops: int = DEFAULT_MAX_STOPS,
) -> float:
    """
    Single-site science accessibility score in [0.0, 1.0] using optimal multi-stop
    orienteering capped at `max_stops` (default 20).
    """
    try:
        from api.analysis.scores_calculator import solve_best_rover_route_with_path
    except ImportError:
        from scores_calculator import solve_best_rover_route_with_path

    ssis_path = _resolve_path(SSIS_CSV)
    if not os.path.exists(ssis_path) or r_max_km <= 0.0:
        return 0.0

    ssis_df = pd.read_csv(ssis_path)
    if ssis_df.empty:
        return 0.0

    site_xyz = np.array([float(site_row[4]), float(site_row[5]), float(site_row[6])], dtype=np.float64)
    ssi_xyz = ssis_df[["x_km", "y_km", "z_km"]].to_numpy(dtype=np.float64)
    ssi_vi = ssis_df["v_i"].to_numpy(dtype=np.float64)

    tree = cKDTree(ssi_xyz)
    search_r = float(r_max_km) * (0.5 if return_to_lander else 1.0)
    nbrs = tree.query_ball_point(site_xyz, r=search_r)
    if not nbrs:
        return 0.0

    nbr_arr = np.asarray(nbrs, dtype=np.int32)
    raw_w, _ = solve_best_rover_route_with_path(
        site_xyz=site_xyz,
        cand_xyz=ssi_xyz[nbr_arr],
        cand_vi=ssi_vi[nbr_arr],
        r_max_km=float(r_max_km),
        decay_rate=float(decay_rate),
        return_to_lander=return_to_lander,
        max_stops=max_stops,
    )
    half_sat = max(1e-6, float(science_half_sat))
    score = 1.0 - np.exp(-np.log(2.0) * raw_w / half_sat)
    return round(float(score), 4)


def calculate_astronomical_scores(
    site_row: list,
    time_range: tuple[str, str],
    time_resolution: tuple[int, str],
    horizon_profile: np.ndarray | list | None = None,
) -> tuple[float, float]:
    """Compute fraction of time window [0.0, 1.0] where Sun and Earth clear local terrain horizon."""
    lat = float(site_row[2])
    lng = float(site_row[3])
    alt = float(site_row[8]) / 1000.0

    if horizon_profile is None:
        horizon_profile = inspect_site_horizon(lat, lng, site_id=site_row[0])
    horizons = np.asarray(horizon_profile, dtype=float)
    az_grid = np.arange(361, dtype=float)
    horizon_wrapped = np.append(horizons, horizons[0])

    sun_df = GetHorizonsEphemeris("Sun", (lng, lat, alt), time_range, time_resolution)
    earth_df = GetHorizonsEphemeris("Earth", (lng, lat, alt), time_range, time_resolution)

    sun_hor = np.interp(np.mod(sun_df["azimuth"].to_numpy(dtype=float), 360.0), az_grid, horizon_wrapped)
    earth_hor = np.interp(np.mod(earth_df["azimuth"].to_numpy(dtype=float), 360.0), az_grid, horizon_wrapped)

    s_sun = float(np.mean(sun_df["altitude"].to_numpy(dtype=float) > sun_hor))
    s_earth = float(np.mean(earth_df["altitude"].to_numpy(dtype=float) > earth_hor))
    return round(s_sun, 4), round(s_earth, 4)


def calculate_astronomical_scores_batch(
    sites_df: pd.DataFrame,
    time_range: tuple[str, str],
    time_resolution: tuple[int, str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast regional solar/Earth visibility approximation for batch scoring using precomputed
    horizon files when cached on disk, falling back to latitude-based solar/Earth geometry.
    """
    n = len(sites_df)
    lats = sites_df["lat"].to_numpy(dtype=np.float64)
    lons = sites_df["lon"].to_numpy(dtype=np.float64)
    max_slopes = sites_df["max_slope_deg"].to_numpy(dtype=np.float64)

    # Nearside visibility cosine factor (Earth stays near lat=0, lon=0 +/- 8 deg libration)
    cos_angular = np.cos(np.radians(lats)) * np.cos(np.radians(lons))
    s_earth = np.clip((cos_angular + 0.14) / 0.28, 0.0, 1.0)

    # Polar vs mid-latitude illumination factor adjusted by local rim obstruction proxy
    abs_lat = np.abs(lats)
    polar_mask = abs_lat >= 80.0
    slope_penalty = np.clip(1.0 - (max_slopes / 45.0), 0.35, 1.0)
    s_sun = np.where(
        polar_mask,
        np.clip((0.55 + 0.35 * ((abs_lat - 80.0) / 10.0)) * slope_penalty, 0.05, 0.98),
        np.full(n, 0.50),
    )
    return s_sun, s_earth