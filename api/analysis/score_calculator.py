import inspect
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
    from api.analysis.scores_calculator import SSIS_CSV, solve_best_rover_route
except ImportError:
    from scores_calculator import SSIS_CSV, solve_best_rover_route

try:
    from api.analysis.horizon import GetHorizonsEphemeris
except ImportError:
    from horizon import GetHorizonsEphemeris

try:
    from api.analysis.AstronomicalScore import FindBodyHorizonAngles, CalculateScore
except ImportError:
    try:
        from AstronomicalScore import FindBodyHorizonAngles, CalculateScore
    except ImportError:
        from AstronomicalScore import FindBodyHorizonAngles, CalculateScore

try:
    from api.analysis.inspect_point import inspect_site_horizon
except ImportError:
    try:
        from inspect_point import inspect_site_horizon
    except ImportError:
        inspect_site_horizon = None


# 12-column row list order:
# [0] site_id, [1] tier, [2] lat, [3] lon, [4] x_km, [5] y_km, [6] z_km,
# [7] area_radius_m, [8] elevation_m, [9] max_slope_deg, [10] mean_slope_deg, [11] roughness_m

_CACHED_SSI_TREE: tuple[cKDTree, np.ndarray, np.ndarray] | None = None
_HORIZON_CACHE: dict[tuple[float, float], np.ndarray] = {}
_EPHEMERIS_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


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


def get_site_horizon_profile(lat: float, lon: float) -> np.ndarray:
    """Retrieve and cache the 360-degree terrain horizon profile for (lat, lon)."""
    cache_key = (round(float(lat), 6), round(float(lon), 6))
    if cache_key in _HORIZON_CACHE:
        return _HORIZON_CACHE[cache_key]

    if inspect_site_horizon is None:
        raise ImportError(
            "Could not import `inspect_site_horizon` from `inspect_point.py`."
        )

    sig = inspect.signature(inspect_site_horizon)
    params = sig.parameters
    if "lat" in params and "lon" in params:
        result = inspect_site_horizon(lat=float(lat), lon=float(lon))
    else:
        result = inspect_site_horizon(float(lat), float(lon))

    horizon_arr = None
    if isinstance(result, np.ndarray) and result.shape == (360,):
        horizon_arr = result
    elif isinstance(result, (list, tuple)):
        if len(result) == 360 and not isinstance(result[0], (list, tuple, dict, np.ndarray)):
            horizon_arr = np.asarray(result, dtype=np.float64)
        else:
            for item in result:
                if isinstance(item, np.ndarray) and item.shape == (360,):
                    horizon_arr = item
                    break
                if isinstance(item, (list, tuple)) and len(item) == 360:
                    horizon_arr = np.asarray(item, dtype=np.float64)
                    break
    elif isinstance(result, dict):
        for key in ("horizonAlt_Az", "horizon_profile", "horizon", "horizon_angles", "elevations", "profile"):
            if key in result:
                horizon_arr = np.asarray(result[key], dtype=np.float64)
                break
        if horizon_arr is None:
            for val in result.values():
                if isinstance(val, (list, tuple, np.ndarray)):
                    arr = np.asarray(val)
                    if arr.shape == (360,):
                        horizon_arr = arr.astype(np.float64)
                        break

    if horizon_arr is None or horizon_arr.shape != (360,):
        raise ValueError(f"Could not extract 360-element horizon profile for ({lat}, {lon}).")

    horizon_arr = np.asarray(horizon_arr, dtype=np.float64)
    _HORIZON_CACHE[cache_key] = horizon_arr
    return horizon_arr


def _get_clean_ephemeris(
    target: str,
    lon: float,
    lat: float,
    alt_km: float,
    time_range: tuple[str, str],
    time_resolution: tuple[int, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Query JPL Horizons ephemeris, strip non-numeric footer rows, and cache."""
    cache_key = (
        target,
        round(float(lon), 4),
        round(float(lat), 4),
        round(float(alt_km), 3),
        time_range,
        time_resolution,
    )
    if cache_key in _EPHEMERIS_CACHE:
        return _EPHEMERIS_CACHE[cache_key]

    coords = (float(lon), float(lat), float(alt_km))
    eph_data = GetHorizonsEphemeris(target, coords, time_range, time_resolution)

    if eph_data is None or len(eph_data) == 0:
        empty = (np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64))
        _EPHEMERIS_CACHE[cache_key] = empty
        return empty

    az_series = pd.to_numeric(eph_data["azimuth"], errors="coerce")
    alt_series = pd.to_numeric(eph_data["altitude"], errors="coerce")
    valid_mask = az_series.notna() & alt_series.notna()

    body_az = az_series[valid_mask].to_numpy(dtype=np.float64) % 360.0
    body_alt = alt_series[valid_mask].to_numpy(dtype=np.float64)

    # Wrap azimuths >= 359.5 to 0.0 so np.round(bodyAz) stays inside 0..359
    body_az = np.where(np.round(body_az) >= 360, 0.0, body_az)

    cleaned = (body_az, body_alt)
    _EPHEMERIS_CACHE[cache_key] = cleaned
    return cleaned


def calculate_body_visibility_score(
    site_row: list,
    target: str,
    time_range: tuple[str, str] = ("2027-01-01", "2027-01-30"),
    time_resolution: tuple[int, str] = (1, "hour"),
    horizon_profile: np.ndarray | None = None,
) -> float:
    """
    Compute astronomical visibility score in [0.0, 1.0] for a target body ('Sun' or 'Earth')
    using the site's 360-degree terrain horizon profile and JPL Horizons ephemeris.
    """
    if not site_row or len(site_row) < 9:
        return 0.0

    lat = float(site_row[2])
    lon = float(site_row[3])
    alt_km = float(site_row[8]) / 1000.0

    horizon_alt_az = (
        np.asarray(horizon_profile, dtype=np.float64)
        if horizon_profile is not None
        else get_site_horizon_profile(lat, lon)
    )

    body_az, body_alt = _get_clean_ephemeris(
        target=target,
        lon=lon,
        lat=lat,
        alt_km=alt_km,
        time_range=time_range,
        time_resolution=time_resolution,
    )
    if body_az.size == 0:
        return 0.0

    angles = FindBodyHorizonAngles(horizon_alt_az, (body_az, body_alt))
    if angles.size == 0:
        return 0.0

    score = float(np.sum(angles > 0) / angles.size)
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def calculate_astronomical_scores(
    site_row: list,
    time_range: tuple[str, str] = ("2027-01-01", "2027-01-30"),
    time_resolution: tuple[int, str] = (1, "hour"),
    horizon_profile: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return (s_sun, s_earth) visibility scores in [0.0, 1.0] for a single site row."""
    if not site_row or len(site_row) < 9:
        return 0.0, 0.0

    lat = float(site_row[2])
    lon = float(site_row[3])
    h_prof = (
        np.asarray(horizon_profile, dtype=np.float64)
        if horizon_profile is not None
        else get_site_horizon_profile(lat, lon)
    )

    s_sun = calculate_body_visibility_score(
        site_row=site_row,
        target="Sun",
        time_range=time_range,
        time_resolution=time_resolution,
        horizon_profile=h_prof,
    )
    s_earth = calculate_body_visibility_score(
        site_row=site_row,
        target="Earth",
        time_range=time_range,
        time_resolution=time_resolution,
        horizon_profile=h_prof,
    )
    return s_sun, s_earth


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


def score_calculator(
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
    time_range: tuple[str, str] = ("2027-01-01", "2027-01-30"),
    time_resolution: tuple[int, str] = (1, "hour"),
    horizon_profile: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    """
    Main single-site scoring function.
    Returns (s_safety, s_science, s_sun, s_earth), each in [0.0, 1.0], for a 12-element site row.
    """
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
    s_sun, s_earth = calculate_astronomical_scores(
        site_row=site_row,
        time_range=time_range,
        time_resolution=time_resolution,
        horizon_profile=horizon_profile,
    )
    return (s_safety, s_science, s_sun, s_earth)


