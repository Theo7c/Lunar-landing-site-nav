import inspect
import os
import sys
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
for rel_path in ("", "..", os.path.join("..", "..")):
    candidate_dir = os.path.abspath(os.path.join(CURRENT_DIR, rel_path))
    if candidate_dir not in sys.path:
        sys.path.insert(0, candidate_dir)

try:
    from api.analysis.score_calculator import score_calculator
    from api.analysis.scores_calculator import load_sites_and_ssis
except ImportError:
    from score_calculator import score_calculator
    from scores_calculator import load_sites_and_ssis

try:
    from api.analysis.inspect_point import inspect_site_horizon
except ImportError:
    try:
        from inspect_point import inspect_site_horizon
    except ImportError:
        inspect_site_horizon = None


LUNAR_RADIUS_KM = 1737.4


def _lat_lon_alt_to_xyz_km(lat: float, lon: float, elevation_m: float = 0.0) -> tuple[float, float, float]:
    r_km = LUNAR_RADIUS_KM + (float(elevation_m) / 1000.0)
    lat_rad = np.radians(float(lat))
    lon_rad = np.radians(float(lon))
    x_km = float(r_km * np.cos(lat_rad) * np.cos(lon_rad))
    y_km = float(r_km * np.cos(lat_rad) * np.sin(lon_rad))
    z_km = float(r_km * np.sin(lat_rad))
    return x_km, y_km, z_km


def _inspect_point_data(lat: float, lon: float) -> tuple[np.ndarray | None, dict]:
    """Call inspect_site_horizon(lat, lon) if available and parse horizon array + metadata."""
    if inspect_site_horizon is None:
        return None, {}

    params = inspect.signature(inspect_site_horizon).parameters
    if "lat" in params and "lon" in params:
        result = inspect_site_horizon(lat=float(lat), lon=float(lon))
    else:
        result = inspect_site_horizon(float(lat), float(lon))

    horizon_arr = None
    metadata = {}

    if isinstance(result, np.ndarray) and result.shape == (360,):
        horizon_arr = result.astype(np.float64)
    elif isinstance(result, (list, tuple)):
        if len(result) == 360 and not isinstance(result[0], (list, tuple, dict, np.ndarray)):
            horizon_arr = np.asarray(result, dtype=np.float64)
        else:
            for item in result:
                if isinstance(item, np.ndarray) and item.shape == (360,):
                    horizon_arr = item.astype(np.float64)
                elif isinstance(item, (list, tuple)) and len(item) == 360:
                    horizon_arr = np.asarray(item, dtype=np.float64)
                elif isinstance(item, dict):
                    metadata.update(item)
    elif isinstance(result, dict):
        metadata = dict(result)
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

    return horizon_arr, metadata


def _build_site_row_for_point(
    lat: float,
    lon: float,
    elevation_m: float | None = None,
    max_slope_deg: float | None = None,
    mean_slope_deg: float | None = None,
    roughness_m: float | None = None,
) -> tuple[list, np.ndarray | None, str]:
    """
    Build the 12-element site_row required by score_calculator:
    [0] site_id, [1] tier, [2] lat, [3] lon, [4] x_km, [5] y_km, [6] z_km,
    [7] area_radius_m, [8] elevation_m, [9] max_slope_deg, [10] mean_slope_deg, [11] roughness_m
    """
    horizon_arr, meta = _inspect_point_data(lat, lon)

    # Check if this coordinate already exists in landing_sites.csv or user_inspected_sites.csv
    sites_df, _ = load_sites_and_ssis()
    matched_csv_row = None
    if not sites_df.empty and "lat" in sites_df.columns and "lon" in sites_df.columns:
        dists = np.hypot(sites_df["lat"].to_numpy(dtype=float) - float(lat), sites_df["lon"].to_numpy(dtype=float) - float(lon))
        min_idx = int(np.argmin(dists))
        if dists[min_idx] < 1e-4:
            matched_csv_row = sites_df.iloc[min_idx]

    if matched_csv_row is not None:
        source_label = f"Matched existing site `{matched_csv_row.get('site_id', 'unknown')}` from CSV"
        site_id = str(matched_csv_row.get("site_id", f"TEST_{lat:.4f}_{lon:.4f}"))
        tier = str(matched_csv_row.get("tier", "custom"))
        area_radius_m = float(matched_csv_row.get("area_radius_m", 100.0))
        elev = float(elevation_m) if elevation_m is not None else float(matched_csv_row.get("elevation_m", 0.0))
        max_slp = float(max_slope_deg) if max_slope_deg is not None else float(matched_csv_row.get("max_slope_deg", 5.0))
        mean_slp = float(mean_slope_deg) if mean_slope_deg is not None else float(matched_csv_row.get("mean_slope_deg", 2.5))
        rough = float(roughness_m) if roughness_m is not None else float(matched_csv_row.get("roughness_m", 4.0))
        x_km = float(matched_csv_row.get("x_km", _lat_lon_alt_to_xyz_km(lat, lon, elev)[0]))
        y_km = float(matched_csv_row.get("y_km", _lat_lon_alt_to_xyz_km(lat, lon, elev)[1]))
        z_km = float(matched_csv_row.get("z_km", _lat_lon_alt_to_xyz_km(lat, lon, elev)[2]))
    else:
        source_label = "Inspected custom coordinate"
        site_id = str(meta.get("site_id", f"POINT_{lat:.4f}_{lon:.4f}"))
        tier = str(meta.get("tier", "custom"))
        area_radius_m = float(meta.get("area_radius_m", 100.0))
        elev = float(elevation_m) if elevation_m is not None else float(meta.get("elevation_m", 0.0))
        max_slp = float(max_slope_deg) if max_slope_deg is not None else float(meta.get("max_slope_deg", 5.0))
        mean_slp = float(mean_slope_deg) if mean_slope_deg is not None else float(meta.get("mean_slope_deg", 2.5))
        rough = float(roughness_m) if roughness_m is not None else float(meta.get("roughness_m", 4.0))
        x_km, y_km, z_km = _lat_lon_alt_to_xyz_km(lat, lon, elev)

    site_row = [
        site_id,
        tier,
        float(lat),
        float(lon),
        x_km,
        y_km,
        z_km,
        area_radius_m,
        elev,
        max_slp,
        mean_slp,
        rough,
    ]
    return site_row, horizon_arr, source_label


def test_point_scores(
    lat: float,
    lon: float,
    time_range: tuple[str, str] = ("2027-01-01", "2027-01-30"),
    time_resolution: tuple[int, str] = (1, "hour"),
    elevation_m: float | None = None,
    max_slope_deg: float | None = None,
    mean_slope_deg: float | None = None,
    roughness_m: float | None = None,
    r_max_km: float = 12.0,
) -> tuple[float, float, float, float]:
    """
    Computes and prints all 4 scores (s_safety, s_science, s_sun, s_earth)
    for a given lunar coordinate (lat, lon).
    """
    site_row, horizon_arr, source_label = _build_site_row_for_point(
        lat=lat,
        lon=lon,
        elevation_m=elevation_m,
        max_slope_deg=max_slope_deg,
        mean_slope_deg=mean_slope_deg,
        roughness_m=roughness_m,
    )

    s_safety, s_science, s_sun, s_earth = score_calculator(
        site_row=site_row,
        r_max_km=r_max_km,
        time_range=time_range,
        time_resolution=time_resolution,
        horizon_profile=horizon_arr,
    )

    print("=" * 64)
    print(f"POINT SCORE REPORT | {source_label}")
    print(f"Coordinates : Lat = {lat} deg, Lon = {lon} deg, Elev = {site_row[8]:.1f} m")
    print(f"Terrain     : Max Slope = {site_row[9]:.2f} deg, Mean Slope = {site_row[10]:.2f} deg, Roughness = {site_row[11]:.2f} m")
    print(f"Time Window : {time_range[0]} -> {time_range[1]} (step: {time_resolution[0]} {time_resolution[1]})")
    print("-" * 64)
    print(f"  Safety Score (s_safety)   : {s_safety:.4f}")
    print(f"  Science Score (s_science) : {s_science:.4f}")
    print(f"  Sun Visibility (s_sun)    : {s_sun:.4f}  ({s_sun * 100:.2f}%)")
    print(f"  Earth Visibility (s_earth): {s_earth:.4f}  ({s_earth * 100:.2f}%)")
    print("=" * 64)

    return (s_safety, s_science, s_sun, s_earth)


if __name__ == "__main__":
    # Edit coordinates and time range here and run the file directly
    LAT = -89.54
    LON = 129.78
    TIME_RANGE = ("2027-01-01", "2027-01-30")
    TIME_RESOLUTION = (1, "hour")

    scores = test_point_scores(
        lat=LAT,
        lon=LON,
        time_range=TIME_RANGE,
        time_resolution=TIME_RESOLUTION,
    )