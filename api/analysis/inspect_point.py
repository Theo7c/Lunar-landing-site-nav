import atexit
import os
import uuid
import warnings
import requests
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import transform
from rasterio.windows import Window
from scipy.ndimage import map_coordinates

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

R_MOON_KM = 1737.4
R_MOON_M = 1_737_400.0
HORIZON_MAX_DIST_M = 80_000.0

BASE_SITES_CSV = "processed_data/landing_sites.csv"
CUSTOM_SITES_CSV = "processed_data/user_inspected_sites.csv"
CUSTOM_HORIZONS_CSV = "processed_data/user_horizon_profiles.csv"

CSV_COLUMNS = [
    "site_id", "tier", "lat", "lon", "x_km", "y_km", "z_km",
    "area_radius_m", "elevation_m", "max_slope_deg", "mean_slope_deg", "roughness_m",
]
AZIMUTH_COLS = [f"az_{deg:03d}" for deg in range(360)]
HORIZON_CSV_COLUMNS = ["site_id"] + AZIMUTH_COLS

# Local rasters first for <15 ms latency; remote NASA SVS stream as fallback
SP_SOURCES = [
    "raw_data/south_pole/LDEM_80S_80MPP_ADJ.tiff",
    "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/ldem_64.tif",
]

GLOBAL_SOURCES = [
    "raw_data/ldem_64.tif",
    "raw_data/ldem_16.tif",
    "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/ldem_64.tif",
]

GDAL_CURL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff,.TIFF",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
}

_OPEN_DATASETS: dict[str, rasterio.DatasetReader] = {}
_GDAL_ENV = rasterio.Env(**GDAL_CURL_ENV)
_GDAL_ENV.__enter__()


def _cleanup_datasets() -> None:
    for ds in _OPEN_DATASETS.values():
        try:
            if not ds.closed:
                ds.close()
        except Exception:
            pass
    _OPEN_DATASETS.clear()
    try:
        _GDAL_ENV.__exit__(None, None, None)
    except Exception:
        pass


atexit.register(_cleanup_datasets)


def normalize_lon(lon_deg: float) -> float:
    return float(((lon_deg + 180.0) % 360.0) - 180.0)


def latlon_to_xyz(lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    x = R_MOON_KM * np.cos(lat) * np.cos(lon)
    y = R_MOON_KM * np.cos(lat) * np.sin(lon)
    z = R_MOON_KM * np.sin(lat)
    return float(x), float(y), float(z)


def resolve_valid_raster(candidates: list[str]) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
        if path.startswith("http"):
            try:
                resp = requests.get(path, headers={"Range": "bytes=0-3"}, timeout=4.0, allow_redirects=True)
                if resp.status_code in (200, 206) and resp.content[:2] in (b"II", b"MM"):
                    return f"/vsicurl/{path}"
            except requests.RequestException:
                continue
    raise FileNotFoundError(f"None of the DEM sources could be opened: {candidates}")


def get_dataset(candidates: list[str]) -> tuple[rasterio.DatasetReader, str]:
    """Resolve and cache an open rasterio dataset handle so repeated clicks avoid re-opening headers."""
    path = resolve_valid_raster(candidates)
    if path not in _OPEN_DATASETS or _OPEN_DATASETS[path].closed:
        _OPEN_DATASETS[path] = rasterio.open(path)
    return _OPEN_DATASETS[path], path


def load_all_landing_sites() -> pd.DataFrame:
    dfs = []
    if os.path.exists(BASE_SITES_CSV):
        dfs.append(pd.read_csv(BASE_SITES_CSV))
    if os.path.exists(CUSTOM_SITES_CSV):
        dfs.append(pd.read_csv(CUSTOM_SITES_CSV))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=CSV_COLUMNS)


def _compute_window_horizon(
    dem_window: np.ndarray,
    center_r: float,
    center_c: float,
    lat_deg: float,
    lon_deg: float,
    res_m: float,
    is_projected: bool,
    r0: int,
    c0: int,
    full_nrows: int,
    full_ncols: int,
) -> list[float]:
    """
    Compute 360-degree horizon elevation angles (degrees) from the +/- 80 km DEM window
    using exact spherical curvature geometry. Returns a 360-element Python list.
    """
    az_rad = np.radians(np.arange(360, dtype=np.float64))
    dists_m = np.concatenate([
        np.arange(max(res_m, 150.0), 5_000.0, 250.0),
        np.arange(5_000.0, HORIZON_MAX_DIST_M + 1.0, 1_250.0),
    ]).astype(np.float64)

    z0 = float(dem_window[int(round(center_r)), int(round(center_c))])
    delta = dists_m / R_MOON_M
    sin_d = np.sin(delta)[None, :]  # (1, K)
    cos_d = np.cos(delta)[None, :]  # (1, K)

    if is_projected:
        sample_r = center_r - (dists_m[None, :] * np.cos(az_rad)[:, None]) / res_m
        sample_c = center_c + (dists_m[None, :] * np.sin(az_rad)[:, None]) / res_m
    else:
        sin_az = np.sin(az_rad)[:, None]  # (360, 1)
        cos_az = np.cos(az_rad)[:, None]  # (360, 1)

        lat1, lon1 = np.radians(lat_deg), np.radians(lon_deg)
        sin_lat1, cos_lat1 = np.sin(lat1), np.cos(lat1)

        sin_lat2 = sin_lat1 * cos_d + cos_lat1 * sin_d * cos_az
        lat2_deg = np.degrees(np.arcsin(np.clip(sin_lat2, -1.0, 1.0)))
        lon2_deg = (
            (np.degrees(lon1 + np.arctan2(sin_az * sin_d * cos_lat1, cos_d - sin_lat1 * sin_lat2)) + 180.0) % 360.0
        ) - 180.0

        global_r = (90.0 - lat2_deg) / 180.0 * full_nrows - 0.5
        global_c = ((lon2_deg + 180.0) / 360.0 * full_ncols - 0.5) % full_ncols

        sample_r = global_r - r0
        sample_c = global_c - c0

    z1 = map_coordinates(dem_window, [sample_r, sample_c], order=1, mode="constant", cval=np.nan).astype(np.float64)

    r0_val = R_MOON_M + z0
    r1_val = R_MOON_M + z1
    elev_deg = np.degrees(np.arctan2(r1_val * cos_d - r0_val, r1_val * sin_d))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        horizon_deg = np.nan_to_num(np.nanmax(elev_deg, axis=1), nan=0.0)

    return [round(float(val), 2) for val in horizon_deg]


def _append_user_horizon_csv(site_id: str, horizon_list: list[float]) -> None:
    """Append [site_id, az_000, ..., az_359] to processed_data/user_horizon_profiles.csv."""
    os.makedirs(os.path.dirname(CUSTOM_HORIZONS_CSV) or ".", exist_ok=True)
    row_values = [site_id] + horizon_list
    horizon_df = pd.DataFrame([row_values], columns=HORIZON_CSV_COLUMNS)
    file_exists = os.path.exists(CUSTOM_HORIZONS_CSV)
    horizon_df.to_csv(CUSTOM_HORIZONS_CSV, mode="a", header=not file_exists, index=False)


def inspect_and_save_point(
    lat: float,
    lon: float,
    dedup_radius_km: float = 0.15,
) -> dict:
    """
    Reads a single +/- 80 km DEM window around (lat, lon) from local DEMs (or remote
    NASA stream fallback), computes the 12 static terrain columns AND the 360-degree
    horizon elevation profile, and saves both to:
      - processed_data/user_inspected_sites.csv
      - processed_data/user_horizon_profiles.csv
    """
    lon = normalize_lon(lon)
    is_south_pole = lat <= -80.0
    tier = "south_pole" if is_south_pole else "global"
    src, dataset_path = get_dataset(SP_SOURCES if is_south_pole else GLOBAL_SOURCES)

    is_projected = is_south_pole and (src.crs is not None)

    if is_projected:
        res_m = abs(src.res[0])
        xs, ys = transform("+proj=longlat +R=1737400 +no_defs", src.crs, [lon], [lat])
        row, col = src.index(xs[0], ys[0])
        row = int(np.clip(row, 0, src.height - 1))
        col = int(np.clip(col, 0, src.width - 1))
        inner_half = max(3, int(round(250.0 / res_m)))
        horizon_half = int(np.ceil(HORIZON_MAX_DIST_M / res_m)) + 2
        area_radius_m = 250
    else:
        res_m = R_MOON_M * (np.pi / src.height)
        if src.crs is not None:
            query_lon = lon + 360.0 if (src.bounds.left >= 0 and lon < 0) else lon
            row, col = src.index(query_lon, lat)
            row = int(np.clip(row, 0, src.height - 1))
            col = int(np.clip(col, 0, src.width - 1))
        else:
            row = int(np.clip((90.0 - lat) / 180.0 * src.height, 0, src.height - 1))
            col = int(np.clip((lon + 180.0) / 360.0 * src.width, 0, src.width - 1))
        inner_half = 1
        cos_lat = max(0.15, np.cos(np.radians(lat)))
        horizon_half = int(np.ceil(HORIZON_MAX_DIST_M / (res_m * cos_lat))) + 2
        area_radius_m = int(round(1.5 * res_m))

    r0 = max(0, row - horizon_half)
    c0 = max(0, col - horizon_half)
    r1 = min(src.height, row + horizon_half + 1)
    c1 = min(src.width, col + horizon_half + 1)

    dem_win = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)
    if src.nodata is not None:
        dem_win[dem_win == src.nodata] = np.nan
    if np.nanmax(np.abs(dem_win)) < 50.0:
        dem_win *= 1000.0

    center_r = row - r0
    center_c = col - c0

    # 1. Inner touchdown dispersion patch metrics
    pr0 = max(0, center_r - inner_half)
    pc0 = max(0, center_c - inner_half)
    pr1 = min(dem_win.shape[0], center_r + inner_half + 1)
    pc1 = min(dem_win.shape[1], center_c + inner_half + 1)
    patch = dem_win[pr0:pr1, pc0:pc1]

    dx_m = res_m if is_projected else max(100.0, res_m * np.cos(np.radians(lat)))
    dz_dy, dz_dx = np.gradient(patch, res_m, dx_m)
    slope_patch = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

    elevation_m = float(np.nanmean(patch))
    max_slope_deg = float(np.nanmax(slope_patch))
    mean_slope_deg = float(np.nanmean(slope_patch))
    roughness_m = float(np.nanstd(patch))

    # 2. 360-degree horizon profile
    horizon_list = _compute_window_horizon(
        dem_window=dem_win,
        center_r=float(center_r),
        center_c=float(center_c),
        lat_deg=lat,
        lon_deg=lon,
        res_m=res_m,
        is_projected=is_projected,
        r0=r0,
        c0=c0,
        full_nrows=src.height,
        full_ncols=src.width,
    )

    if np.isnan(elevation_m) or np.isnan(max_slope_deg):
        return {
            "saved": False,
            "source": dataset_path,
            "reason": "No valid DEM elevation data (NaN/nodata) at this coordinate.",
        }

    x_km, y_km, z_km = latlon_to_xyz(lat, lon)
    new_site_id = f"USER_{uuid.uuid4().hex[:6].upper()}"

    row_dict = {
        "site_id": new_site_id,
        "tier": tier,
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        "x_km": round(x_km, 3),
        "y_km": round(y_km, 3),
        "z_km": round(z_km, 3),
        "area_radius_m": area_radius_m,
        "elevation_m": round(elevation_m, 1),
        "max_slope_deg": round(max_slope_deg, 2),
        "mean_slope_deg": round(mean_slope_deg, 2),
        "roughness_m": round(roughness_m, 2),
    }

    all_sites_df = load_all_landing_sites()
    if not all_sites_df.empty:
        pt_xyz = np.array([x_km, y_km, z_km])
        existing_xyz = all_sites_df[["x_km", "y_km", "z_km"]].values
        dists_km = np.linalg.norm(existing_xyz - pt_xyz, axis=1)
        min_idx = int(np.argmin(dists_km))
        if dists_km[min_idx] <= dedup_radius_km:
            existing_row = all_sites_df.iloc[min_idx].to_dict()
            return {
                "saved": False,
                "source": dataset_path,
                "reason": f"Already covered by {existing_row['site_id']} ({dists_km[min_idx] * 1000:.0f} m away).",
                "site": existing_row,
                "horizon_deg": horizon_list,
            }

    new_df = pd.DataFrame([row_dict])[CSV_COLUMNS]
    try:
        os.makedirs(os.path.dirname(CUSTOM_SITES_CSV) or ".", exist_ok=True)
        file_exists = os.path.exists(CUSTOM_SITES_CSV)
        new_df.to_csv(CUSTOM_SITES_CSV, mode="a", header=not file_exists, index=False)
        _append_user_horizon_csv(new_site_id, horizon_list)
        return {
            "saved": True,
            "source": dataset_path,
            "file": CUSTOM_SITES_CSV,
            "horizon_file": CUSTOM_HORIZONS_CSV,
            "site": row_dict,
            "horizon_deg": horizon_list,
        }
    except OSError as exc:
        return {
            "saved": False,
            "source": dataset_path,
            "reason": f"Read-only filesystem ({exc}); returned data without disk write.",
            "site": row_dict,
            "horizon_deg": horizon_list,
        }


def main(lat: float, lon: float, dedup_radius_km: float = 0.15) -> tuple[list, list[float]]:
    """
    Inspects (lat, lon), saves the 12-column site row to user_inspected_sites.csv and
    its 360-degree horizon profile to user_horizon_profiles.csv, and returns two lists:
      1. site_row: 12-element list ordered by CSV_COLUMNS
      2. horizon_list: 360-element list of horizon angles in degrees (0 to 359 deg)
    Returns ([], []) if no valid DEM data exists at the coordinate.
    """
    result = inspect_and_save_point(lat=lat, lon=lon, dedup_radius_km=dedup_radius_km)
    site = result.get("site")
    horizons = result.get("horizon_deg")
    if not site or horizons is None:
        return [], []
    return [site[col] for col in CSV_COLUMNS], horizons


if __name__ == "__main__":
    site_row, horizons = main(lat=-78.4521, lon=-130.4890)
    print("Site row (12):", site_row)
    print(f"Horizons ({len(horizons)}):", horizons[:10], "...")