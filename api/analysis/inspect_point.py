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
BASE_HORIZONS_CSV = "processed_data/landing_site_horizons.csv"
CUSTOM_SITES_CSV = "processed_data/user_inspected_sites.csv"
CUSTOM_HORIZONS_CSV = "processed_data/user_horizon_profiles.csv"

CSV_COLUMNS = [
    "site_id", "tier", "lat", "lon", "x_km", "y_km", "z_km",
    "area_radius_m", "elevation_m", "max_slope_deg", "mean_slope_deg", "roughness_m",
]
AZIMUTH_COLS = [f"az_{deg:03d}" for deg in range(360)]
HORIZON_CSV_COLUMNS = ["site_id"] + AZIMUTH_COLS

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
_CACHED_BASE_SITES: tuple[pd.DataFrame, np.ndarray] | None = None
_CACHED_BASE_HORIZONS: dict[str, list[float]] | None = None
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
    """Resolve and cache an open rasterio dataset handle."""
    path = resolve_valid_raster(candidates)
    if path not in _OPEN_DATASETS or _OPEN_DATASETS[path].closed:
        _OPEN_DATASETS[path] = rasterio.open(path)
    return _OPEN_DATASETS[path], path


def _locate_pixel(
    src: rasterio.DatasetReader,
    lat: float,
    lon: float,
    is_south_pole: bool,
) -> tuple[int, int, float, bool]:
    """Return (row, col, res_m, is_projected) for (lat, lon) on `src`."""
    is_projected = is_south_pole and (src.crs is not None)
    if is_projected:
        res_m = abs(src.res[0])
        xs, ys = transform("+proj=longlat +R=1737400 +no_defs", src.crs, [lon], [lat])
        row, col = src.index(xs[0], ys[0])
    else:
        res_m = R_MOON_M * (np.pi / src.height)
        if src.crs is not None:
            query_lon = lon + 360.0 if (src.bounds.left >= 0 and lon < 0) else lon
            row, col = src.index(query_lon, lat)
        else:
            row = int((90.0 - lat) / 180.0 * src.height)
            col = int((lon + 180.0) / 360.0 * src.width)

    row = int(np.clip(row, 0, src.height - 1))
    col = int(np.clip(col, 0, src.width - 1))
    return row, col, float(res_m), is_projected


def _find_nearby_existing_site(x_km: float, y_km: float, z_km: float, radius_km: float) -> dict | None:
    """Check cached base sites + user sites for an existing site within `radius_km`."""
    global _CACHED_BASE_SITES
    pt_xyz = np.array([x_km, y_km, z_km], dtype=np.float64)

    if _CACHED_BASE_SITES is None and os.path.exists(BASE_SITES_CSV):
        base_df = pd.read_csv(BASE_SITES_CSV)
        _CACHED_BASE_SITES = (base_df, base_df[["x_km", "y_km", "z_km"]].to_numpy(dtype=np.float64))

    if _CACHED_BASE_SITES is not None:
        base_df, base_xyz = _CACHED_BASE_SITES
        if len(base_df) > 0:
            dists = np.linalg.norm(base_xyz - pt_xyz, axis=1)
            min_idx = int(np.argmin(dists))
            if dists[min_idx] <= radius_km:
                return base_df.iloc[min_idx].to_dict()

    if os.path.exists(CUSTOM_SITES_CSV):
        cust_df = pd.read_csv(CUSTOM_SITES_CSV)
        if not cust_df.empty:
            cust_xyz = cust_df[["x_km", "y_km", "z_km"]].to_numpy(dtype=np.float64)
            dists = np.linalg.norm(cust_xyz - pt_xyz, axis=1)
            min_idx = int(np.argmin(dists))
            if dists[min_idx] <= radius_km:
                return cust_df.iloc[min_idx].to_dict()

    return None


def _find_existing_horizon(site_id: str) -> list[float] | None:
    """Check cached base horizons and user_horizon_profiles.csv for `site_id`."""
    global _CACHED_BASE_HORIZONS
    if not site_id:
        return None

    # 1. Check precomputed grid horizons (cached in memory on first lookup)
    if _CACHED_BASE_HORIZONS is None and os.path.exists(BASE_HORIZONS_CSV):
        hdf = pd.read_csv(BASE_HORIZONS_CSV)
        if not hdf.empty and "site_id" in hdf.columns:
            vals = hdf[AZIMUTH_COLS].to_numpy(dtype=float)
            _CACHED_BASE_HORIZONS = {
                str(sid): [round(float(v), 2) for v in row]
                for sid, row in zip(hdf["site_id"], vals)
            }

    if _CACHED_BASE_HORIZONS is not None and site_id in _CACHED_BASE_HORIZONS:
        return _CACHED_BASE_HORIZONS[site_id]

    # 2. Check custom user-inspected horizons CSV
    if os.path.exists(CUSTOM_HORIZONS_CSV):
        uhdf = pd.read_csv(CUSTOM_HORIZONS_CSV)
        if not uhdf.empty and "site_id" in uhdf.columns:
            match = uhdf[uhdf["site_id"] == site_id]
            if not match.empty:
                row_vals = match.iloc[-1][AZIMUTH_COLS].to_numpy(dtype=float)
                return [round(float(v), 2) for v in row_vals]

    return None


# ==============================================================================
# FUNCTION 1: FAST TERRAIN PATCH INSPECTION
# ==============================================================================
def inspect_site_terrain(
    lat: float,
    lon: float,
    dedup_radius_km: float = 0.15,
    save: bool = True,
) -> list:
    """
    Checks existing landing_sites.csv and user_inspected_sites.csv first.
    If not found within `dedup_radius_km`, extracts the touchdown patch from the DEM,
    saves to processed_data/user_inspected_sites.csv, and returns the 12-element
    site row list ordered by CSV_COLUMNS.
    """
    lon = normalize_lon(lon)
    x_km, y_km, z_km = latlon_to_xyz(lat, lon)

    # 1. Check existing CSV datasets BEFORE opening/reading any DEM raster
    existing = _find_nearby_existing_site(x_km, y_km, z_km, dedup_radius_km)
    if existing is not None:
        return [existing[col_name] for col_name in CSV_COLUMNS]

    # 2. Not in CSVs -> compute from DEM
    is_south_pole = lat <= -80.0
    tier = "south_pole" if is_south_pole else "global"
    src, _ = get_dataset(SP_SOURCES if is_south_pole else GLOBAL_SOURCES)

    row, col, res_m, is_projected = _locate_pixel(src, lat, lon, is_south_pole)
    inner_half = max(3, int(round(250.0 / res_m))) if is_projected else 1
    area_radius_m = 250 if is_projected else int(round(1.5 * res_m))

    pr0 = max(0, row - inner_half)
    pc0 = max(0, col - inner_half)
    pr1 = min(src.height, row + inner_half + 1)
    pc1 = min(src.width, col + inner_half + 1)

    patch = src.read(1, window=Window(pc0, pr0, pc1 - pc0, pr1 - pr0)).astype(np.float32)
    if src.nodata is not None:
        patch[patch == src.nodata] = np.nan
    if np.nanmax(np.abs(patch)) < 50.0:
        patch *= 1000.0

    dx_m = res_m if is_projected else max(100.0, res_m * np.cos(np.radians(lat)))
    dz_dy, dz_dx = np.gradient(patch, res_m, dx_m)
    slope_patch = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

    elevation_m = float(np.nanmean(patch))
    max_slope_deg = float(np.nanmax(slope_patch))
    mean_slope_deg = float(np.nanmean(slope_patch))
    roughness_m = float(np.nanstd(patch))

    if np.isnan(elevation_m) or np.isnan(max_slope_deg):
        return []

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

    if save:
        try:
            os.makedirs(os.path.dirname(CUSTOM_SITES_CSV) or ".", exist_ok=True)
            file_exists = os.path.exists(CUSTOM_SITES_CSV)
            pd.DataFrame([row_dict])[CSV_COLUMNS].to_csv(
                CUSTOM_SITES_CSV, mode="a", header=not file_exists, index=False
            )
        except OSError:
            pass

    return [row_dict[col_name] for col_name in CSV_COLUMNS]


# ==============================================================================
# FUNCTION 2: 360-DEGREE HORIZON PROFILE EXTRACTOR (+/- 80 km DEM Window)
# ==============================================================================
def inspect_site_horizon(
    lat: float,
    lon: float,
    site_id: str | None = None,
    dedup_radius_km: float = 0.15,
    save: bool = True,
) -> list[float]:
    """
    Checks landing_site_horizons.csv and user_horizon_profiles.csv first.
    If not already saved, computes the 360-degree horizon elevation profile out to
    80 km around (lat, lon), appends to user_horizon_profiles.csv, and returns the
    360-element list.
    """
    lon = normalize_lon(lon)

    # 1. Resolve site_id from existing sites if not explicitly provided, then check horizon CSVs
    resolved_site_id = site_id
    if resolved_site_id is None:
        x_km, y_km, z_km = latlon_to_xyz(lat, lon)
        existing = _find_nearby_existing_site(x_km, y_km, z_km, dedup_radius_km)
        if existing is not None:
            resolved_site_id = str(existing["site_id"])

    if resolved_site_id is not None:
        cached_horizon = _find_existing_horizon(resolved_site_id)
        if cached_horizon is not None:
            return cached_horizon

    # 2. Not in horizon CSVs -> compute from +/- 80 km DEM window
    is_south_pole = lat <= -80.0
    src, _ = get_dataset(SP_SOURCES if is_south_pole else GLOBAL_SOURCES)

    row, col, res_m, is_projected = _locate_pixel(src, lat, lon, is_south_pole)

    if is_projected:
        horizon_half = int(np.ceil(HORIZON_MAX_DIST_M / res_m)) + 2
    else:
        cos_lat = max(0.15, np.cos(np.radians(lat)))
        horizon_half = int(np.ceil(HORIZON_MAX_DIST_M / (res_m * cos_lat))) + 2

    r0 = max(0, row - horizon_half)
    c0 = max(0, col - horizon_half)
    r1 = min(src.height, row + horizon_half + 1)
    c1 = min(src.width, col + horizon_half + 1)

    dem_win = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)
    if src.nodata is not None:
        dem_win[dem_win == src.nodata] = np.nan
    if np.nanmax(np.abs(dem_win)) < 50.0:
        dem_win *= 1000.0

    center_r = float(row - r0)
    center_c = float(col - c0)
    z0 = float(dem_win[int(round(center_r)), int(round(center_c))])
    if np.isnan(z0):
        return []

    az_rad = np.radians(np.arange(360, dtype=np.float32))
    dists_m = np.concatenate([
        np.arange(max(res_m, 150.0), 5_000.0, 250.0),
        np.arange(5_000.0, HORIZON_MAX_DIST_M + 1.0, 1_250.0),
    ]).astype(np.float32)

    delta = dists_m / np.float32(R_MOON_M)
    sin_d = np.sin(delta)[None, :]
    cos_d = np.cos(delta)[None, :]

    if is_projected:
        sample_r = center_r - (dists_m[None, :] * np.cos(az_rad)[:, None]) / np.float32(res_m)
        sample_c = center_c + (dists_m[None, :] * np.sin(az_rad)[:, None]) / np.float32(res_m)
    else:
        sin_az = np.sin(az_rad)[:, None]
        cos_az = np.cos(az_rad)[:, None]

        lat1, lon1 = np.radians(np.float32(lat)), np.radians(np.float32(lon))
        sin_lat1, cos_lat1 = np.sin(lat1), np.cos(lat1)

        sin_lat2 = sin_lat1 * cos_d + cos_lat1 * sin_d * cos_az
        lat2_deg = np.degrees(np.arcsin(np.clip(sin_lat2, -1.0, 1.0)))
        lon2_deg = (
            (np.degrees(lon1 + np.arctan2(sin_az * sin_d * cos_lat1, cos_d - sin_lat1 * sin_lat2)) + 180.0) % 360.0
        ) - 180.0

        global_r = (90.0 - lat2_deg) / 180.0 * src.height - 0.5
        global_c = ((lon2_deg + 180.0) / 360.0 * src.width - 0.5) % src.width

        sample_r = global_r - r0
        sample_c = global_c - c0

    z1 = map_coordinates(
        dem_win,
        [sample_r, sample_c],
        order=1,
        prefilter=False,
        mode="constant",
        cval=np.nan,
    )

    r0_val = np.float32(R_MOON_M + z0)
    r1_val = np.float32(R_MOON_M) + z1
    elev_deg = np.degrees(np.arctan2(r1_val * cos_d - r0_val, r1_val * sin_d))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        horizon_deg = np.nan_to_num(np.nanmax(elev_deg, axis=1), nan=0.0)

    horizon_list = [round(float(val), 2) for val in horizon_deg]

    if save and resolved_site_id:
        try:
            os.makedirs(os.path.dirname(CUSTOM_HORIZONS_CSV) or ".", exist_ok=True)
            file_exists = os.path.exists(CUSTOM_HORIZONS_CSV)
            pd.DataFrame([[resolved_site_id] + horizon_list], columns=HORIZON_CSV_COLUMNS).to_csv(
                CUSTOM_HORIZONS_CSV, mode="a", header=not file_exists, index=False
            )
        except OSError:
            pass

    return horizon_list


# ==============================================================================
# COMBINED WRAPPER (Backward-Compatible)
# ==============================================================================
def main(lat: float, lon: float, dedup_radius_km: float = 0.15) -> tuple[list, list[float]]:
    """
    Convenience wrapper calling both `inspect_site_terrain` and `inspect_site_horizon`.
    Returns (site_row_12, horizon_list_360).
    """
    site_row = inspect_site_terrain(lat=lat, lon=lon, dedup_radius_km=dedup_radius_km, save=True)
    if not site_row:
        return [], []
    horizons = inspect_site_horizon(
        lat=lat, lon=lon, site_id=site_row[0], dedup_radius_km=dedup_radius_km, save=True
    )
    return site_row, horizons