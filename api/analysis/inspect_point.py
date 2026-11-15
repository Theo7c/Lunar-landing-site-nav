import atexit
import os
import time
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

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)  #[cite: 3]

R_MOON_KM = 1737.4  #[cite: 3]
R_MOON_M = 1_737_400.0  #[cite: 3]
HORIZON_MAX_DIST_M = 80_000.0  #[cite: 3]

BASE_SITES_CSV = "processed_data/landing_sites.csv"  #[cite: 3]
BASE_HORIZONS_CSV = "processed_data/landing_site_horizons.csv"
CUSTOM_SITES_CSV = "processed_data/user_inspected_sites.csv"  #[cite: 3]
CUSTOM_HORIZONS_CSV = "processed_data/user_horizon_profiles.csv"  #[cite: 3]#
LANDING_SITES_CSV = "processed_data/landing_sites.csv"

CSV_COLUMNS = [
    "site_id", "tier", "lat", "lon", "x_km", "y_km", "z_km",
    "area_radius_m", "elevation_m", "max_slope_deg", "mean_slope_deg", "roughness_m",
]  #[cite: 3]
AZIMUTH_COLS = [f"az_{deg:03d}" for deg in range(360)]  #[cite: 3]
HORIZON_CSV_COLUMNS = ["site_id"] + AZIMUTH_COLS  #[cite: 3]

SP_SOURCES = [
    "raw_data/south_pole/LDEM_80S_80MPP_ADJ.tiff",
    "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/ldem_64.tif",
]  #[cite: 3]

GLOBAL_SOURCES = [
    "raw_data/ldem_64.tif",
    "raw_data/ldem_16.tif",
    "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/ldem_64.tif",
]  #[cite: 3]

GDAL_CURL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff,.TIFF",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
}  #[cite: 3]

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_OPEN_DATASETS: dict[str, rasterio.DatasetReader] = {}  #[cite: 3]
_CACHED_BASE_SITES: tuple[pd.DataFrame, np.ndarray] | None = None  #[cite: 3]
_CACHED_BASE_HORIZONS: tuple[dict[str, int], np.ndarray] | None = None
_GDAL_ENV = rasterio.Env(**GDAL_CURL_ENV)  #[cite: 3]
_GDAL_ENV.__enter__()  #[cite: 3]


def _resolve_path(rel_path: str) -> str:
    if os.path.exists(rel_path):
        return rel_path
    for up in ("..", os.path.join("..", "..")):
        candidate = os.path.abspath(os.path.join(_CURRENT_DIR, up, rel_path))
        if os.path.exists(candidate):
            return candidate
    return rel_path


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
        pass  #[cite: 3]


atexit.register(_cleanup_datasets)  #[cite: 3]


def normalize_lon(lon_deg: float) -> float:
    return float(((lon_deg + 180.0) % 360.0) - 180.0)  #[cite: 3]


def latlon_to_xyz(lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    x = R_MOON_KM * np.cos(lat) * np.cos(lon)
    y = R_MOON_KM * np.cos(lat) * np.sin(lon)
    z = R_MOON_KM * np.sin(lat)
    return float(x), float(y), float(z)  #[cite: 3]


def resolve_valid_raster(candidates: list[str]) -> str:
    for path in candidates:
        if path.startswith("http"):
            try:
                resp = requests.get(path, headers={"Range": "bytes=0-3"}, timeout=4.0, allow_redirects=True)
                if resp.status_code in (200, 206) and resp.content[:2] in (b"II", b"MM"):
                    return f"/vsicurl/{path}"
            except requests.RequestException:
                continue  #[cite: 3]
        else:
            resolved = _resolve_path(path)
            if os.path.exists(resolved):
                return resolved
    raise FileNotFoundError(f"None of the DEM sources could be opened: {candidates}")  #[cite: 3]


def get_dataset(candidates: list[str]) -> tuple[rasterio.DatasetReader, str]:
    """Resolve and cache an open rasterio dataset handle."""
    path = resolve_valid_raster(candidates)
    if path not in _OPEN_DATASETS or _OPEN_DATASETS[path].closed:
        _OPEN_DATASETS[path] = rasterio.open(path)
    return _OPEN_DATASETS[path], path  #[cite: 3]


def load_all_landing_sites() -> pd.DataFrame:
    dfs = []
    base_path = _resolve_path(BASE_SITES_CSV)
    custom_path = _resolve_path(CUSTOM_SITES_CSV)
    if os.path.exists(base_path):
        dfs.append(pd.read_csv(base_path))
    if os.path.exists(custom_path):
        dfs.append(pd.read_csv(custom_path))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=CSV_COLUMNS)  #[cite: 1]


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
    return row, col, float(res_m), is_projected  #[cite: 3]


def _find_nearby_existing_site(x_km: float, y_km: float, z_km: float, radius_km: float) -> dict | None:
    """Check cached base sites + user sites for an existing site within `radius_km`."""
    global _CACHED_BASE_SITES
    pt_xyz = np.array([x_km, y_km, z_km], dtype=np.float64)

    base_path = _resolve_path(BASE_SITES_CSV)
    if _CACHED_BASE_SITES is None and os.path.exists(base_path):
        base_df = pd.read_csv(base_path)
        _CACHED_BASE_SITES = (base_df, base_df[["x_km", "y_km", "z_km"]].to_numpy(dtype=np.float64))  #[cite: 3]

    if _CACHED_BASE_SITES is not None:
        base_df, base_xyz = _CACHED_BASE_SITES
        if len(base_df) > 0:
            dists = np.linalg.norm(base_xyz - pt_xyz, axis=1)
            min_idx = int(np.argmin(dists))
            if dists[min_idx] <= radius_km:
                return base_df.iloc[min_idx].to_dict()  #[cite: 3]

    custom_path = _resolve_path(CUSTOM_SITES_CSV)
    if os.path.exists(custom_path):
        cust_df = pd.read_csv(custom_path)
        if not cust_df.empty:
            cust_xyz = cust_df[["x_km", "y_km", "z_km"]].to_numpy(dtype=np.float64)
            dists = np.linalg.norm(cust_xyz - pt_xyz, axis=1)
            min_idx = int(np.argmin(dists))
            if dists[min_idx] <= radius_km:
                return cust_df.iloc[min_idx].to_dict()  #[cite: 3]

    return None  #[cite: 3]


def _find_existing_horizon(site_id: str) -> list[float] | None:
    """Check user_horizon_profiles.csv and cached base horizons for `site_id`."""
    global _CACHED_BASE_HORIZONS
    if not site_id:
        return None

    # 1. Check small custom user horizons CSV first (<1 ms)
    custom_h_path = _resolve_path(CUSTOM_HORIZONS_CSV)
    if os.path.exists(custom_h_path):
        uhdf = pd.read_csv(custom_h_path)
        if not uhdf.empty and "site_id" in uhdf.columns:
            match = uhdf[uhdf["site_id"] == site_id]
            if not match.empty:
                return np.round(match.iloc[-1][AZIMUTH_COLS].to_numpy(dtype=np.float64), 2).tolist()

    # 2. Load base horizons via fast binary .npz cache (auto-rebuilt if CSV updates)
    if _CACHED_BASE_HORIZONS is None:
        base_h_path = _resolve_path(BASE_HORIZONS_CSV)
        if os.path.exists(base_h_path):
            npz_path = os.path.splitext(base_h_path)[0] + "_cache.npz"
            csv_mtime = os.path.getmtime(base_h_path)

            if os.path.exists(npz_path) and os.path.getmtime(npz_path) >= csv_mtime:
                with np.load(npz_path, allow_pickle=False) as data:
                    sids = data["site_ids"]
                    mat = data["horizons"]
                _CACHED_BASE_HORIZONS = ({str(s): i for i, s in enumerate(sids)}, mat)
            else:
                dtype_map = {"site_id": str, **{col: np.float32 for col in AZIMUTH_COLS}}
                hdf = pd.read_csv(base_h_path, dtype=dtype_map, engine="c")
                if not hdf.empty and "site_id" in hdf.columns:
                    sids = hdf["site_id"].to_numpy(dtype=str)
                    mat = np.round(hdf[AZIMUTH_COLS].to_numpy(dtype=np.float32), 2)
                    _CACHED_BASE_HORIZONS = ({str(s): i for i, s in enumerate(sids)}, mat)
                    try:
                        np.savez(npz_path, site_ids=sids, horizons=mat)
                    except OSError:
                        pass

    if _CACHED_BASE_HORIZONS is not None:
        idx_map, mat = _CACHED_BASE_HORIZONS
        row_idx = idx_map.get(site_id)
        if row_idx is not None:
            return np.round(mat[row_idx].astype(np.float64), 2).tolist()

    return None


# ==============================================================================
# FUNCTION 1: FAST TERRAIN PATCH INSPECTION
# ==============================================================================
def inspect_site_terrain(
    lat: float,
    lon: float,
    area_radius_m: float | None = None,
    dedup_radius_km: float = 0.15,
    save: bool = True,
) -> list:
    """
    Checks existing landing_sites.csv and user_inspected_sites.csv first (when using
    default radius or matching cached radius). Otherwise extracts a touchdown patch
    sized to `area_radius_m` from the DEM and returns the 12-element site row list.
    """
    t0 = time.perf_counter()
    lon = normalize_lon(lon)
    x_km, y_km, z_km = latlon_to_xyz(lat, lon)

    # 1. Check existing CSV datasets BEFORE opening/reading any DEM raster
    existing = _find_nearby_existing_site(x_km, y_km, z_km, dedup_radius_km)
    csv_ms = (time.perf_counter() - t0) * 1000.0
    if existing is not None:
        cached_radius = float(existing["area_radius_m"])
        if area_radius_m is None or abs(cached_radius - float(area_radius_m)) < 1.0:
            print(f"[inspect_site_terrain] Cache HIT ({existing['site_id']}) in {csv_ms:.2f} ms")
            return [existing[col_name] for col_name in CSV_COLUMNS]

    # 2. Compute from DEM using the requested area_radius_m
    t_dem = time.perf_counter()
    is_south_pole = lat <= -80.0
    tier = "south_pole" if is_south_pole else "global"
    src, dataset_path = get_dataset(SP_SOURCES if is_south_pole else GLOBAL_SOURCES)

    row, col, res_m, is_projected = _locate_pixel(src, lat, lon, is_south_pole)
    effective_radius_m = (
        float(area_radius_m)
        if area_radius_m is not None
        else (250.0 if is_projected else float(round(1.5 * res_m)))
    )
    inner_half = max(1, int(round(effective_radius_m / res_m)))

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
    dem_ms = (time.perf_counter() - t_dem) * 1000.0

    if np.isnan(elevation_m) or np.isnan(max_slope_deg):
        return []

    site_id = str(existing["site_id"]) if existing is not None else f"USER_{uuid.uuid4().hex[:6].upper()}"
    row_dict = {
        "site_id": site_id,
        "tier": tier,
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        "x_km": round(x_km, 3),
        "y_km": round(y_km, 3),
        "z_km": round(z_km, 3),
        "area_radius_m": int(round(effective_radius_m)),
        "elevation_m": round(elevation_m, 1),
        "max_slope_deg": round(max_slope_deg, 2),
        "mean_slope_deg": round(mean_slope_deg, 2),
        "roughness_m": round(roughness_m, 2),
    }

    # Only persist default-radius brand-new sites so custom radii don't pollute base caches
    if save and existing is None and area_radius_m is None:
        try:
            custom_path = _resolve_path(CUSTOM_SITES_CSV)
            os.makedirs(os.path.dirname(custom_path) or ".", exist_ok=True)
            file_exists = os.path.exists(custom_path)
            pd.DataFrame([row_dict])[CSV_COLUMNS].to_csv(
                custom_path, mode="a", header=not file_exists, index=False
            )
        except OSError:
            pass

    total_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"[inspect_site_terrain] DEM eval ({lat:.4f}, {lon:.4f}, r={effective_radius_m:.0f}m) src='{dataset_path}' -> "
        f"csv={csv_ms:.1f}ms | dem_patch={dem_ms:.1f}ms | total={total_ms:.1f}ms"
    )
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
    Checks landing_site_horizons.csv (via vectorized .npz cache) and
    user_horizon_profiles.csv first. If not already saved, computes the 360-degree
    horizon elevation profile out to 80 km around (lat, lon), appends to
    user_horizon_profiles.csv, and returns the 360-element list.
    """
    t0 = time.perf_counter()
    lon = normalize_lon(lon)  #[cite: 3]

    # 1. Resolve site_id and check existing horizon caches BEFORE opening DEM
    resolved_site_id = site_id
    if resolved_site_id is None:
        x_km, y_km, z_km = latlon_to_xyz(lat, lon)
        existing = _find_nearby_existing_site(x_km, y_km, z_km, dedup_radius_km)
        if existing is not None:
            resolved_site_id = str(existing["site_id"])

    if resolved_site_id is not None:
        cached_horizon = _find_existing_horizon(resolved_site_id)
        if cached_horizon is not None:
            print(f"[inspect_site_horizon] Cache HIT ({resolved_site_id}) in {(time.perf_counter() - t0) * 1000.0:.2f} ms")
            return cached_horizon
    csv_check_ms = (time.perf_counter() - t0) * 1000.0

    # 2. DEM open & +/- 80 km window read
    t_dem = time.perf_counter()
    is_south_pole = lat <= -80.0
    src, dataset_path = get_dataset(SP_SOURCES if is_south_pole else GLOBAL_SOURCES)  #[cite: 3]

    row, col, res_m, is_projected = _locate_pixel(src, lat, lon, is_south_pole)  #[cite: 3]

    if is_projected:
        horizon_half = int(np.ceil(HORIZON_MAX_DIST_M / res_m)) + 2
    else:
        cos_lat = max(0.15, np.cos(np.radians(lat)))
        horizon_half = int(np.ceil(HORIZON_MAX_DIST_M / (res_m * cos_lat))) + 2  #[cite: 3]

    r0 = max(0, row - horizon_half)
    c0 = max(0, col - horizon_half)
    r1 = min(src.height, row + horizon_half + 1)
    c1 = min(src.width, col + horizon_half + 1)

    dem_win = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)
    if src.nodata is not None:
        dem_win[dem_win == src.nodata] = np.nan
    if np.nanmax(np.abs(dem_win)) < 50.0:
        dem_win *= 1000.0  #[cite: 3]
    dem_read_ms = (time.perf_counter() - t_dem) * 1000.0

    center_r = float(row - r0)
    center_c = float(col - c0)
    z0 = float(dem_win[int(round(center_r)), int(round(center_c))])
    if np.isnan(z0):
        return []  #[cite: 3]

    # 3. 360-degree ray-casting & elevation angles
    t_ray = time.perf_counter()
    az_rad = np.radians(np.arange(360, dtype=np.float32))
    dists_m = np.concatenate([
        np.arange(max(res_m, 150.0), 5_000.0, 250.0),
        np.arange(5_000.0, HORIZON_MAX_DIST_M + 1.0, 1_250.0),
    ]).astype(np.float32)  #[cite: 3]

    delta = dists_m / np.float32(R_MOON_M)
    sin_d = np.sin(delta)[None, :]
    cos_d = np.cos(delta)[None, :]  #[cite: 3]

    if is_projected:
        sample_r = center_r - (dists_m[None, :] * np.cos(az_rad)[:, None]) / np.float32(res_m)
        sample_c = center_c + (dists_m[None, :] * np.sin(az_rad)[:, None]) / np.float32(res_m)  #[cite: 3]
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
        sample_c = global_c - c0  #[cite: 3]

    z1 = map_coordinates(
        dem_win,
        [sample_r, sample_c],
        order=1,
        prefilter=False,
        mode="constant",
        cval=np.nan,
    )  #[cite: 3]

    r0_val = np.float32(R_MOON_M + z0)
    r1_val = np.float32(R_MOON_M) + z1
    elev_deg = np.degrees(np.arctan2(r1_val * cos_d - r0_val, r1_val * sin_d))  #[cite: 3]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        horizon_deg = np.nan_to_num(np.nanmax(elev_deg, axis=1), nan=0.0)  #[cite: 3]

    horizon_list = [round(float(val), 2) for val in horizon_deg]  #[cite: 3]
    raycast_ms = (time.perf_counter() - t_ray) * 1000.0

    if save and resolved_site_id:
        try:
            custom_h_path = _resolve_path(CUSTOM_HORIZONS_CSV)
            os.makedirs(os.path.dirname(custom_h_path) or ".", exist_ok=True)
            file_exists = os.path.exists(custom_h_path)
            pd.DataFrame([[resolved_site_id] + horizon_list], columns=HORIZON_CSV_COLUMNS).to_csv(
                custom_h_path, mode="a", header=not file_exists, index=False
            )  #[cite: 3]
        except OSError:
            pass

    total_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"[inspect_site_horizon] Cache MISS ({lat:.4f}, {lon:.4f}) src='{dataset_path}' -> "
        f"csv={csv_check_ms:.1f}ms | dem_read={dem_read_ms:.1f}ms | raycast={raycast_ms:.1f}ms | total={total_ms:.1f}ms"
    )
    return horizon_list  #[cite: 3]


# ==============================================================================
# COMBINED WRAPPER (Backward-Compatible)
# ==============================================================================
def main(lat: float, lon: float, dedup_radius_km: float = 0.15) -> tuple[list, list[float]]:
    """
    Convenience wrapper calling both `inspect_site_terrain` and `inspect_site_horizon`.
    Returns (site_row_12, horizon_list_360).
    """
    site_row = inspect_site_terrain(lat=lat, lon=lon, dedup_radius_km=dedup_radius_km, save=True)  #[cite: 3]
    if not site_row:
        return [], []
    horizons = inspect_site_horizon(
        lat=lat, lon=lon, site_id=site_row[0], dedup_radius_km=dedup_radius_km, save=True
    )  #[cite: 3]
    return site_row, horizons  #[cite: 3]