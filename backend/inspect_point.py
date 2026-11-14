import os
import uuid
import requests
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform
from rasterio.windows import Window

R_MOON_KM = 1737.4
R_MOON_M = 1_737_400.0

BASE_SITES_CSV = "processed_data/landing_sites.csv"
CUSTOM_SITES_CSV = "processed_data/user_inspected_sites.csv"

CSV_COLUMNS = [
    "site_id", "tier", "lat", "lon", "x_km", "y_km", "z_km",
    "area_radius_m", "elevation_m", "max_slope_deg", "mean_slope_deg", "roughness_m",
]

# Candidate remote NASA URLs + local DEM fallbacks
SP_SOURCES = [
    "https://pgda.gsfc.nasa.gov/data/LOLA_80S_20MPP_ADJ/LDEM_80S_20MPP_ADJ.TIF",
    "https://pgda.gsfc.nasa.gov/data/LOLA_80S_20MPP_ADJ/LDEM_80S_20MPP_ADJ.tiff",
    "raw_data/south_pole/LDEM_80S_80MPP_ADJ.tiff",
]

GLOBAL_SOURCES = [
    "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/ldem_16.tif",
    "raw_data/ldem_16.tif",
]

GDAL_CURL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff,.TIFF",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
}


def normalize_lon(lon_deg: float) -> float:
    return float(((lon_deg + 180.0) % 360.0) - 180.0)


def latlon_to_xyz(lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    x = R_MOON_KM * np.cos(lat) * np.cos(lon)
    y = R_MOON_KM * np.cos(lat) * np.sin(lon)
    z = R_MOON_KM * np.sin(lat)
    return float(x), float(y), float(z)


def resolve_valid_raster(candidates: list[str]) -> str:
    """
    Pick the first working GeoTIFF source. For HTTP URLs, verify the server
    returns TIFF magic bytes (not an HTML 404/login page) via a 4-byte Range request.
    """
    for path in candidates:
        if path.startswith("http"):
            try:
                resp = requests.get(path, headers={"Range": "bytes=0-3"}, timeout=4.0, allow_redirects=True)
                if resp.status_code in (200, 206) and resp.content[:2] in (b"II", b"MM"):
                    return f"/vsicurl/{path}"
            except requests.RequestException:
                continue
        elif os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of the DEM sources could be opened: {candidates}")


def load_all_landing_sites() -> pd.DataFrame:
    dfs = []
    if os.path.exists(BASE_SITES_CSV):
        dfs.append(pd.read_csv(BASE_SITES_CSV))
    if os.path.exists(CUSTOM_SITES_CSV):
        dfs.append(pd.read_csv(CUSTOM_SITES_CSV))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=CSV_COLUMNS)


def inspect_and_save_point(
    lat: float,
    lon: float,
    dedup_radius_km: float = 0.15,
) -> dict:
    """
    Streams the exact NASA DEM window for (lat, lon), computes the 12 static
    terrain columns, and saves feasible points (max_slope_deg <= 15.0) to
    processed_data/user_inspected_sites.csv.
    """
    lon = normalize_lon(lon)
    is_south_pole = lat <= -80.0
    tier = "south_pole" if is_south_pole else "global"
    dataset_path = resolve_valid_raster(SP_SOURCES if is_south_pole else GLOBAL_SOURCES)

    with rasterio.Env(**GDAL_CURL_ENV):
        with rasterio.open(dataset_path) as src:
            res_m = abs(src.res[0])
            if is_south_pole:
                xs, ys = transform("+proj=longlat +R=1737400 +no_defs", src.crs, [lon], [lat])
                row, col = src.index(xs[0], ys[0])
                # Scale window size dynamically to cover a ~500 m box (25x25 at 20m, 7x7 at 80m)
                half_win = max(3, int(round(250.0 / res_m)))
                area_radius_m = 250
            else:
                query_lon = lon + 360.0 if (src.bounds.left >= 0 and lon < 0) else lon
                row, col = src.index(query_lon, lat)
                dlat_rad = np.pi / src.height
                res_m = R_MOON_M * dlat_rad
                half_win = 2
                area_radius_m = 950

            r0 = max(0, row - half_win)
            c0 = max(0, col - half_win)
            win_h = min(src.height - r0, 2 * half_win + 1)
            win_w = min(src.width - c0, 2 * half_win + 1)

            patch = src.read(1, window=Window(c0, r0, win_w, win_h)).astype(np.float32)
            if src.nodata is not None:
                patch[patch == src.nodata] = np.nan

            if np.nanmax(np.abs(patch)) < 50.0:
                patch *= 1000.0

            dx_m = res_m if is_south_pole else max(100.0, res_m * np.cos(np.radians(lat)))
            dz_dy, dz_dx = np.gradient(patch, res_m, dx_m)
            slope_patch = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

            elevation_m = float(np.nanmean(patch))
            max_slope_deg = float(np.nanmax(slope_patch))
            mean_slope_deg = float(np.nanmean(slope_patch))
            roughness_m = float(np.nanstd(patch))

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

    if max_slope_deg > 15.0:
        return {
            "saved": False,
            "source": dataset_path,
            "reason": f"Rejected: max_slope_deg ({row_dict['max_slope_deg']}°) exceeds 15.0° limit.",
            "site": row_dict,
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
            }

    new_df = pd.DataFrame([row_dict])[CSV_COLUMNS]
    try:
        os.makedirs(os.path.dirname(CUSTOM_SITES_CSV), exist_ok=True)
        file_exists = os.path.exists(CUSTOM_SITES_CSV)
        new_df.to_csv(CUSTOM_SITES_CSV, mode="a", header=not file_exists, index=False)
        return {
            "saved": True,
            "source": dataset_path,
            "file": CUSTOM_SITES_CSV,
            "site": row_dict,
        }
    except OSError as exc:
        return {
            "saved": False,
            "source": dataset_path,
            "reason": f"Read-only filesystem ({exc}); returned row without disk write.",
            "site": row_dict,
        }