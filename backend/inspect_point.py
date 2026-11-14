import os
import json
import uuid
import requests
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform
from rasterio.windows import Window

# Standard IAU / NASA LOLA mean lunar reference radius
R_MOON_KM = 1737.4
R_MOON_M = 1_737_400.0

# Database file paths
BASE_SITES_CSV = "processed_data/landing_sites.csv"
CUSTOM_SITES_CSV = "processed_data/user_inspected_sites.csv"
SSIS_CSV = "processed_data/ssis.csv"

# Remote NASA-hosted full-resolution GeoTIFFs (accessed via HTTP Range Requests)
NASA_SP_20M_URL = "/vsicurl/https://pgda.gsfc.nasa.gov/data/LOLA_80S_20MPP_ADJ/LDEM_80S_20MPP_ADJ.TIF"
NASA_GLOBAL_URL = "/vsicurl/https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/ldem_16.tif"
NASA_PDS_ODE_API = "https://oderest.rsl.wustl.edu/live2/"


# ==============================================================================
# HELPER FUNCTIONS: COORDINATES & UNIFIED DATABASE LOADER
# ==============================================================================

def normalize_lon(lon_deg: float) -> float:
    """Wrap longitude into [-180, +180] degrees."""
    return float(((lon_deg + 180.0) % 360.0) - 180.0)


def latlon_to_xyz(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Convert spherical (lat, lon) in degrees to 3D Moon-centered Cartesian (km)."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([
        R_MOON_KM * np.cos(lat) * np.cos(lon),
        R_MOON_KM * np.cos(lat) * np.sin(lon),
        R_MOON_KM * np.sin(lat),
    ])


def load_all_landing_sites() -> pd.DataFrame:
    """
    Load the baseline pre-computed landing grid and merge any cached
    user-inspected coordinates into a single unified DataFrame.
    """
    dfs = []
    if os.path.exists(BASE_SITES_CSV):
        dfs.append(pd.read_csv(BASE_SITES_CSV))
    if os.path.exists(CUSTOM_SITES_CSV):
        dfs.append(pd.read_csv(CUSTOM_SITES_CSV))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ==============================================================================
# 1. EXACT TERRAIN QUERY VIA REMOTE NASA GEOTIFF STREAMING (/vsicurl/)
# ==============================================================================
def fetch_exact_nasa_terrain(lat: float, lon: float) -> dict:
    """
    Streams only the tiny raster window around (lat, lon) directly from NASA's
    public servers over HTTPS to compute exact elevation, 250 m touchdown max slope,
    and local roughness at full native resolution (20 m/px at the South Pole).
    """
    is_south_pole = lat <= -80.0
    dataset_url = NASA_SP_20M_URL if is_south_pole else NASA_GLOBAL_URL

    with rasterio.open(dataset_url) as src:
        if is_south_pole:
            # Project (lon, lat) -> South Pole Stereographic meters (x, y)
            xs, ys = transform("+proj=longlat +R=1737400 +no_defs", src.crs, [lon], [lat])
            row, col = src.index(xs[0], ys[0])
            res_m = abs(src.res[0])  # 20.0 m/px
            half_win = 12            # 25x25 px = 500 m x 500 m touchdown box (~250 m radius)
            area_radius_m = 250
        else:
            # Adjust longitude if global raster uses [0, 360] bounds
            query_lon = lon + 360.0 if (src.bounds.left >= 0 and lon < 0) else lon
            row, col = src.index(query_lon, lat)
            dlat_rad = np.pi / src.height
            res_m = R_MOON_M * dlat_rad
            half_win = 2             # 5x5 regional window
            area_radius_m = 950

        # Clamp window inside raster boundaries
        r0 = max(0, row - half_win)
        c0 = max(0, col - half_win)
        win_h = min(src.height - r0, 2 * half_win + 1)
        win_w = min(src.width - c0, 2 * half_win + 1)

        # HTTP Range Request: downloads ONLY this tiny patch (a few kilobytes)
        patch = src.read(1, window=Window(c0, r0, win_w, win_h)).astype(np.float32)
        if src.nodata is not None:
            patch[patch == src.nodata] = np.nan

        # Convert kilometers to meters if reading SVS ldem_16.tif
        if np.nanmax(np.abs(patch)) < 50.0:
            patch *= 1000.0

        dx_m = res_m if is_south_pole else max(100.0, res_m * np.cos(np.radians(lat)))
        dz_dy, dz_dx = np.gradient(patch, res_m, dx_m)
        slope_patch = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

        center_elev = float(patch[patch.shape[0] // 2, patch.shape[1] // 2])
        max_slope = float(np.nanmax(slope_patch))
        mean_slope = float(np.nanmean(slope_patch))
        roughness = float(np.nanstd(patch))

        # Engineering Safety Score S_safety in [0, 1]
        s_safety = (
            0.60 * max(0.0, 1.0 - max_slope / 15.0)
            + 0.40 * max(0.0, 1.0 - min(roughness / 25.0, 1.0))
        )

        return {
            "tier": "south_pole" if is_south_pole else "global",
            "source_dataset": "NASA PGDA LOLA 20m/px" if is_south_pole else "NASA SVS LOLA Global",
            "native_resolution_m": round(res_m, 1),
            "area_radius_m": area_radius_m,
            "elevation_m": round(center_elev, 1),
            "max_slope_deg": round(max_slope, 2),
            "mean_slope_deg": round(mean_slope, 2),
            "roughness_m": round(roughness, 2),
            "feasible_touchdown": max_slope <= 15.0,
            "s_safety": round(s_safety, 4),
        }


# ==============================================================================
# 2. LIVE NASA PDS ODE REST API QUERY (OBSERVATIONAL METADATA AT POINT)
# ==============================================================================
def fetch_nasa_pds_observations(lat: float, lon: float, box_deg: float = 0.05) -> dict:
    """
    Calls NASA's live Planetary Data System (PDS) Orbital Data Explorer REST API
    to retrieve real LRO spacecraft observations covering the exact coordinate.
    """
    lon_360 = lon % 360.0
    params = {
        "target": "moon",
        "query": "product",
        "results": "m",
        "output": "JSON",
        "ihid": "LRO",
        "minlat": max(-90.0, lat - box_deg),
        "maxlat": min(90.0, lat + box_deg),
        "westlon": max(0.0, lon_360 - box_deg),
        "eastlon": min(360.0, lon_360 + box_deg),
        "limit": 5,
    }
    try:
        resp = requests.get(NASA_PDS_ODE_API, params=params, timeout=6.0)
        resp.raise_for_status()
        data = resp.json()
        products = data.get("ODEResults", {}).get("Products", {}).get("Product", [])
        if isinstance(products, dict):
            products = [products]
        return {
            "api_status": "live",
            "matching_lro_products": len(products),
            "sample_product_ids": [p.get("pdsid") for p in products[:3]],
            "instruments": list({p.get("iid") for p in products if p.get("iid")}),
        }
    except Exception as exc:
        return {"api_status": f"unavailable ({exc})", "matching_lro_products": 0}


# ==============================================================================
# 3. LIVE SCIENCE LINKER AGAINST processed_data/ssis.csv
# ==============================================================================
def evaluate_reachable_science(lat: float, lon: float, r_max_km: float = 10.0) -> dict:
    """Finds all exact SSIs within r_max_km of the clicked point and computes S_science."""
    ssis_df = pd.read_csv(SSIS_CSV)
    pt_xyz = latlon_to_xyz(lat, lon)
    ssi_xyz = ssis_df[["x_km", "y_km", "z_km"]].values

    dists_km = np.linalg.norm(ssi_xyz - pt_xyz, axis=1)
    mask = dists_km <= r_max_km

    reachable = ssis_df[mask].copy()
    reachable["distance_km"] = np.round(dists_km[mask], 2)
    reachable = reachable.sort_values("distance_km")

    if len(reachable) > 0:
        r_raw = float(np.sum(reachable["v_i"] / (1.0 + 0.15 * reachable["distance_km"])))
    else:
        r_raw = 0.0

    # Normalize against a strong polar benchmark (R_raw = 12.0 -> 1.0)
    s_science = min(1.0, r_raw / 12.0)

    return {
        "r_max_km": r_max_km,
        "reachable_ssi_count": len(reachable),
        "raw_science_sum": round(r_raw, 3),
        "s_science": round(s_science, 4),
        "nearest_ssis": reachable[["ssi_id", "name", "category", "distance_km", "v_i"]].head(5).to_dict("records"),
    }


# ==============================================================================
# 4. PERSIST VALID INSPECTED POINT INTO processed_data/user_inspected_sites.csv
# ==============================================================================
def save_point_to_custom_csv(
    lat: float,
    lon: float,
    terrain: dict,
    f_sun: float = np.nan,
    f_earth: float = np.nan,
    dedup_radius_km: float = 0.15,
) -> dict:
    """
    Appends the inspected point to processed_data/user_inspected_sites.csv if:
      1. It passes the hard touchdown slope constraint (max_slope_deg <= 15.0).
      2. No existing site in either landing_sites.csv or user_inspected_sites.csv
         is already within 150 m (0.15 km).
    """
    if not terrain["feasible_touchdown"]:
        return {
            "saved": False,
            "reason": f"Rejected: max_slope_deg ({terrain['max_slope_deg']}°) exceeds 15.0° limit.",
        }

    pt_xyz = latlon_to_xyz(lat, lon)
    all_sites_df = load_all_landing_sites()

    # Proximity deduplication across BOTH baseline grid and custom cache
    if not all_sites_df.empty:
        existing_xyz = all_sites_df[["x_km", "y_km", "z_km"]].values
        dists_km = np.linalg.norm(existing_xyz - pt_xyz, axis=1)
        min_idx = int(np.argmin(dists_km))
        if dists_km[min_idx] <= dedup_radius_km:
            nearest_id = all_sites_df.iloc[min_idx]["site_id"]
            return {
                "saved": False,
                "reason": f"Already covered by {nearest_id} ({dists_km[min_idx] * 1000:.0f} m away).",
                "existing_site_id": nearest_id,
            }

    new_site_id = f"USER_{uuid.uuid4().hex[:6].upper()}"
    new_row = pd.DataFrame([{
        "site_id": new_site_id,
        "tier": terrain["tier"],
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        "x_km": round(float(pt_xyz[0]), 3),
        "y_km": round(float(pt_xyz[1]), 3),
        "z_km": round(float(pt_xyz[2]), 3),
        "area_radius_m": terrain["area_radius_m"],
        "elevation_m": terrain["elevation_m"],
        "max_slope_deg": terrain["max_slope_deg"],
        "mean_slope_deg": terrain["mean_slope_deg"],
        "roughness_m": terrain["roughness_m"],
        "s_safety": terrain["s_safety"],
        "f_sun": f_sun,
        "f_earth": f_earth,
    }])

    try:
        os.makedirs(os.path.dirname(CUSTOM_SITES_CSV), exist_ok=True)
        file_exists = os.path.exists(CUSTOM_SITES_CSV)
        new_row.to_csv(CUSTOM_SITES_CSV, mode="a", header=not file_exists, index=False)
        return {
            "saved": True,
            "site_id": new_site_id,
            "file": CUSTOM_SITES_CSV,
            "message": f"Saved {new_site_id} to {CUSTOM_SITES_CSV}.",
        }
    except OSError as exc:
        # Graceful fallback if deployed on a read-only serverless filesystem
        return {
            "saved": False,
            "reason": f"Read-only filesystem ({exc}); returned live result without disk write.",
        }


# ==============================================================================
# 5. UNIFIED ENDPOINT FUNCTION
# ==============================================================================
def inspect_exact_lunar_point(
    lat: float,
    lon: float,
    r_max_km: float = 10.0,
    save_to_db: bool = True,
) -> dict:
    lon = normalize_lon(lon)
    terrain = fetch_exact_nasa_terrain(lat, lon)
    pds_meta = fetch_nasa_pds_observations(lat, lon)
    science = evaluate_reachable_science(lat, lon, r_max_km=r_max_km)

    db_status = {"saved": False, "reason": "save_to_db=False"}
    if save_to_db:
        db_status = save_point_to_custom_csv(lat, lon, terrain)

    return {
        "query": {"lat": round(lat, 5), "lon": round(lon, 5), "r_max_km": r_max_km},
        "database_sync": db_status,
        "terrain_safety": terrain,
        "science_accessibility": science,
        "nasa_pds_live": pds_meta,
    }


if __name__ == "__main__":
    result = inspect_exact_lunar_point(lat=-89.4521, lon=-137.4890, r_max_km=12.0, save_to_db=True)
    print(json.dumps(result, indent=2))