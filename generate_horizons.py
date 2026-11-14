import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import transform
from scipy.ndimage import map_coordinates

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

R_MOON_M = 1_737_400.0
MAX_HORIZON_DIST_M = 80_000.0

SITES_CSV = "processed_data/landing_sites.csv"
HORIZONS_CSV = "processed_data/landing_site_horizons.csv"
SP_DEM_PATH = "raw_data/south_pole/LDEM_80S_80MPP_ADJ.tiff"
GL_DEM_PATH = "raw_data/ldem_64.tif"

AZIMUTH_COLS = [f"az_{deg:03d}" for deg in range(360)]
OUTPUT_COLS = ["site_id"] + AZIMUTH_COLS


def log_step(msg: str, t0: float | None = None) -> float:
    if t0 is not None:
        print(f"   [OK] {msg} ({time.perf_counter() - t0:.2f}s)", flush=True)
    else:
        print(f"   -> {msg}", flush=True)
    return time.perf_counter()


def print_progress_bar(current: int, total: int, prefix: str = "", bar_len: int = 28) -> None:
    frac = min(1.0, current / max(1, total))
    filled = int(round(bar_len * frac))
    bar = "#" * filled + "-" * (bar_len - filled)
    sys.stdout.write(f"\r   -> {prefix} [{bar}] {frac * 100:5.1f}% ({current:,}/{total:,})")
    sys.stdout.flush()
    if current >= total:
        print("", flush=True)


def compute_spherical_horizons_batch(
    lats_deg: np.ndarray,
    lons_deg: np.ndarray,
    gl_dem: np.ndarray,
    dists_m: np.ndarray,
    batch_size: int = 250,
    progress_label: str = "Ray-casting horizons",
) -> np.ndarray:
    """
    Compute 360-degree horizon angles (degrees) for N sites on equirectangular ldem_64.tif
    using exact spherical geodesics and exact spherical curvature geometry.
    Returns float32 array of shape (N, 360).
    """
    n_sites = len(lats_deg)
    nrows, ncols = gl_dem.shape
    horizons_out = np.empty((n_sites, 360), dtype=np.float32)

    az_rad = np.radians(np.arange(360, dtype=np.float64))  # (360,)
    sin_az = np.sin(az_rad)[None, :, None]  # (1, 360, 1)
    cos_az = np.cos(az_rad)[None, :, None]  # (1, 360, 1)

    delta = dists_m.astype(np.float64) / R_MOON_M  # (K,)
    sin_d = np.sin(delta)[None, None, :]  # (1, 1, K)
    cos_d = np.cos(delta)[None, None, :]  # (1, 1, K)

    # Observer elevation z0 at each site
    r0_idx = np.clip(((90.0 - lats_deg) / 180.0 * nrows).astype(int), 0, nrows - 1)
    c0_idx = np.clip(((lons_deg + 180.0) / 360.0 * ncols).astype(int), 0, ncols - 1)
    z0_all = gl_dem[r0_idx, c0_idx].astype(np.float64)

    for start in range(0, n_sites, batch_size):
        end = min(start + batch_size, n_sites)
        lat1 = np.radians(lats_deg[start:end])[:, None, None]  # (B, 1, 1)
        lon1 = np.radians(lons_deg[start:end])[:, None, None]  # (B, 1, 1)
        sin_lat1, cos_lat1 = np.sin(lat1), np.cos(lat1)

        # Destination (lat2, lon2) for (B, 360, K)
        sin_lat2 = sin_lat1 * cos_d + cos_lat1 * sin_d * cos_az
        lat2 = np.arcsin(np.clip(sin_lat2, -1.0, 1.0))
        lon2 = lon1 + np.arctan2(sin_az * sin_d * cos_lat1, cos_d - sin_lat1 * sin_lat2)

        lat2_deg = np.degrees(lat2)
        lon2_deg = ((np.degrees(lon2) + 180.0) % 360.0) - 180.0

        rows = (90.0 - lat2_deg) / 180.0 * nrows - 0.5
        cols = ((lon2_deg + 180.0) / 360.0 * ncols - 0.5) % ncols

        z1 = map_coordinates(gl_dem, [rows, cols], order=1, mode="nearest").astype(np.float64)

        r0_vec = R_MOON_M + z0_all[start:end, None, None]
        r1_vec = R_MOON_M + z1

        # Exact spherical tangent-plane elevation angle
        dz_exact = r1_vec * cos_d - r0_vec
        dx_exact = r1_vec * sin_d
        elev_deg = np.degrees(np.arctan2(dz_exact, dx_exact))

        horizons_out[start:end] = np.nan_to_num(np.nanmax(elev_deg, axis=2), nan=0.0).astype(np.float32)
        print_progress_bar(end, n_sites, prefix=progress_label)

    return horizons_out


def compute_polar_nearfield_horizons(
    lats_deg: np.ndarray,
    lons_deg: np.ndarray,
    sp_dem: np.ndarray,
    sp_transform,
    sp_crs,
    res_m: float,
    near_dists_m: np.ndarray,
    batch_size: int = 250,
) -> np.ndarray:
    """
    Compute high-resolution (80 m/px) near-field horizon angles (< 5 km) on the
    South Pole stereographic DEM. Out-of-bounds samples return -90 deg so far-field
    global rays seamlessly take over.
    """
    n_sites = len(lats_deg)
    horizons_near = np.full((n_sites, 360), -90.0, dtype=np.float32)

    xs, ys = transform("+proj=longlat +R=1737400 +no_defs", sp_crs, lons_deg.tolist(), lats_deg.tolist())
    rows0, cols0 = rasterio.transform.rowcol(sp_transform, xs, ys)
    rows0 = np.asarray(rows0, dtype=np.float64)
    cols0 = np.asarray(cols0, dtype=np.float64)

    r_clamp = np.clip(np.round(rows0).astype(int), 0, sp_dem.shape[0] - 1)
    c_clamp = np.clip(np.round(cols0).astype(int), 0, sp_dem.shape[1] - 1)
    z0_all = sp_dem[r_clamp, c_clamp].astype(np.float64)

    az_rad = np.radians(np.arange(360, dtype=np.float64))
    d_rows = -(near_dists_m[None, :] * np.cos(az_rad)[:, None]) / res_m  # (360, K_near)
    d_cols = (near_dists_m[None, :] * np.sin(az_rad)[:, None]) / res_m   # (360, K_near)

    delta = near_dists_m.astype(np.float64) / R_MOON_M
    sin_d = np.sin(delta)[None, None, :]
    cos_d = np.cos(delta)[None, None, :]

    for start in range(0, n_sites, batch_size):
        end = min(start + batch_size, n_sites)
        sample_r = rows0[start:end, None, None] + d_rows[None, :, :]
        sample_c = cols0[start:end, None, None] + d_cols[None, :, :]

        z1 = map_coordinates(sp_dem, [sample_r, sample_c], order=1, mode="constant", cval=np.nan).astype(np.float64)

        r0_vec = R_MOON_M + z0_all[start:end, None, None]
        r1_vec = R_MOON_M + z1
        elev_deg = np.degrees(np.arctan2(r1_vec * cos_d - r0_vec, r1_vec * sin_d))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            batch_max = np.nanmax(elev_deg, axis=2)

        horizons_near[start:end] = np.nan_to_num(batch_max, nan=-90.0).astype(np.float32)
        print_progress_bar(end, n_sites, prefix="South Pole near-field (<5 km)")

    return horizons_near


def main() -> None:
    t_start = time.perf_counter()
    if not os.path.exists(SITES_CSV):
        raise FileNotFoundError(f"Missing {SITES_CSV}. Run generate_landing_sites.py first.")

    print("\n[1/3] Loading candidate landing sites & global DEM...", flush=True)
    t0 = time.perf_counter()
    sites_df = pd.read_csv(SITES_CSV)
    n_total = len(sites_df)
    log_step(f"Loaded {n_total:,} sites from {SITES_CSV}", t0)

    t0 = time.perf_counter()
    with rasterio.open(GL_DEM_PATH) as src:
        gl_dem = src.read(1).astype(np.float32)
        if src.nodata is not None:
            gl_dem[gl_dem == src.nodata] = np.nan
    if np.nanmax(np.abs(gl_dem)) < 50.0:
        gl_dem *= 1000.0
    log_step(f"Loaded global DEM {gl_dem.shape[1]}x{gl_dem.shape[0]} px", t0)

    # Radial step schedule: dense near lander (0.4-5 km) + coarser to 80 km
    near_dists_global = np.arange(450.0, 5_000.0, 450.0)
    far_dists_global = np.arange(5_000.0, MAX_HORIZON_DIST_M + 1.0, 1_250.0)
    all_dists_global = np.concatenate([near_dists_global, far_dists_global])

    print("\n[2/3] Computing 360-degree horizon profiles...", flush=True)
    t0 = time.perf_counter()
    lats = sites_df["lat"].to_numpy(dtype=np.float64)
    lons = sites_df["lon"].to_numpy(dtype=np.float64)

    # Compute spherical horizons for all sites first
    horizons_deg = compute_spherical_horizons_batch(
        lats_deg=lats,
        lons_deg=lons,
        gl_dem=gl_dem,
        dists_m=all_dists_global,
        batch_size=250,
        progress_label="Global 360-deg ray-cast",
    )
    del gl_dem

    # Refine near-field (< 5 km) for South Pole sites using 80 m/px stereographic DEM if available
    sp_mask = (sites_df["tier"] == "south_pole").to_numpy()
    if np.any(sp_mask) and os.path.exists(SP_DEM_PATH):
        with rasterio.open(SP_DEM_PATH) as sp_src:
            sp_dem = sp_src.read(1).astype(np.float32)
            sp_res_m = abs(sp_src.res[0])
            sp_transform = sp_src.transform
            sp_crs = sp_src.crs
            if sp_src.nodata is not None:
                sp_dem[sp_dem == sp_src.nodata] = np.nan
        sp_dem[np.abs(sp_dem) > 20000] = np.nan

        near_dists_sp = np.arange(max(sp_res_m, 100.0), 5_000.0, 160.0)
        sp_near = compute_polar_nearfield_horizons(
            lats_deg=lats[sp_mask],
            lons_deg=lons[sp_mask],
            sp_dem=sp_dem,
            sp_transform=sp_transform,
            sp_crs=sp_crs,
            res_m=sp_res_m,
            near_dists_m=near_dists_sp,
            batch_size=250,
        )
        horizons_deg[sp_mask] = np.maximum(horizons_deg[sp_mask], sp_near)
        del sp_dem

    log_step("Completed 360-degree horizon profiles for all sites", t0)

    print("\n[3/3] Exporting horizon profiles CSV...", flush=True)
    t0 = time.perf_counter()
    horizons_rounded = np.round(horizons_deg, 2)
    out_df = pd.DataFrame(horizons_rounded, columns=AZIMUTH_COLS)
    out_df.insert(0, "site_id", sites_df["site_id"].values)

    os.makedirs(os.path.dirname(HORIZONS_CSV), exist_ok=True)
    out_df.to_csv(HORIZONS_CSV, index=False)
    log_step(f"Saved {len(out_df):,} x {len(OUTPUT_COLS)} table to {HORIZONS_CSV}", t0)
    print(f"\nDone in {time.perf_counter() - t_start:.2f}s!", flush=True)


if __name__ == "__main__":
    main()