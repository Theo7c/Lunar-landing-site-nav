import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import transform
from scipy.ndimage import maximum_filter, uniform_filter

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

R_MOON_KM = 1737.4
R_MOON_M = 1_737_400.0
TARGET_SP_SITES = 30_000
TARGET_GL_SITES = 30_000

os.makedirs("processed_data", exist_ok=True)

CSV_COLUMNS = [
    "site_id", "tier", "lat", "lon", "x_km", "y_km", "z_km",
    "area_radius_m", "elevation_m", "max_slope_deg", "mean_slope_deg", "roughness_m",
]


def log_step(msg: str, t0: float | None = None) -> float:
    """Print a flushed ASCII status line with optional elapsed time."""
    if t0 is not None:
        print(f"   [OK] {msg} ({time.perf_counter() - t0:.2f}s)", flush=True)
    else:
        print(f"   -> {msg}", flush=True)
    return time.perf_counter()


def print_progress_bar(current: int, total: int, prefix: str = "", bar_len: int = 28) -> None:
    """ASCII terminal progress bar safe for Windows cp1252 consoles."""
    frac = min(1.0, current / max(1, total))
    filled = int(round(bar_len * frac))
    bar = "#" * filled + "-" * (bar_len - filled)
    sys.stdout.write(f"\r   -> {prefix} [{bar}] {frac * 100:5.1f}% ({current:,}/{total:,})")
    sys.stdout.flush()
    if current >= total:
        print("", flush=True)


def normalize_lon(lon_deg):
    """Wrap longitudes into [-180, +180] degrees."""
    return ((np.asarray(lon_deg, dtype=float) + 180.0) % 360.0) - 180.0


def latlon_to_xyz(lat_deg, lon_deg):
    """Convert spherical lunar (lat, lon) in degrees to 3D Cartesian coordinates (km)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = R_MOON_KM * np.cos(lat) * np.cos(lon)
    y = R_MOON_KM * np.cos(lat) * np.sin(lon)
    z = R_MOON_KM * np.sin(lat)
    return x, y, z


t_start_all = time.perf_counter()

# ==============================================================================
# 1. SOUTH POLE LANDING AREAS (80S - 90S FROM 80 m/px STEREOGRAPHIC DEM)
# ==============================================================================
print("\n[1/3] Processing South Pole DEM (80 m/px)...", flush=True)
t0 = time.perf_counter()
sp_path = "raw_data/south_pole/LDEM_80S_80MPP_ADJ.tiff"

with rasterio.open(sp_path) as src:
    sp_dem = src.read(1).astype(np.float32)
    res_m = abs(src.res[0])  # 80.0 m/px
    sp_transform = src.transform
    sp_crs = src.crs
    if src.nodata is not None:
        sp_dem[sp_dem == src.nodata] = np.nan

sp_dem[np.abs(sp_dem) > 20000] = np.nan
t1 = log_step(f"Loaded raster {sp_dem.shape[1]}x{sp_dem.shape[0]} px", t0)

dz_dy, dz_dx = np.gradient(sp_dem, res_m, res_m)
sp_slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
t1 = log_step("Computed South Pole slope grid", t1)

# 250 m touchdown dispersion box (7x7 px) and 2 km EVA mobility zone (51x51 px)
sp_max_slope = maximum_filter(sp_slope, size=7)
sp_mean_slope = uniform_filter(sp_slope, size=7)
sp_mean_elev = uniform_filter(sp_dem, size=7)
sp_roughness = np.sqrt(
    np.maximum(0.0, uniform_filter(sp_dem**2, size=51) - uniform_filter(sp_dem, size=51) ** 2)
)
t1 = log_step("Applied 7x7 dispersion & 51x51 roughness filters", t1)

sp_step = max(1, int(round(np.sqrt((0.70 * sp_dem.shape[0] * sp_dem.shape[1]) / TARGET_SP_SITES))))
r_idx, c_idx = np.meshgrid(
    np.arange(30, sp_dem.shape[0] - 30, sp_step),
    np.arange(30, sp_dem.shape[1] - 30, sp_step),
    indexing="ij",
)
r_idx, c_idx = r_idx.ravel(), c_idx.ravel()

valid_sp = ~np.isnan(sp_max_slope[r_idx, c_idx]) & ~np.isnan(sp_mean_slope[r_idx, c_idx])
r_idx, c_idx = r_idx[valid_sp], c_idx[valid_sp]

xs, ys = rasterio.transform.xy(sp_transform, r_idx, c_idx)
raw_lons, raw_lats = transform(sp_crs, "+proj=longlat +R=1737400 +no_defs", np.array(xs), np.array(ys))
sp_lats = np.asarray(raw_lats, dtype=float)
sp_lons = normalize_lon(raw_lons)

in_polar_cap = sp_lats <= -80.0
r_idx, c_idx = r_idx[in_polar_cap], c_idx[in_polar_cap]
sp_lats, sp_lons = sp_lats[in_polar_cap], sp_lons[in_polar_cap]

sp_df = pd.DataFrame({
    "site_id": [f"SP_{i:05d}" for i in range(len(r_idx))],
    "tier": "south_pole",
    "lat": np.round(sp_lats, 5),
    "lon": np.round(sp_lons, 5),
    "area_radius_m": 250,
    "elevation_m": np.round(sp_mean_elev[r_idx, c_idx], 1),
    "max_slope_deg": np.round(sp_max_slope[r_idx, c_idx], 2),
    "mean_slope_deg": np.round(sp_mean_slope[r_idx, c_idx], 2),
    "roughness_m": np.round(sp_roughness[r_idx, c_idx], 2),
})
log_step(f"Generated {len(sp_df):,} South Pole sites (step={sp_step} px / ~{sp_step * res_m / 1000:.2f} km)", t0)

del sp_dem, dz_dy, dz_dx, sp_slope, sp_max_slope, sp_mean_slope, sp_mean_elev, sp_roughness


# ==============================================================================
# 2. GLOBAL LANDING ZONES (FAST PATCH-BASED EQUAL-AREA GRID, 80S - 85N)
# ==============================================================================
print("\n[2/3] Processing Global DEM (ldem_64.tif, ~474 m/px)...", flush=True)
t0 = time.perf_counter()
gl_path = "raw_data/ldem_64.tif"

with rasterio.open(gl_path) as src:
    gl_dem = src.read(1).astype(np.float32)
    gl_transform = src.transform
    gl_crs = src.crs
    if src.nodata is not None:
        gl_dem[gl_dem == src.nodata] = np.nan

if np.nanmax(np.abs(gl_dem)) < 50.0:
    gl_dem *= 1000.0
t1 = log_step(f"Loaded global raster {gl_dem.shape[1]}x{gl_dem.shape[0]} px ({gl_dem.nbytes / 1e6:.0f} MB)", t0)

nrows, ncols = gl_dem.shape
dlat_rad = np.pi / nrows
dlon_rad = (2.0 * np.pi) / ncols
dy_m = R_MOON_M * dlat_rad

row_lats_deg = 90.0 - (np.arange(nrows) + 0.5) * (180.0 / nrows)

# 1. Pick ~30,000 equal-area sample coordinates FIRST (avoids 265M-pixel full-array filters)
area_fraction = (np.sin(np.radians(85.0)) - np.sin(np.radians(-80.0))) / 2.0
effective_pixels = area_fraction * nrows * ncols
base_step = max(1, int(round(np.sqrt(effective_pixels / TARGET_GL_SITES))))

row_min = max(3, int(round((90.0 - 85.0) / 180.0 * nrows)))
row_max = min(nrows - 3, int(round((90.0 - (-80.0)) / 180.0 * nrows)))

gr_list, gc_list = [], []
for r in range(row_min, row_max, base_step):
    cos_lat = max(0.05, np.cos(np.radians(row_lats_deg[r])))
    col_step = max(1, int(round(base_step / cos_lat)))
    offset = (col_step // 2) if ((r // base_step) % 2 == 1) else 0
    cols = np.arange(3 + offset, ncols - 3, col_step)
    gr_list.append(np.full(len(cols), r, dtype=int))
    gc_list.append(cols)

gr_idx = np.concatenate(gr_list)
gc_idx = np.concatenate(gc_list)

# 2. Extract a 5x5 pixel neighborhood around ONLY the ~30k sampled points: shape (N, 5, 5)
dr = np.arange(-2, 3)
dc = np.arange(-2, 3)
patches = gl_dem[
    gr_idx[:, None, None] + dr[None, :, None],
    gc_idx[:, None, None] + dc[None, None, :],
]

# 3. Compute directional slopes inside the inner 3x3 dispersion box of each patch
dx_m_pts = np.maximum(100.0, R_MOON_M * np.cos(np.radians(row_lats_deg[gr_idx])) * dlon_rad)[:, None, None]
dz_dy_inner = (patches[:, 2:5, 1:4] - patches[:, 0:3, 1:4]) / (2.0 * dy_m)
dz_dx_inner = (patches[:, 1:4, 2:5] - patches[:, 1:4, 0:3]) / (2.0 * dx_m_pts)
slope_inner = np.degrees(np.arctan(np.hypot(dz_dx_inner, dz_dy_inner)))

inner_elev = patches[:, 1:4, 1:4]
gl_elev_vals = np.nanmean(inner_elev, axis=(1, 2))
gl_max_slope_vals = np.nanmax(slope_inner, axis=(1, 2))
gl_mean_slope_vals = np.nanmean(slope_inner, axis=(1, 2))
gl_roughness_vals = np.nanstd(inner_elev, axis=(1, 2))

valid_gl = ~np.isnan(gl_max_slope_vals) & ~np.isnan(gl_mean_slope_vals)
gr_idx, gc_idx = gr_idx[valid_gl], gc_idx[valid_gl]
gl_elev_vals = gl_elev_vals[valid_gl]
gl_max_slope_vals = gl_max_slope_vals[valid_gl]
gl_mean_slope_vals = gl_mean_slope_vals[valid_gl]
gl_roughness_vals = gl_roughness_vals[valid_gl]

if gl_crs is not None:
    gl_lons_raw, gl_lats_raw = rasterio.transform.xy(gl_transform, gr_idx, gc_idx)
    gl_lats = np.asarray(gl_lats_raw, dtype=float)
    gl_lons = normalize_lon(gl_lons_raw)
else:
    gl_lats = 90.0 - (gr_idx + 0.5) * (180.0 / nrows)
    gl_lons = normalize_lon(-180.0 + (gc_idx + 0.5) * (360.0 / ncols))

in_non_polar = (gl_lats > -80.0) & (gl_lats <= 85.0)
gl_df = pd.DataFrame({
    "site_id": [f"GL_{i:05d}" for i in range(int(np.sum(in_non_polar)))],
    "tier": "global",
    "lat": np.round(gl_lats[in_non_polar], 5),
    "lon": np.round(gl_lons[in_non_polar], 5),
    "area_radius_m": 710,
    "elevation_m": np.round(gl_elev_vals[in_non_polar], 1),
    "max_slope_deg": np.round(gl_max_slope_vals[in_non_polar], 2),
    "mean_slope_deg": np.round(gl_mean_slope_vals[in_non_polar], 2),
    "roughness_m": np.round(gl_roughness_vals[in_non_polar], 2),
})
del gl_dem, patches
log_step(f"Generated {len(gl_df):,} evenly spaced Global sites (~{base_step * dy_m / 1000:.1f} km spacing)", t0)

# ==============================================================================
# 3. MERGE FEATURED LANDMARKS, ADD 3D XYZ & EXPORT 12-COLUMN CSV
# ==============================================================================
print("\n[3/3] Merging tiers, computing 3D XYZ coordinates, and exporting CSV...", flush=True)
t0 = time.perf_counter()

featured_records = [
    ("ARTEMIS_SHACKLETON_RIM", "south_pole", -89.680, -130.000, 250, 1850.0, 6.2, 3.1, 14.5),
    ("ARTEMIS_CONNECTING_RIDGE", "south_pole", -89.450, -137.500, 250, 1620.0, 7.4, 3.8, 16.2),
    ("ARTEMIS_DE_GERLACHE_RIM", "south_pole", -88.710, -68.300, 250, 2110.0, 5.8, 2.9, 13.8),
    ("ARTEMIS_FAUSTINI_RIM_A", "south_pole", -87.310, 86.500, 250, 1490.0, 6.9, 3.4, 15.9),
    ("ARTEMIS_MALAPERT_MASSIF", "south_pole", -86.040, -2.500, 250, 4750.0, 8.1, 4.2, 19.0),
    ("ARTEMIS_NOBILE_RIM_1", "south_pole", -85.220, 36.800, 250, 1980.0, 5.4, 2.7, 12.9),
    ("FEAT_TRANQUILLITATIS_PIT", "global", 8.380, 33.250, 250, -1820.0, 1.8, 1.1, 9.4),
    ("FEAT_MARIUS_HILLS_BASE", "global", 14.120, -56.720, 250, -1540.0, 2.4, 1.4, 11.2),
    ("FEAT_ARISTARCHUS_ISRU", "global", 25.850, -50.800, 250, -1210.0, 3.1, 1.9, 13.5),
    ("FEAT_SCHRODINGER_FLOOR", "global", -75.200, 132.400, 250, -2640.0, 2.1, 1.2, 10.8),
    ("FEAT_MONS_RUMKER_PLAIN", "global", 40.820, -58.100, 250, -2150.0, 1.5, 0.8, 8.6),
    ("FEAT_REINER_GAMMA_EAST", "global", 7.520, -58.850, 250, -1980.0, 1.2, 0.7, 7.9),
]

featured_df = pd.DataFrame(
    featured_records,
    columns=[
        "site_id", "tier", "lat", "lon", "area_radius_m",
        "elevation_m", "max_slope_deg", "mean_slope_deg", "roughness_m",
    ],
)

sites_df = pd.concat([sp_df, gl_df, featured_df], ignore_index=True)

lx, ly, lz = latlon_to_xyz(sites_df["lat"].values, sites_df["lon"].values)
sites_df["x_km"] = np.round(lx, 3)
sites_df["y_km"] = np.round(ly, 3)
sites_df["z_km"] = np.round(lz, 3)

sites_df = sites_df[CSV_COLUMNS]
out_path = "processed_data/landing_sites.csv"
sites_df.to_csv(out_path, index=False)

log_step(f"Saved {len(sites_df):,} total sites to {out_path}", t0)
print(f"\nDone in {time.perf_counter() - t_start_all:.2f}s! Breakdown by tier:")
print(sites_df["tier"].value_counts().to_string())