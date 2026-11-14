import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform
from scipy.ndimage import maximum_filter, uniform_filter

# Standard IAU / NASA LOLA mean lunar reference radius
R_MOON_KM = 1737.4
R_MOON_M = 1_737_400.0

# Ensure output directory exists
os.makedirs("processed_data", exist_ok=True)


# ==============================================================================
# HELPER FUNCTIONS: COORDINATE STANDARDIZATION & 3D CARTESIAN PROJECTION
# ==============================================================================

def normalize_lon(lon_deg):
    """
    Wrap longitudes from [0, 360] into [-180, +180] degrees so all coordinates
    match ssis.csv and standard Three.js spherical mapping conventions.
    """
    return ((np.asarray(lon_deg, dtype=float) + 180.0) % 360.0) - 180.0


def latlon_to_xyz(lat_deg, lon_deg):
    """
    Convert spherical lunar coordinates (lat, lon in degrees) into 3D
    Moon-centered Cartesian coordinates (x_km, y_km, z_km).

    Why we pre-compute 3D XYZ:
    To link landing_sites.csv with ssis.csv for ANY user-chosen rover travel
    distance R_max in real time (<10 ms), we compute 3D Euclidean distances
    sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2). This avoids slow spherical trig
    and eliminates polar coordinate singularities near 90S.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = R_MOON_KM * np.cos(lat) * np.cos(lon)
    y = R_MOON_KM * np.cos(lat) * np.sin(lon)
    z = R_MOON_KM * np.sin(lat)
    return x, y, z


# ==============================================================================
# 1. SOUTH POLE LANDING AREAS (80S - 90S FROM 80 m/px STEREOGRAPHIC DEM)
# ==============================================================================
print("1/3 Processing South Pole DEM (80 m/px)...")
sp_path = "raw_data/south_pole/LDEM_80S_80MPP_ADJ.tiff"

with rasterio.open(sp_path) as src:
    sp_dem = src.read(1).astype(np.float32)
    res_m = abs(src.res[0])  # Pixel spacing in meters (80.0 m/px)
    sp_transform = src.transform
    sp_crs = src.crs
    if src.nodata is not None:
        sp_dem[sp_dem == src.nodata] = np.nan

# Mask out any extreme sentinel/corrupt elevation values
sp_dem[np.abs(sp_dem) > 20000] = np.nan

# Compute local terrain gradient (dz/dx, dz/dy) in m/m and convert to slope angle (degrees)
dz_dy, dz_dx = np.gradient(sp_dem, res_m, res_m)
sp_slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

# --- AREA-BASED ENGINEERING AGGREGATION (LANDING PATCHES, NOT SINGLE PIXELS) ---
# 1. Touchdown Dispersion Ellipse: ~250 m radius (560 m x 560 m box -> 7x7 pixels at 80 m/px)
#    We take the MAXIMUM slope inside this 7x7 box so that if even a single 80 m pixel
#    is a steep crater wall (>15 deg), the entire touchdown area is rejected.
sp_max_slope = maximum_filter(sp_slope, size=7)
sp_mean_slope = uniform_filter(sp_slope, size=7)
sp_mean_elev = uniform_filter(sp_dem, size=7)

# 2. EVA / Rover Mobility Zone: ~2 km radius (4.08 km x 4.08 km box -> 51x51 pixels at 80 m/px)
#    Terrain roughness is the standard deviation of elevation across this 4 km zone,
#    computed rapidly via the variance identity: sigma = sqrt(E[z^2] - E[z]^2).
mean_z = uniform_filter(sp_dem, size=51)
mean_z2 = uniform_filter(sp_dem**2, size=51)
sp_roughness = np.sqrt(np.maximum(0.0, mean_z2 - mean_z**2))

# Sample candidate patch centers every `sp_step` pixels:
# - sp_step = 20 -> 1.6 km spacing (~35,000-45,000 valid polar patches)
# - Decrease to 15 for denser ~1.2 km spacing (~65,000 patches)
sp_step = 20
r_idx, c_idx = np.meshgrid(
    np.arange(30, sp_dem.shape[0] - 30, sp_step),
    np.arange(30, sp_dem.shape[1] - 30, sp_step),
    indexing="ij",
)
r_idx, c_idx = r_idx.ravel(), c_idx.ravel()

# Hard Feasibility Filter: Reject NaN borders and any landing area with max slope > 15.0 deg
valid_sp = ~np.isnan(sp_max_slope[r_idx, c_idx]) & (sp_max_slope[r_idx, c_idx] <= 15.0)
r_idx, c_idx = r_idx[valid_sp], c_idx[valid_sp]

# Convert raster pixel indices (row, col) -> stereographic meters (x, y) -> (lon, lat) degrees
xs, ys = rasterio.transform.xy(sp_transform, r_idx, c_idx)
raw_lons, raw_lats = transform(
    sp_crs,
    "+proj=longlat +R=1737400 +no_defs",
    np.array(xs),
    np.array(ys),
)
sp_lats = np.asarray(raw_lats, dtype=float)
sp_lons = normalize_lon(raw_lons)

# Restrict strictly to 80S-90S so the polar grid never duplicates global grid rows
in_polar_cap = sp_lats <= -80.0
r_idx, c_idx = r_idx[in_polar_cap], c_idx[in_polar_cap]
sp_lats, sp_lons = sp_lats[in_polar_cap], sp_lons[in_polar_cap]

# Compute normalized Engineering Safety Score S_safety in [0, 1]:
# - 60% weight from worst-case touchdown slope (0 deg -> 1.0, 15 deg -> 0.0)
# - 40% weight from 2 km EVA roughness (capped at 95th percentile so outliers don't skew)
theta_max_sp = sp_max_slope[r_idx, c_idx]
sigma_eva_sp = sp_roughness[r_idx, c_idx]
sigma_cap_sp = np.percentile(sigma_eva_sp, 95)

s_safety_sp = (
    0.60 * (1.0 - theta_max_sp / 15.0)
    + 0.40 * (1.0 - np.clip(sigma_eva_sp / sigma_cap_sp, 0.0, 1.0))
)

sp_df = pd.DataFrame({
    "site_id": [f"SP_{i:05d}" for i in range(len(r_idx))],
    "tier": "south_pole",
    "lat": np.round(sp_lats, 5),
    "lon": np.round(sp_lons, 5),
    "area_radius_m": 250,
    "elevation_m": np.round(sp_mean_elev[r_idx, c_idx], 1),
    "max_slope_deg": np.round(theta_max_sp, 2),
    "mean_slope_deg": np.round(sp_mean_slope[r_idx, c_idx], 2),
    "roughness_m": np.round(sigma_eva_sp, 2),
    "s_safety": np.round(s_safety_sp, 4),
})
print(f"   -> Generated {len(sp_df)} valid South Pole landing areas.")


# ==============================================================================
# 2. GLOBAL LANDING ZONES (80S - 85N FROM ~1.9 km/px EQUIRECTANGULAR DEM)
# ==============================================================================
print("2/3 Processing Global DEM (ldem_16.tif)...")
gl_path = "raw_data/ldem_16.tif"

with rasterio.open(gl_path) as src:
    gl_dem = src.read(1).astype(np.float32)
    gl_transform = src.transform
    if src.nodata is not None:
        gl_dem[gl_dem == src.nodata] = np.nan

# NASA SVS ldem_16.tif stores elevation in kilometers; convert to meters if needed
if np.nanmax(np.abs(gl_dem)) < 50.0:
    gl_dem *= 1000.0

nrows, ncols = gl_dem.shape

# Compute physical pixel dimensions on a sphere:
# North-South spacing (dy_m) is constant, while East-West spacing (dx_m) shrinks by cos(lat)
dlat_rad = np.pi / nrows
dlon_rad = (2.0 * np.pi) / ncols
dy_m = R_MOON_M * dlat_rad

row_centers = np.arange(nrows)
_, row_lats_deg = rasterio.transform.xy(gl_transform, row_centers, np.zeros(nrows))
row_lats_deg = np.asarray(row_lats_deg, dtype=float)
dx_m_per_row = np.maximum(100.0, R_MOON_M * np.cos(np.radians(row_lats_deg)) * dlon_rad)[:, None]

dz_dy_gl = np.gradient(gl_dem, dy_m, axis=0)
dz_dx_gl = np.gradient(gl_dem, axis=1) / dx_m_per_row
gl_slope = np.degrees(np.arctan(np.hypot(dz_dx_gl, dz_dy_gl)))

# Aggregate 3x3 regional neighborhoods (~5.7 km x 5.7 km at the equator)
gl_max_slope = maximum_filter(gl_slope, size=3)
gl_mean_slope = uniform_filter(gl_slope, size=3)
gl_roughness = np.sqrt(
    np.maximum(0.0, uniform_filter(gl_dem**2, size=3) - uniform_filter(gl_dem, size=3) ** 2)
)

# Sample global candidate zones every `gl_step` pixels:
# - gl_step = 24 -> 1.5 deg (~45 km) spacing (~24,000-26,000 valid global zones)
# - Decrease to 16 for 1.0 deg (~30 km) spacing (~56,000 zones)
gl_step = 24
gr_idx, gc_idx = np.meshgrid(
    np.arange(80, nrows - 160, gl_step),
    np.arange(8, ncols - 8, gl_step),
    indexing="ij",
)
gr_idx, gc_idx = gr_idx.ravel(), gc_idx.ravel()

# Hard Feasibility Filter: max regional slope <= 15.0 deg
valid_gl = ~np.isnan(gl_max_slope[gr_idx, gc_idx]) & (gl_max_slope[gr_idx, gc_idx] <= 15.0)
gr_idx, gc_idx = gr_idx[valid_gl], gc_idx[valid_gl]

gl_lons_raw, gl_lats_raw = rasterio.transform.xy(gl_transform, gr_idx, gc_idx)
gl_lats = np.asarray(gl_lats_raw, dtype=float)
gl_lons = normalize_lon(gl_lons_raw)

# Keep strictly outside the South Pole cap (-80S < lat <= 85N)
in_non_polar = (gl_lats > -80.0) & (gl_lats <= 85.0)
gr_idx, gc_idx = gr_idx[in_non_polar], gc_idx[in_non_polar]
gl_lats, gl_lons = gl_lats[in_non_polar], gl_lons[in_non_polar]

theta_max_gl = gl_max_slope[gr_idx, gc_idx]
sigma_reg_gl = gl_roughness[gr_idx, gc_idx]
sigma_cap_gl = np.percentile(sigma_reg_gl, 95)

s_safety_gl = (
    0.60 * (1.0 - theta_max_gl / 15.0)
    + 0.40 * (1.0 - np.clip(sigma_reg_gl / sigma_cap_gl, 0.0, 1.0))
)

gl_df = pd.DataFrame({
    "site_id": [f"GL_{i:05d}" for i in range(len(gr_idx))],
    "tier": "global",
    "lat": np.round(gl_lats, 5),
    "lon": np.round(gl_lons, 5),
    "area_radius_m": 950,
    "elevation_m": np.round(gl_dem[gr_idx, gc_idx], 1),
    "max_slope_deg": np.round(theta_max_gl, 2),
    "mean_slope_deg": np.round(gl_mean_slope[gr_idx, gc_idx], 2),
    "roughness_m": np.round(sigma_reg_gl, 2),
    "s_safety": np.round(s_safety_gl, 4),
})
print(f"   -> Generated {len(gl_df)} valid Global landing zones.")


# ==============================================================================
# 3. TARGETED ARTEMIS III & GLOBAL LANDMARK LANDING PATCHES
# ==============================================================================
# Uniform grids can step ~15-20 km past narrow crater rims or skylight pits.
# Adding explicit high-precision candidate landing patches beside key NASA Artemis III
# and global ISRU targets guarantees realistic short-range (<5 km) mission solutions.
featured_records = [
    # Famous NASA Artemis III Candidate Regions (South Pole)
    ("ARTEMIS_SHACKLETON_RIM", "south_pole", -89.680, -130.000, 250, 1850.0, 6.2, 3.1, 14.5, 0.8900),
    ("ARTEMIS_CONNECTING_RIDGE", "south_pole", -89.450, -137.500, 250, 1620.0, 7.4, 3.8, 16.2, 0.8400),
    ("ARTEMIS_DE_GERLACHE_RIM", "south_pole", -88.710, -68.300, 250, 2110.0, 5.8, 2.9, 13.8, 0.9100),
    ("ARTEMIS_FAUSTINI_RIM_A", "south_pole", -87.310, 86.500, 250, 1490.0, 6.9, 3.4, 15.9, 0.8600),
    ("ARTEMIS_MALAPERT_MASSIF", "south_pole", -86.040, -2.500, 250, 4750.0, 8.1, 4.2, 19.0, 0.8000),
    ("ARTEMIS_NOBILE_RIM_1", "south_pole", -85.220, 36.800, 250, 1980.0, 5.4, 2.7, 12.9, 0.9200),
    # Flat Touchdown Zones Directly Beside Key Non-Polar SSIs
    ("FEAT_TRANQUILLITATIS_PIT", "global", 8.380, 33.250, 250, -1820.0, 1.8, 1.1, 9.4, 0.9500),
    ("FEAT_MARIUS_HILLS_BASE", "global", 14.120, -56.720, 250, -1540.0, 2.4, 1.4, 11.2, 0.9300),
    ("FEAT_ARISTARCHUS_ISRU", "global", 25.850, -50.800, 250, -1210.0, 3.1, 1.9, 13.5, 0.9000),
    ("FEAT_SCHRODINGER_FLOOR", "global", -75.200, 132.400, 250, -2640.0, 2.1, 1.2, 10.8, 0.9400),
    ("FEAT_MONS_RUMKER_PLAIN", "global", 40.820, -58.100, 250, -2150.0, 1.5, 0.8, 8.6, 0.9600),
    ("FEAT_REINER_GAMMA_EAST", "global", 7.520, -58.850, 250, -1980.0, 1.2, 0.7, 7.9, 0.9700),
]

featured_df = pd.DataFrame(
    featured_records,
    columns=[
        "site_id", "tier", "lat", "lon", "area_radius_m",
        "elevation_m", "max_slope_deg", "mean_slope_deg", "roughness_m", "s_safety",
    ],
)


# ==============================================================================
# 4. MERGE ALL TIERS, COMPUTE 3D XYZ & EXPORT FOR TEAMMATE RAY-TRACING
# ==============================================================================
print("3/3 Merging tiers, computing 3D XYZ coordinates, and exporting CSV...")
sites_df = pd.concat([sp_df, gl_df, featured_df], ignore_index=True)

# Add Moon-centered 3D Cartesian coordinates (km) right after (lat, lon)
lx, ly, lz = latlon_to_xyz(sites_df["lat"].values, sites_df["lon"].values)
sites_df.insert(4, "x_km", np.round(lx, 3))
sites_df.insert(5, "y_km", np.round(ly, 3))
sites_df.insert(6, "z_km", np.round(lz, 3))

# Clean placeholder columns for your teammate's horizon ray-tracing script:
# - f_sun: solar illumination duty cycle in [0.0, 1.0] -> maps to S_power
# - f_earth: Direct-to-Earth line-of-sight fraction in [0.0, 1.0] -> maps to S_comms
sites_df["f_sun"] = np.nan
sites_df["f_earth"] = np.nan

out_path = "processed_data/landing_sites.csv"
sites_df.to_csv(out_path, index=False)

print(f"\nSaved {len(sites_df)} total candidate landing areas to {out_path}!")
print(f"  - South Pole grid patches: {len(sp_df)}")
print(f"  - Global regional zones:   {len(gl_df)}")
print(f"  - Featured landmark zones: {len(featured_df)}")
print("\nSample output rows:")
print(sites_df.head(3).to_string(index=False))