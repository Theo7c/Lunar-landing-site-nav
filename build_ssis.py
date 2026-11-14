import os
import re
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

# Mean lunar radius in kilometers (standard IAU / NASA LOLA reference sphere)
R_MOON_KM = 1737.4

# Ensure output directory exists before saving
os.makedirs("processed_data", exist_ok=True)


# ==============================================================================
# HELPER FUNCTIONS: COORDINATE NORMALIZATION, 3D PROJECTION & WKT PARSING
# ==============================================================================

def normalize_lon(lon_deg):
    """
    Standardize longitudes into the [-180, +180] degree range.
    Planetary datasets often mix [0, 360] and [-180, 180] conventions;
    wrapping everything to [-180, 180] ensures clean distance and map alignment.
    """
    return ((np.asarray(lon_deg, dtype=float) + 180.0) % 360.0) - 180.0


def latlon_to_xyz(lat_deg, lon_deg):
    """
    Convert spherical lunar coordinates (lat, lon in degrees) into 3D
    Moon-centered Cartesian coordinates (x_km, y_km, z_km).

    Why we do this:
    Computing great-circle distances with sin/cos/asin slows down real-time
    queries and suffers from polar distortion. With 3D (x, y, z) coordinates,
    the straight-line chord distance between any landing site and any SSI is
    a simple Euclidean distance (sqrt(dx^2 + dy^2 + dz^2)) that runs in <10 ms
    for any user-selected rover travel radius.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = R_MOON_KM * np.cos(lat) * np.cos(lon)
    y = R_MOON_KM * np.cos(lat) * np.sin(lon)
    z = R_MOON_KM * np.sin(lat)
    return x, y, z


def wkt_centroid(wkt_str):
    """
    Extract the geometric centroid (mean X, mean Y) from a Well-Known Text (WKT)
    POLYGON or MULTIPOLYGON string without requiring heavy external GIS libraries.
    """
    # Match the first exterior coordinate ring inside parentheses: e.g. POLYGON((x1 y1, x2 y2, ...))
    ring_match = re.search(r"\(\s*\(?([^()]+)\)", str(wkt_str))
    text = ring_match.group(1) if ring_match else str(wkt_str)

    # Extract all floating-point numbers (including scientific notation)
    nums = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]

    # Even indices are X coordinates, odd indices are Y coordinates
    xs = np.array(nums[0::2])
    ys = np.array(nums[1::2])

    # WKT polygons repeat the first vertex at the end to close the ring;
    # drop the duplicate vertex so it does not bias the mean centroid
    if len(xs) > 1 and np.isclose(xs[0], xs[-1]) and np.isclose(ys[0], ys[-1]):
        xs, ys = xs[:-1], ys[:-1]

    return float(np.mean(xs)), float(np.mean(ys))


# ==============================================================================
# 1. LOAD POLAR PSR CATALOG & EXTRACT WKT CENTROIDS (1,531 SOUTH POLE TARGETS)
# ==============================================================================
psr_path = "raw_data/south_pole/LPSR_80S_20MPP_ADJ_1km2.csv"

# Read NASA PGDA Permanently Shadowed Region catalog (ignoring any '#' comment headers)
psr_raw = pd.read_csv(psr_path, comment="#")
psr_raw.columns = [c.strip() for c in psr_raw.columns]

# Parse every WKT polygon into its (X, Y) centroid
centroids = np.array([wkt_centroid(w) for w in psr_raw["WKT"]])
raw_x, raw_y = centroids[:, 0], centroids[:, 1]

# Auto-detect coordinate units:
# Values > 360 indicate South Pole Stereographic projection in meters;
# otherwise they are already geographic (lon, lat) degrees.
if np.nanmax(np.abs(raw_x)) > 360.0 or np.nanmax(np.abs(raw_y)) > 90.0:
    print("Detected stereographic meter coordinates in WKT -> projecting to (lat, lon)...")
    # Borrow the exact stereographic Coordinate Reference System (CRS) from our South Pole DEM
    with rasterio.open("raw_data/south_pole/LDEM_80S_80MPP_ADJ.tiff") as dem_src:
        polar_crs = dem_src.crs

    # Reproject polar stereographic (meters) -> spherical lunar (lon, lat) degrees
    raw_lons, raw_lats = transform(
        polar_crs,
        "+proj=longlat +R=1737400 +no_defs",
        raw_x,
        raw_y,
    )
    lats = np.asarray(raw_lats, dtype=float)
    lons = normalize_lon(raw_lons)
else:
    print("Detected degree (lon, lat) coordinates in WKT...")
    lons = normalize_lon(raw_x)
    lats = np.asarray(raw_y, dtype=float)

# Start building the polar SSI DataFrame while preserving all original NASA columns (gid, WKT, etc.)
polar_df = psr_raw.copy()
polar_df["lat"] = np.round(lats, 6)
polar_df["lon"] = np.round(lons, 6)
polar_df["area_km2"] = psr_raw["area_km2"].astype(float)

# Build clean human-readable labels: keep NASA's official crater/PSR code if present,
# and append the unique 'gid' so every target has an unambiguous display name
clean_names = []
for idx, row in polar_df.iterrows():
    orig_name = str(row["name"]).strip() if pd.notna(row["name"]) else ""
    gid_val = row["gid"] if "gid" in polar_df.columns else idx + 1
    if orig_name and orig_name.lower() not in ("nan", "none", "null", ""):
        clean_names.append(f"{orig_name} PSR (#{gid_val})")
    else:
        clean_names.append(f"South Pole PSR #{gid_val}")

polar_df["name"] = clean_names
polar_df["category"] = "polar_psr"
polar_df["region"] = "South Pole (80S-90S)"
polar_df["rationale"] = (
    "Permanently Shadowed Region (PSR) thermally capable of trapping water ice and volatiles."
)

# Compute unified Intrinsic Scientific Value V_i in [1.0, 5.0]:
# - 75% weight from sqrt(area_km2) capped at 25 km^2 (using square root prevents giant
#   craters like Shackleton from completely drowning out smaller reachable micro-traps)
# - 25% weight from extreme polar latitude (closer to 90S -> lower solar elevation & colder traps)
a_norm = np.clip(np.sqrt(np.maximum(0.1, polar_df["area_km2"])) / np.sqrt(25.0), 0.0, 1.0)
lat_norm = np.clip((np.abs(polar_df["lat"]) - 80.0) / 10.0, 0.0, 1.0)
polar_df["v_i"] = np.round(1.0 + 4.0 * (0.75 * a_norm + 0.25 * lat_norm), 4)


# ==============================================================================
# 2. CURATED NON-POLAR & ARTEMIS RIDGE SSIs (28 TARGETS ON SAME [1.0, 5.0] SCALE)
# ==============================================================================
# Outside the polar cold traps, surface ice cannot survive daytime temperatures (>110 K).
# Instead, we represent key global geological, ISRU, lava tube, farside, and historic
# benchmarks using the exact same [1.0, 5.0] V_i scale so polar and global sites compete fairly.
curated_records = [
    # --- Tier 1: Polar Ridge & Lava Tube Skylights / High-Priority ISRU (4.4 - 4.9) ---
    {
        "name": "Shackleton-de Gerlache Connecting Ridge",
        "category": "polar_ridge",
        "region": "South Pole (89.5S)",
        "lat": -89.450,
        "lon": -137.500,
        "area_km2": 6.0,
        "v_i": 4.90,
        "rationale": "Prime Artemis III massif bridge combining near-continuous solar illumination with direct access to two bordering PSRs.",
    },
    {
        "name": "Aristarchus Plateau Pyroclastics",
        "category": "isru_volcanic",
        "region": "Oceanus Procellarum",
        "lat": 26.000,
        "lon": -51.000,
        "area_km2": 120.0,
        "v_i": 4.80,
        "rationale": "Largest dark-mantle pyroclastic glass deposit; high titanium/ilmenite and magmatic hydrogen for ISRU oxygen extraction.",
    },
    {
        "name": "Mare Tranquillitatis Skylight",
        "category": "lava_tube",
        "region": "Mare Tranquillitatis",
        "lat": 8.335,
        "lon": 33.222,
        "area_km2": 0.01,
        "v_i": 4.65,
        "rationale": "Confirmed intact subsurface lava tube conduit (~100 m deep) offering natural radiation and thermal shielding for human bases.",
    },
    {
        "name": "Marius Hills Pit",
        "category": "lava_tube",
        "region": "Oceanus Procellarum",
        "lat": 14.091,
        "lon": -56.770,
        "area_km2": 0.01,
        "v_i": 4.60,
        "rationale": "Skylight collapse pit inside a dense volcanic dome complex; prime candidate for subsurface habitat and volcanic stratigraphy.",
    },
    {
        "name": "Von Karman Crater (SPA Basin)",
        "category": "farside_basin",
        "region": "South Pole-Aitken Basin",
        "lat": -44.800,
        "lon": 175.900,
        "area_km2": 200.0,
        "v_i": 4.55,
        "rationale": "Ancient farside floor within the largest impact basin on the Moon; preserves deep lower-crust/mantle ejecta.",
    },
    {
        "name": "Schrodinger Peak-Ring Basin",
        "category": "farside_basin",
        "region": "South Pole-Aitken Border",
        "lat": -75.000,
        "lon": 132.400,
        "area_km2": 150.0,
        "v_i": 4.50,
        "rationale": "Best-preserved peak-ring impact basin on the Moon with a pyroclastic volcanic vent and deep lower-crustal uplift.",
    },
    {
        "name": "Mare Ingenii Skylight",
        "category": "lava_tube",
        "region": "Farside Highlands",
        "lat": -35.950,
        "lon": 166.050,
        "area_km2": 0.01,
        "v_i": 4.45,
        "rationale": "Rare farside mare skylight pit combining subsurface shelter with farside radio-quiet shielding.",
    },
    {
        "name": "Taurus-Littrow Valley (Apollo 17)",
        "category": "isru_volcanic",
        "region": "Mare Serenitatis Rim",
        "lat": 20.191,
        "lon": 30.772,
        "area_km2": 25.0,
        "v_i": 4.40,
        "rationale": "High-Ti mare basalts, orange pyroclastic volcanic glass beads, and ancient highland massif contacts.",
    },

    # --- Tier 1/2: Deep-Crust Excavations, Thorium Anomalies & Farside Basins (4.0 - 4.35) ---
    {
        "name": "Compton-Belkovich Thorium Anomaly",
        "category": "silicic_volcanism",
        "region": "Northern Farside",
        "lat": 61.100,
        "lon": 99.500,
        "area_km2": 45.0,
        "v_i": 4.35,
        "rationale": "Unique farside silicic volcanic complex with extreme thorium enrichment and localized hydroxyl/water signatures.",
    },
    {
        "name": "Tycho Crater Central Peak",
        "category": "crater_peak",
        "region": "Southern Highlands",
        "lat": -43.310,
        "lon": -11.360,
        "area_km2": 15.0,
        "v_i": 4.25,
        "rationale": "Young (~108 Ma) Copernican impact central uplift exposing pristine deep crustal anorthosite and impact melt.",
    },
    {
        "name": "Copernicus Crater Central Peaks",
        "category": "crater_peak",
        "region": "Mare Insularum",
        "lat": 9.620,
        "lon": -20.080,
        "area_km2": 18.0,
        "v_i": 4.20,
        "rationale": "Key stratigraphic anchor for the Copernican geological period with olivine-rich central peaks.",
    },
    {
        "name": "Tsiolkovskiy Crater",
        "category": "farside_basin",
        "region": "Farside Highlands",
        "lat": -20.400,
        "lon": 129.100,
        "area_km2": 80.0,
        "v_i": 4.15,
        "rationale": "Prominent farside crater with smooth dark mare floor and towering central peak ideal for farside radio astronomy.",
    },
    {
        "name": "Mons Rumker Volcanic Complex (Chang'e-5)",
        "category": "young_volcanism",
        "region": "Northern Oceanus Procellarum",
        "lat": 40.800,
        "lon": -58.100,
        "area_km2": 70.0,
        "v_i": 4.10,
        "rationale": "Young (~2.0 Ga) KREEP-poor mare basalts that extended the known timeline of lunar volcanic activity.",
    },
    {
        "name": "Giordano Bruno Crater",
        "category": "crater_peak",
        "region": "Northern Farside",
        "lat": 35.970,
        "lon": 102.890,
        "area_km2": 12.0,
        "v_i": 4.05,
        "rationale": "Youngest large lunar impact crater (<10 Ma), offering unweathered regolith and fresh impact melt sheets.",
    },

    # --- Tier 2: Volcanic Complexes, Pyroclastic Vents & Magnetic Swirls (3.65 - 3.95) ---
    {
        "name": "Reiner Gamma Magnetic Swirl",
        "category": "magnetic_swirl",
        "region": "Oceanus Procellarum",
        "lat": 7.500,
        "lon": -59.000,
        "area_km2": 40.0,
        "v_i": 3.95,
        "rationale": "Strong crustal magnetic anomaly deflecting solar wind ions and creating high-albedo space-weathering contrasts.",
    },
    {
        "name": "Gruithuisen Domes (Gamma & Delta)",
        "category": "silicic_volcanism",
        "region": "Mare Imbrium Border",
        "lat": 36.300,
        "lon": -39.800,
        "area_km2": 30.0,
        "v_i": 3.95,
        "rationale": "Rare non-mare silicic/rhyolitic volcanic domes (NASA CLPS target) revealing evolved lunar magmatism.",
    },
    {
        "name": "Alphonsus Dark-Halo Craters",
        "category": "isru_volcanic",
        "region": "Eastern Mare Nubium",
        "lat": -13.400,
        "lon": -2.800,
        "area_km2": 18.0,
        "v_i": 3.90,
        "rationale": "Floor-fractured crater containing localized pyroclastic vents along rilles near the lunar prime meridian.",
    },
    {
        "name": "Rima Bode Pyroclastic Vent",
        "category": "isru_volcanic",
        "region": "Sinus Medii",
        "lat": 12.900,
        "lon": -3.900,
        "area_km2": 35.0,
        "v_i": 3.85,
        "rationale": "Equatorial dark-mantle volcanic ash deposit with high iron/titanium content and continuous Direct-to-Earth visibility.",
    },
    {
        "name": "Lichtenberg Young Mare Basalts",
        "category": "young_volcanism",
        "region": "Western Oceanus Procellarum",
        "lat": 31.800,
        "lon": -67.700,
        "area_km2": 55.0,
        "v_i": 3.85,
        "rationale": "Among the youngest basalt flows on the Moon (<1.5 Ga), overlying ray ejecta from Lichtenberg crater.",
    },
    {
        "name": "Sulpicius Gallus Pyroclastics",
        "category": "isru_volcanic",
        "region": "Mare Serenitatis Rim",
        "lat": 19.630,
        "lon": 11.620,
        "area_km2": 22.0,
        "v_i": 3.80,
        "rationale": "Explosive volcanic glass beads rich in trapped volatiles (zinc, sulfur, water) along a graben rille.",
    },
    {
        "name": "Ina Irregular Mare Patch",
        "category": "young_volcanism",
        "region": "Lacus Felicitatis",
        "lat": 18.650,
        "lon": 5.300,
        "area_km2": 4.0,
        "v_i": 3.75,
        "rationale": "Enigmatic blister-like volcanic depression suggesting surprisingly recent (<100 Ma) basaltic outgassing.",
    },
    {
        "name": "Mare Moscoviense Basin",
        "category": "farside_basin",
        "region": "Northern Farside",
        "lat": 27.300,
        "lon": 147.900,
        "area_km2": 90.0,
        "v_i": 3.70,
        "rationale": "One of the few farside maria, exhibiting unusually thin crust and multi-ring basin tectonics.",
    },
    {
        "name": "Mare Crisium Basin (NASA CLPS)",
        "category": "farside_basin",
        "region": "Northeastern Nearside",
        "lat": 18.560,
        "lon": 61.810,
        "area_km2": 110.0,
        "v_i": 3.65,
        "rationale": "Isolated, flat mascon impact basin ideal for geophysical seismometer and heat-flow network deployment.",
    },

    # --- Tier 3: Historic Benchmark & Calibration Sites (2.95 - 3.25) ---
    {
        "name": "Hadley-Apennine (Apollo 15)",
        "category": "historic",
        "region": "Palus Putredinis",
        "lat": 26.132,
        "lon": 3.634,
        "area_km2": 2.0,
        "v_i": 3.25,
        "rationale": "Rille and mountain-front geological benchmark where the Genesis Rock (anorthosite) was sampled.",
    },
    {
        "name": "Oceanus Procellarum / Surveyor 3 (Apollo 12)",
        "category": "historic",
        "region": "Mare Cognitum",
        "lat": -3.012,
        "lon": -23.421,
        "area_km2": 1.5,
        "v_i": 3.15,
        "rationale": "Pinpoint landing benchmark beside the Surveyor 3 probe; sampled Copernicus ray material and mare basalts.",
    },
    {
        "name": "Tranquility Base (Apollo 11)",
        "category": "historic",
        "region": "Mare Tranquillitatis",
        "lat": 0.674,
        "lon": 23.473,
        "area_km2": 1.0,
        "v_i": 3.10,
        "rationale": "First crewed lunar landing site; high-Ti mare baseline and heritage protection zone.",
    },
    {
        "name": "Fra Mauro Highlands (Apollo 14)",
        "category": "historic",
        "region": "Fra Mauro Formation",
        "lat": -3.645,
        "lon": -17.471,
        "area_km2": 1.5,
        "v_i": 3.00,
        "rationale": "Imbrium basin ejecta blanket providing critical radiometric calibration for lunar impact chronology.",
    },
    {
        "name": "Descartes Highlands (Apollo 16)",
        "category": "historic",
        "region": "Central Highlands",
        "lat": -8.973,
        "lon": 15.500,
        "area_km2": 1.5,
        "v_i": 2.95,
        "rationale": "Primary lunar highlands ground-truth site for ancient feldspathic crust composition.",
    },
]

curated_df = pd.DataFrame(curated_records)


# ==============================================================================
# 3. MERGE BOTH DATABASES, COMPUTE 3D CARTESIAN COORDINATES & EXPORT CSV
# ==============================================================================
# Concatenate polar PSRs and curated non-polar SSIs into one unified table
ssis_df = pd.concat([polar_df, curated_df], ignore_index=True, sort=False)

# Assign a unique primary key (SSI_0000, SSI_0001, ...) to every target
ssis_df.insert(0, "ssi_id", [f"SSI_{i:04d}" for i in range(len(ssis_df))])

# Compute 3D Moon-centered Cartesian coordinates (km) for fast Euclidean distance queries
x_km, y_km, z_km = latlon_to_xyz(ssis_df["lat"].values, ssis_df["lon"].values)
ssis_df["x_km"] = np.round(x_km, 3)
ssis_df["y_km"] = np.round(y_km, 3)
ssis_df["z_km"] = np.round(z_km, 3)

# Place standardized schema columns at the front while keeping original NASA columns
# ('gid' and 'WKT' polygon geometries) preserved at the end of the CSV
core_cols = [
    "ssi_id", "name", "category", "region",
    "lat", "lon", "x_km", "y_km", "z_km",
    "area_km2", "v_i", "rationale",
]
extra_cols = [c for c in ssis_df.columns if c not in core_cols]
ssis_df = ssis_df[core_cols + extra_cols]

# Export final SSI database
out_path = "processed_data/ssis.csv"
ssis_df.to_csv(out_path, index=False)

print(f"Saved {len(ssis_df)} total SSIs to {out_path}!")
print(f"  - Polar PSRs: {len(polar_df)}")
print(f"  - Curated Global & Ridge Targets: {len(curated_df)}")
print(f"  - Sample Polar Row:\n{ssis_df[core_cols].head(2).to_string(index=False)}")