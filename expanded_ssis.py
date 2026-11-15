import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

R_MOON_KM = 1737.4
BASE_SSIS_CSV = "processed_data/ssis.csv"
RAW_EXPAND_DIR = "raw_data/expanded_ssis"
LROC_PITS_CSV = os.path.join(RAW_EXPAND_DIR, "lroc_lunar_pits.csv")

OUTPUT_COLUMNS = [
    "ssi_id",
    "name",
    "category",
    "region",
    "lat",
    "lon",
    "x_km",
    "y_km",
    "z_km",
    "area_km2",
    "v_i",
    "rationale",
]


def normalize_lon(lon: np.ndarray | float) -> np.ndarray | float:
    """Wrap longitudes into [-180.0, 180.0)."""
    return ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0


def latlon_to_xyz_km(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert spherical selenographic (lat, lon) in degrees to 3D Cartesian (x, y, z) in km."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    cos_lat = np.cos(lat_rad)
    x = R_MOON_KM * cos_lat * np.cos(lon_rad)
    y = R_MOON_KM * cos_lat * np.sin(lon_rad)
    z = R_MOON_KM * np.sin(lat_rad)
    return np.round(x, 4), np.round(y, 4), np.round(z, 4)


def _col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name from `candidates`."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_lroc_pits() -> pd.DataFrame:
    """
    Load the ASU LROC Lunar Pits Atlas (`raw_data/expanded_ssis/lroc_lunar_pits.csv`)
    and compute surface funnel area and scientific value V_i in [3.65, 4.65].
    """
    if not os.path.exists(LROC_PITS_CSV):
        print(f"[expand_ssis] Warning: {LROC_PITS_CSV} not found; skipping LROC pits.")
        return pd.DataFrame()

    df = pd.read_csv(LROC_PITS_CSV)
    # Normalize column headers by stripping whitespace and trailing periods (e.g. "Lat." -> "lat")
    df.columns = [c.strip().rstrip(".").lower() for c in df.columns]

    lat_c = _col(df, ["lat", "latitude", "center_lat"])
    lon_c = _col(df, ["long", "lon", "longitude", "center_lon"])
    name_c = _col(df, ["name", "pit_name", "id"])
    host_c = _col(df, ["host feat", "host", "feature", "region"])
    fmax_c = _col(df, ["funnel max diam. (m)", "funnel max diam (m)"])
    fmin_c = _col(df, ["funnel min diam. (m)", "funnel min diam (m)"])
    depth_c = _col(df, ["depth (m)", "depth"])

    if not lat_c or not lon_c:
        print("[expand_ssis] Warning: Could not locate Lat/Long columns in lroc_lunar_pits.csv.")
        return pd.DataFrame()

    lats = pd.to_numeric(df[lat_c], errors="coerce").to_numpy(dtype=np.float64)
    lons = normalize_lon(pd.to_numeric(df[lon_c], errors="coerce").to_numpy(dtype=np.float64))
    valid = ~(np.isnan(lats) | np.isnan(lons))
    df = df[valid].reset_index(drop=True)
    lats, lons = lats[valid], lons[valid]

    names = (
        df[name_c].astype(str).str.strip().tolist()
        if name_c
        else [f"LROC Pit #{i + 1}" for i in range(len(df))]
    )
    hosts = (
        df[host_c].fillna("Unknown").astype(str).str.strip().tolist()
        if host_c
        else ["Lunar Surface"] * len(df)
    )

    if fmax_c and fmin_c:
        d_max = pd.to_numeric(df[fmax_c], errors="coerce").fillna(20.0).to_numpy(dtype=np.float64)
        d_min = pd.to_numeric(df[fmin_c], errors="coerce").fillna(d_max).to_numpy(dtype=np.float64)
        area_km2 = np.round((np.pi * (d_max / 2.0) * (d_min / 2.0)) / 1_000_000.0, 6)
    else:
        d_max = np.full(len(df), 25.0, dtype=np.float64)
        area_km2 = np.full(len(df), 0.0005, dtype=np.float64)

    depths = (
        pd.to_numeric(df[depth_c], errors="coerce").fillna(10.0).to_numpy(dtype=np.float64)
        if depth_c
        else np.full(len(df), 10.0, dtype=np.float64)
    )
    depth_norm = np.clip(depths / 80.0, 0.0, 1.0)
    diam_norm = np.clip(d_max / 100.0, 0.0, 1.0)
    v_scores = np.round(3.65 + 1.00 * (0.65 * depth_norm + 0.35 * diam_norm), 4)

    return pd.DataFrame({
        "name": [f"{n} Pit" if not n.lower().endswith("pit") else n for n in names],
        "category": "lava_tube",
        "region": [f"Host: {h}" for h in hosts],
        "lat": np.round(lats, 6),
        "lon": np.round(lons, 6),
        "area_km2": np.maximum(0.0001, area_km2),
        "v_i": v_scores,
        "rationale": [
            f"LROC collapse skylight pit (~{int(round(dm))} m wide, ~{d:.1f} m deep) offering subsurface thermal/radiation shelter."
            for dm, d in zip(d_max, depths)
        ],
    })


def build_curated_targets() -> pd.DataFrame:
    """
    Curated global geological, volcanic/pyroclastic, Artemis polar ridge, and historic
    mission targets synthesized from NASA/CNSA/ISRO releases and USGS nomenclature.
    """
    records = [
        # Polar Massifs & LCROSS Volatile Ground-Truth
        ("Shackleton-de Gerlache Ridge", "polar_ridge", "South Pole", -89.6500, -135.0000, 12.50, 4.75,
         "Artemis III candidate connecting ridge with near-continuous solar illumination adjacent to Shackleton PSR."),
        ("Malapert Massif", "polar_ridge", "South Pole", -86.0400, -2.7000, 45.00, 4.45,
         "Prominent South Pole massif offering sustained Direct-to-Earth visibility and solar access."),
        ("Haworth-Shoemaker Ridge", "polar_ridge", "South Pole", -87.4500, 14.5000, 18.00, 4.50,
         "Elevated highland ridge separating Haworth and Shoemaker volatile cold traps."),
        ("Nobile Rim 1", "polar_ridge", "South Pole", -85.2000, 53.5000, 22.00, 4.40,
         "Artemis III candidate rim overlooking Nobile crater cold traps."),
        ("Faustini Rim A", "polar_ridge", "South Pole", -87.3000, 77.0000, 15.00, 4.45,
         "Artemis III candidate site on the sunlit rim of Faustini PSR."),
        ("Cabeus LCROSS Impact Site", "volatile_ground_truth", "South Pole", -84.6750, -48.7250, 2.00, 4.90,
         "2009 LCROSS kinetic impact plume site confirming ~5.6 wt% water ice and volatiles in polar regolith."),

        # Volcanic & Pyroclastic Benchmarks
        ("Ina Irregular Mare Patch", "young_volcanism", "Lacus Felicitatis", 18.6500, 5.3000, 4.00, 4.30,
         "Type-locality Irregular Mare Patch (~33-100 Ma) demonstrating geologically recent lunar basaltic volcanism."),
        ("Marius Hills Volcanic Complex", "volcanic_dome", "Oceanus Procellarum", 14.0900, -56.7700, 350.00, 4.20,
         "Densest concentration of volcanic domes, cones, and sinuous rilles on the Moon."),
        ("Gruithuisen Domes", "silicic_volcanism", "Mare Imbrium Border", 36.3000, -39.8000, 180.00, 4.40,
         "Non-mare silica-rich (rhyolitic/dacitic) volcanic domes; NASA CLPS Lunar-VISE target."),
        ("Aristarchus Plateau", "pyroclastic_isru", "Oceanus Procellarum", 23.7000, -47.4000, 400.00, 4.35,
         "Largest lunar dark-mantle pyroclastic deposit and Vallis Schröteri rille system; high ISRU glass potential."),
        ("Sulpicius Gallus Pyroclastics", "pyroclastic_isru", "SW Mare Serenitatis", 19.6300, 11.6800, 120.00, 4.15,
         "Titanium-rich pyroclastic volcanic ash mantle suitable for ISRU oxygen extraction."),
        ("Taurus-Littrow Dark Mantle", "pyroclastic_isru", "SE Mare Serenitatis", 20.1908, 30.7717, 65.00, 4.25,
         "Orange/black volcanic glass beads sampled by Apollo 17; benchmark pyroclastic volatile reservoir."),

        # Farside Basins & Geophysical Anomalies
        ("Von Karman Crater (SPA)", "farside_basin", "South Pole-Aitken Basin", -45.4570, 177.5990, 250.00, 4.55,
         "Farside floor of South Pole-Aitken Basin; Chang'e-4 landing region probing upper mantle/lower crust."),
        ("Schrodinger Basin", "peak_ring_basin", "South Pole-Aitken Basin", -75.0000, 132.4000, 320.00, 4.65,
         "Well-preserved peak-ring impact basin with explosive pyroclastic vents and uplifted deep crustal rocks."),
        ("Apollo Basin (SPA)", "farside_basin", "South Pole-Aitken Basin", -36.1000, -151.8000, 500.00, 4.60,
         "Deepest basin interior within South Pole-Aitken; Chang'e-6 farside sample-return target."),
        ("Compton-Belkovich Thorium Anomaly", "geochemical_anomaly", "Farside Highlands", 61.1000, 99.5000, 90.00, 4.50,
         "Isolated farside silicic volcanic complex with strong thorium enrichment and elevated subsurface heat flow."),
        ("Reiner Gamma Swirl", "magnetic_anomaly", "Oceanus Procellarum", 7.4200, -59.0400, 70.00, 4.25,
         "Prototypical lunar magnetic anomaly and high-albedo swirl; NASA Lunar Vertex rover destination."),

        # Historic Ground-Truth Landing Sites
        ("Apollo 11 (Statio Tranquillitatis)", "historic_site", "Mare Tranquillitatis", 0.6741, 23.4730, 0.25, 4.10,
         "First human lunar landing (1969); calibrated high-Ti mare basalt radiometric chronology."),
        ("Apollo 12 (Oceanus Procellarum)", "historic_site", "Oceanus Procellarum", -3.0124, -23.4216, 0.25, 4.00,
         "Precision landing beside Surveyor 3 (1969); younger low-Ti mare basalts and Copernicus ray ejecta."),
        ("Apollo 14 (Fra Mauro)", "historic_site", "Fra Mauro Highlands", -3.6453, -17.4714, 0.25, 4.05,
         "Imbrium basin ejecta blanket (1971) constraining the age of the Imbrium impact event."),
        ("Apollo 15 (Hadley-Apennine)", "historic_site", "Palus Putredinis", 26.1322, 3.6339, 1.50, 4.25,
         "Hadley Rille and Apennine front (1971); first Lunar Roving Vehicle traverse and Genesis Rock anorthosite."),
        ("Apollo 16 (Descartes Highlands)", "historic_site", "Descartes Highlands", -8.9730, 15.5002, 1.00, 4.10,
         "Central lunar highland plains and Cayley formation ground truth (1972)."),
        ("Apollo 17 (Taurus-Littrow)", "historic_site", "Taurus-Littrow Valley", 20.1908, 30.7717, 1.50, 4.30,
         "Final Apollo mission (1972); ancient highland massifs, volcanic glass beads, and deep drill core."),
        ("Chang'e-4 (Statio Tianhe)", "historic_site", "Von Karman Crater", -45.4570, 177.5990, 0.25, 4.35,
         "First soft landing on the lunar farside (2019) with Yutu-2 rover spectral/GPR traverse."),
        ("Chang'e-5 (Mons Rumker)", "historic_site", "Northern Oceanus Procellarum", 43.0576, -51.9161, 0.25, 4.40,
         "Returned ~2.0 Ga young mare basalts (2020), recalibrating lunar crater-counting chronology."),
        ("Chang'e-6 Farside Sample Site", "historic_site", "Southern Apollo Basin", -41.6385, -153.9852, 0.25, 4.65,
         "First lunar farside sample return (2024), recovering ~1.935 kg of SPA basin material."),
        ("Chandrayaan-3 (Shiv Shakti Point)", "historic_site", "South Polar Highlands", -69.3730, 32.3190, 0.25, 4.30,
         "First soft landing in the southern high-latitude region (2023); Pragyan rover APXS/LIBS measurements."),
    ]

    df = pd.DataFrame(
        records,
        columns=["name", "category", "region", "lat", "lon", "area_km2", "v_i", "rationale"],
    )
    df["lon"] = normalize_lon(df["lon"].to_numpy(dtype=np.float64))
    return df


def expand_ssis(dedup_radius_km: float = 1.5) -> None:
    if os.path.exists(BASE_SSIS_CSV):
        base_df = pd.read_csv(BASE_SSIS_CSV)
        base_df["lon"] = normalize_lon(base_df["lon"].to_numpy(dtype=np.float64))
        if not {"x_km", "y_km", "z_km"}.issubset(base_df.columns):
            bx, by, bz = latlon_to_xyz_km(
                base_df["lat"].to_numpy(dtype=np.float64),
                base_df["lon"].to_numpy(dtype=np.float64),
            )
            base_df["x_km"], base_df["y_km"], base_df["z_km"] = bx, by, bz
    else:
        base_df = pd.DataFrame(columns=OUTPUT_COLUMNS)

    new_frames = [
        load_lroc_pits(),
        build_curated_targets(),
    ]
    new_frames = [f for f in new_frames if not f.empty]

    if not new_frames:
        print("[expand_ssis] No new target tables loaded.")
        return

    cand_df = pd.concat(new_frames, ignore_index=True)
    cx, cy, cz = latlon_to_xyz_km(
        cand_df["lat"].to_numpy(dtype=np.float64),
        cand_df["lon"].to_numpy(dtype=np.float64),
    )
    cand_df["x_km"], cand_df["y_km"], cand_df["z_km"] = cx, cy, cz

    # 3D Cartesian spatial deduplication against existing ssis.csv entries
    kept_rows = []
    existing_xyz = (
        base_df[["x_km", "y_km", "z_km"]].to_numpy(dtype=np.float64)
        if not base_df.empty
        else np.empty((0, 3), dtype=np.float64)
    )
    tree = cKDTree(existing_xyz) if len(existing_xyz) > 0 else None

    for _, row in cand_df.iterrows():
        pt = np.array([row["x_km"], row["y_km"], row["z_km"]], dtype=np.float64)
        if tree is not None and len(tree.query_ball_point(pt, r=dedup_radius_km)) > 0:
            continue
        kept_rows.append(row)
        existing_xyz = np.vstack([existing_xyz, pt])
        tree = cKDTree(existing_xyz)

    if not kept_rows:
        print(f"[expand_ssis] All candidate targets already exist within {dedup_radius_km} km. Total SSIs: {len(base_df)}")
        return

    added_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    start_idx = len(base_df) + 1
    added_df["ssi_id"] = [f"SSI_{i:04d}" for i in range(start_idx, start_idx + len(added_df))]
    added_df = added_df[OUTPUT_COLUMNS]

    combined_df = pd.concat(
        [base_df[OUTPUT_COLUMNS] if not base_df.empty else pd.DataFrame(columns=OUTPUT_COLUMNS), added_df],
        ignore_index=True,
    )
    os.makedirs(os.path.dirname(BASE_SSIS_CSV) or ".", exist_ok=True)
    combined_df.to_csv(BASE_SSIS_CSV, index=False)

    print(f"[expand_ssis] Appended {len(added_df)} new SSIs (dedup radius={dedup_radius_km} km).")
    print(f"[expand_ssis] Total SSIs in {BASE_SSIS_CSV}: {len(combined_df)}")


if __name__ == "__main__":
    expand_ssis()
