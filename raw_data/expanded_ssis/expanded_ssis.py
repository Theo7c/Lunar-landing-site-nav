import os
import re
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform
from scipy.spatial import cKDTree

R_MOON_KM = 1737.4
BASE_SSIS_CSV = "processed_data/ssis.csv"
SOUTH_POLE_DEM = "raw_data/south_pole/LDEM_80S_80MPP_ADJ.tiff"
FULL_PSR_CANDIDATES = [
    "raw_data/south_pole/LPSR_80S_20MPP_ADJ.csv",
    "raw_data/LPSR_80S_20MPP_ADJ.csv",
]
LROC_PITS_CSV = "raw_data/expanded_ssis/lroc_lunar_pits.csv"


def normalize_lon(lon: np.ndarray | float) -> np.ndarray | float:
    """Wrap longitudes into [-180.0, 180.0)."""
    return ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0


def latlon_to_xyz_km(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert spherical lunar coordinates (lat, lon in deg) to 3D Cartesian (x, y, z in km)."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    cos_lat = np.cos(lat_rad)
    x = R_MOON_KM * cos_lat * np.cos(lon_rad)
    y = R_MOON_KM * cos_lat * np.sin(lon_rad)
    z = R_MOON_KM * np.sin(lat_rad)
    return np.round(x, 3), np.round(y, 3), np.round(z, 3)


def wkt_centroid(wkt_str: str) -> tuple[float, float]:
    """Extract the mean (X, Y) centroid from a WKT POLYGON / MULTIPOLYGON string."""
    ring_match = re.search(r"\(\s*\(?([^()]+)\)", str(wkt_str))
    text = ring_match.group(1) if ring_match else str(wkt_str)
    nums = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]
    xs = np.array(nums[0::2], dtype=np.float64)
    ys = np.array(nums[1::2], dtype=np.float64)
    if len(xs) > 1 and np.isclose(xs[0], xs[-1]) and np.isclose(ys[0], ys[-1]):
        xs, ys = xs[:-1], ys[:-1]
    return float(np.mean(xs)), float(np.mean(ys))


def build_extra_curated_targets() -> pd.DataFrame:
    """
    Extra curated Artemis polar ridges, North Pole PSRs, LCROSS ground-truth,
    farside/pyroclastic targets, and Chang'e / Chandrayaan landing sites (SSI_1559..SSI_1577).
    """
    records = [
        ("Cabeus Crater LCROSS Impact Plume Site", "polar_psr", "South Pole (84.9S)", -84.900, -35.500, 35.0, 4.85,
         "Ground-truth volatile cold trap where NASA LCROSS confirmed ~5.6 wt% water ice plus CO, NH3, and H2S."),
        ("Peak Near Shackleton (Artemis Candidate)", "polar_ridge", "South Pole (88.8S)", -88.800, 123.500, 4.5, 4.80,
         "High-illumination massif crest overlooking Shackleton and de Gerlache PSRs."),
        ("Malapert Massif Summit Ridge", "polar_ridge", "South Pole (86.0S)", -86.000, 0.000, 8.0, 4.75,
         "5 km high polar peak providing near-continuous Direct-to-Earth line-of-sight and solar power."),
        ("Haworth-Shoemaker Connecting Ridge", "polar_ridge", "South Pole (87.5S)", -87.500, -15.000, 7.5, 4.70,
         "Elevated divide separating two deep ancient cryogenic cold traps (Haworth and Shoemaker)."),
        ("Nobile Crater Western Rim (VIPER Region)", "polar_ridge", "South Pole (85.2S)", -85.200, 53.500, 10.0, 4.65,
         "Gentle polar rim terrain adjacent to diverse thermal stability zones and shallow subsurface ice."),
        ("Faustini Rim A (Artemis Candidate)", "polar_ridge", "South Pole (87.2S)", -87.200, 77.000, 5.0, 4.60,
         "Illuminated crater rim bench enabling short rover traverses into Faustini PSR."),
        ("Leibnitz Beta Plateau", "polar_ridge", "South Pole (85.5S)", -85.500, 31.000, 12.0, 4.55,
         "Broad, low-slope polar plateau bordering multiple medium-scale PSR cold traps."),
        ("Amundsen Crater Floor & Rim", "polar_ridge", "South Pole (84.5S)", -84.500, 82.800, 18.0, 4.45,
         "Accessible terraced polar crater combining flat sunlit benches with shadow-casting wall traps."),
        ("Whipple Crater North Pole PSR", "polar_psr", "North Pole (89.1N)", 89.140, 120.000, 42.0, 4.88,
         "Deep northern polar cold trap exhibiting strong radar circular polarization ratio (CPR) ice signatures."),
        ("Peary Crater Rim & PSR Complex", "polar_psr", "North Pole (88.6N)", 88.630, 33.000, 65.0, 4.82,
         "Northern polar analog to Shackleton; sunlit rim segments directly border large cryogenic PSRs."),
        ("Hermite Crater Floor PSR", "polar_psr", "North Pole (86.0N)", 86.000, -89.900, 85.0, 4.60,
         "One of the coldest permanently shadowed floors measured by LRO Diviner (<30 K)."),
        ("Rozhdestvenskiy Crater PSR", "polar_psr", "North Pole (85.2N)", 85.200, -155.400, 70.0, 4.52,
         "Large northern farside polar basin harboring extensive perennial shadow zones."),
        ("Lacuna / Mare Fecunditatis Pit", "lava_tube", "Mare Fecunditatis", -0.917, 48.660, 0.01, 4.35,
         "Confirmed mare collapse skylight providing access to subsurface basaltic void stratigraphy."),
        ("King Crater Impact Melt Pit", "lava_tube", "Farside Highlands", 4.960, 120.500, 0.01, 4.10,
         "Large skylight pit in farside impact melt pond; offers shelter and farside radio shielding."),
        ("Schiller-Zucchius Pyroclastics", "isru_volcanic", "Southwestern Highlands", -56.000, -44.500, 40.0, 4.05,
         "High-latitude dark-mantle pyroclastic deposit ideal for regolith oxygen/volatile ISRU trials."),
        ("J. Herschel Pyroclastic Vents", "isru_volcanic", "Mare Frigoris Border", 62.000, -42.000, 32.0, 3.95,
         "Northern mid-high latitude explosive volcanic ash deposit rich in FeO and TiO2."),
        ("Apollo Basin Southern Rim (Chang'e-6)", "farside_basin", "South Pole-Aitken Basin", -41.638, -153.985, 95.0, 4.50,
         "First farside sample-return benchmark; samples ancient SPA impact melt and farside mare basalts."),
        ("Shiv Shakti Point (Chandrayaan-3)", "historic", "Southern Polar Highlands (69.3S)", -69.373, 32.319, 2.0, 3.85,
         "First soft landing in the southern high-latitude highlands; in-situ sulfur and elemental ground truth."),
        ("Statio Tianhe / Von Karman (Chang'e-4)", "historic", "South Pole-Aitken Basin", -45.456, 177.588, 2.0, 3.90,
         "First farside soft landing; Yutu-2 rover ground-penetrating radar and mantle-derived regolith spectra."),
    ]
    df = pd.DataFrame(
        records,
        columns=["name", "category", "region", "lat", "lon", "area_km2", "v_i", "rationale"],
    )
    df["lon"] = normalize_lon(df["lon"].to_numpy(dtype=np.float64))
    return df


def load_south_pole_micro_psrs(start_label_num: int = 1532) -> pd.DataFrame:
    """
    Load sub-kilometer South Pole PSRs (0.05 <= area_km2 < 1.0) from the full unfiltered
    NASA PGDA South Pole PSR catalog (`LPSR_80S_20MPP_ADJ.csv`) and score them via:
    v_i = round(1.20 + 1.80 * sqrt(clip(area_km2, 0.05, 1.0)), 4).
    """
    psr_path = next((p for p in FULL_PSR_CANDIDATES if os.path.exists(p)), None)
    if psr_path is None:
        print(f"[expand_ssis] Warning: Unfiltered PSR catalog not found at {FULL_PSR_CANDIDATES}; skipping micro-PSRs.")
        return pd.DataFrame()

    raw = pd.read_csv(psr_path, comment="#")
    raw.columns = [c.strip() for c in raw.columns]
    raw["area_km2"] = pd.to_numeric(raw["area_km2"], errors="coerce")

    micro = raw[(raw["area_km2"] >= 0.05) & (raw["area_km2"] < 1.0)].copy()
    micro = micro.sort_values(by="area_km2", ascending=False).reset_index(drop=True)

    centroids = np.array([wkt_centroid(w) for w in micro["WKT"]], dtype=np.float64)
    raw_x, raw_y = centroids[:, 0], centroids[:, 1]

    if np.nanmax(np.abs(raw_x)) > 360.0 or np.nanmax(np.abs(raw_y)) > 90.0:
        with rasterio.open(SOUTH_POLE_DEM) as dem_src:
            polar_crs = dem_src.crs
        raw_lons, raw_lats = transform(
            polar_crs,
            "+proj=longlat +R=1737400 +no_defs",
            raw_x,
            raw_y,
        )
        lats = np.asarray(raw_lats, dtype=np.float64)
        lons = normalize_lon(raw_lons)
    else:
        lons = normalize_lon(raw_x)
        lats = np.asarray(raw_y, dtype=np.float64)

    areas = np.round(micro["area_km2"].to_numpy(dtype=np.float64), 3)
    v_scores = np.round(1.20 + 1.80 * np.sqrt(np.clip(areas, 0.05, 1.0)), 4)

    return pd.DataFrame({
        "name": [f"South Pole Micro-PSR #{start_label_num + i}" for i in range(len(micro))],
        "category": "polar_micro_psr",
        "region": "South Pole (80S-90S)",
        "lat": np.round(lats, 6),
        "lon": np.round(lons, 6),
        "area_km2": areas,
        "v_i": v_scores,
        "rationale": "Sub-kilometer polar cold trap reachable by short-range rovers from adjacent sunlit ridges.",
    })


def build_irregular_mare_patches() -> pd.DataFrame:
    """
    39 Irregular Mare Patches (IMPs) from Braden et al. (2014) / ASU LROC IMP catalog
    (SSI_12318..SSI_12356), scored via:
    v_i = round(3.50 + 0.35 * sqrt(clip(area_km2, 0.05, 2.5) / 1.25), 4).
    """
    imps = [
        ("Sosigenes", 8.335, 19.071, 2.10),
        ("Cauchy-5", 12.170, 36.800, 0.85),
        ("Nubium", -25.680, -27.640, 0.42),
        ("Maskelyne", 4.330, 33.750, 0.28),
        ("Carrel (Jansen B)", 10.670, 26.680, 0.22),
        ("Arago", 6.150, 21.430, 0.18),
        ("Maclear", 10.520, 20.100, 0.16),
        ("Ritter", 1.980, 19.270, 0.14),
        ("Manilius", 14.450, 9.050, 0.12),
        ("Hyginus", 7.770, 6.320, 0.11),
        ("Ariadaeus", 6.450, 13.480, 0.09),
        ("Ross", 11.670, 21.740, 0.08),
        ("Cayley", 3.960, 15.090, 0.07),
        ("Manners", 4.580, 19.990, 0.07),
        ("Sabine", 1.380, 20.070, 0.06),
        ("Dionysius", 2.770, 17.290, 0.06),
        ("Menelaus", 16.260, 15.930, 0.05),
        ("Tacquet", 16.640, 19.200, 0.05),
        ("Plinius", 15.360, 23.610, 0.05),
        ("Dawes", 17.210, 26.340, 0.04),
        ("Vitruvius", 17.660, 31.280, 0.04),
        ("Maraldi", 19.370, 34.800, 0.04),
        ("Franck", 22.590, 35.560, 0.04),
        ("Littrow", 21.500, 31.390, 0.04),
        ("Bessel", 21.730, 17.920, 0.03),
        ("Linne", 27.740, 11.800, 0.03),
        ("Lambert", 25.770, -20.990, 0.03),
        ("Euler", 23.260, -29.180, 0.03),
        ("Brayley", 20.900, -36.940, 0.03),
        ("Tobias Mayer", 15.600, -29.090, 0.03),
        ("Hortensius", 6.470, -28.000, 0.03),
        ("Milichius", 10.010, -30.230, 0.03),
        ("Kepler", 8.120, -38.010, 0.02),
        ("Marius", 11.900, -50.840, 0.02),
        ("Herodotus", 23.250, -49.840, 0.02),
        ("Prinz", 25.490, -44.140, 0.02),
        ("Krieger", 29.020, -45.610, 0.02),
        ("Delisle", 29.980, -34.680, 0.02),
        ("Diophantus", 27.620, -34.300, 0.02),
    ]
    df = pd.DataFrame(imps, columns=["name", "lat", "lon", "area_km2"])
    df["category"] = "young_volcanism"
    df["region"] = "Nearside Mare"
    df["lon"] = normalize_lon(df["lon"].to_numpy(dtype=np.float64))
    areas = df["area_km2"].to_numpy(dtype=np.float64)
    df["v_i"] = np.round(3.50 + 0.35 * np.sqrt(np.clip(areas, 0.05, 2.5) / 1.25), 4)
    df["rationale"] = "Irregular Mare Patch (IMP) representing geologically young (<100 Ma) lunar basaltic volcanism."
    return df[["name", "category", "region", "lat", "lon", "area_km2", "v_i", "rationale"]]


def load_optional_lroc_pits() -> pd.DataFrame:
    """Optional loader for `raw_data/expanded_ssis/lroc_lunar_pits.csv` if present."""
    if not os.path.exists(LROC_PITS_CSV):
        return pd.DataFrame()

    df = pd.read_csv(LROC_PITS_CSV)
    df.columns = [c.strip().rstrip(".").lower() for c in df.columns]

    lat_c = next((c for c in ["lat", "latitude", "center_lat"] if c in df.columns), None)
    lon_c = next((c for c in ["long", "lon", "longitude", "center_lon"] if c in df.columns), None)
    if not lat_c or not lon_c:
        return pd.DataFrame()

    name_c = next((c for c in ["name", "pit_name", "id"] if c in df.columns), None)
    host_c = next((c for c in ["host feat", "host", "feature", "region"] if c in df.columns), None)
    fmax_c = next((c for c in ["funnel max diam. (m)", "funnel max diam (m)"] if c in df.columns), None)
    fmin_c = next((c for c in ["funnel min diam. (m)", "funnel min diam (m)"] if c in df.columns), None)
    depth_c = next((c for c in ["depth (m)", "depth"] if c in df.columns), None)

    lats = pd.to_numeric(df[lat_c], errors="coerce").to_numpy(dtype=np.float64)
    lons = normalize_lon(pd.to_numeric(df[lon_c], errors="coerce").to_numpy(dtype=np.float64))
    valid = ~(np.isnan(lats) | np.isnan(lons))
    df = df[valid].reset_index(drop=True)
    lats, lons = lats[valid], lons[valid]

    names = df[name_c].astype(str).str.strip().tolist() if name_c else [f"LROC Pit #{i + 1}" for i in range(len(df))]
    hosts = df[host_c].fillna("Unknown").astype(str).str.strip().tolist() if host_c else ["Lunar Surface"] * len(df)

    if fmax_c and fmin_c:
        d_max = pd.to_numeric(df[fmax_c], errors="coerce").fillna(20.0).to_numpy(dtype=np.float64)
        d_min = pd.to_numeric(df[fmin_c], errors="coerce").fillna(d_max).to_numpy(dtype=np.float64)
        area_km2 = np.round((np.pi * (d_max / 2.0) * (d_min / 2.0)) / 1_000_000.0, 6)
    else:
        d_max = np.full(len(df), 25.0, dtype=np.float64)
        area_km2 = np.full(len(df), 0.0005, dtype=np.float64)

    depths = pd.to_numeric(df[depth_c], errors="coerce").fillna(10.0).to_numpy(dtype=np.float64) if depth_c else np.full(len(df), 10.0)
    v_scores = np.round(3.65 + 1.00 * (0.65 * np.clip(depths / 80.0, 0.0, 1.0) + 0.35 * np.clip(d_max / 100.0, 0.0, 1.0)), 4)

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


def expand_ssis(dedup_radius_km: float = 0.25) -> None:
    if not os.path.exists(BASE_SSIS_CSV):
        raise FileNotFoundError(f"{BASE_SSIS_CSV} not found. Run `python build_ssis.py` first.")

    base_df = pd.read_csv(BASE_SSIS_CSV)
    base_df["lon"] = normalize_lon(base_df["lon"].to_numpy(dtype=np.float64))
    if not {"x_km", "y_km", "z_km"}.issubset(base_df.columns):
        bx, by, bz = latlon_to_xyz_km(
            base_df["lat"].to_numpy(dtype=np.float64),
            base_df["lon"].to_numpy(dtype=np.float64),
        )
        base_df["x_km"], base_df["y_km"], base_df["z_km"] = bx, by, bz

    n_large_psrs = int((base_df["category"] == "polar_psr").sum())

    cand_frames = [
        build_extra_curated_targets(),
        load_south_pole_micro_psrs(start_label_num=n_large_psrs + 1),
        build_irregular_mare_patches(),
        load_optional_lroc_pits(),
    ]
    cand_frames = [f for f in cand_frames if not f.empty]
    cand_df = pd.concat(cand_frames, ignore_index=True)

    cx, cy, cz = latlon_to_xyz_km(
        cand_df["lat"].to_numpy(dtype=np.float64),
        cand_df["lon"].to_numpy(dtype=np.float64),
    )
    cand_df["x_km"], cand_df["y_km"], cand_df["z_km"] = cx, cy, cz

    # 3D Cartesian spatial deduplication against existing ssis.csv entries
    existing_xyz = base_df[["x_km", "y_km", "z_km"]].to_numpy(dtype=np.float64)
    tree = cKDTree(existing_xyz)
    kept_rows = []

    for _, row in cand_df.iterrows():
        pt = np.array([row["x_km"], row["y_km"], row["z_km"]], dtype=np.float64)
        radius = dedup_radius_km if row["category"] == "polar_micro_psr" else 1.5
        if len(tree.query_ball_point(pt, r=radius)) > 0:
            continue
        kept_rows.append(row)
        existing_xyz = np.vstack([existing_xyz, pt])
        tree = cKDTree(existing_xyz)

    if not kept_rows:
        print(f"[expand_ssis] No new non-overlapping targets to append. Total SSIs: {len(base_df)}")
        return

    added_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    start_idx = len(base_df)
    added_df["ssi_id"] = [f"SSI_{i:04d}" for i in range(start_idx, start_idx + len(added_df))]

    combined_df = pd.concat([base_df, added_df], ignore_index=True, sort=False)
    core_cols = [
        "ssi_id", "name", "category", "region",
        "lat", "lon", "x_km", "y_km", "z_km",
        "area_km2", "v_i", "rationale",
    ]
    extra_cols = [c for c in combined_df.columns if c not in core_cols]
    combined_df = combined_df[core_cols + extra_cols]

    combined_df.to_csv(BASE_SSIS_CSV, index=False)
    print(f"[expand_ssis] Appended {len(added_df)} new SSIs -> Total SSIs in {BASE_SSIS_CSV}: {len(combined_df)}")


if __name__ == "__main__":
    expand_ssis()