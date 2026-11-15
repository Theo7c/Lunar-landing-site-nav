# Lunar Landing Site Selection — Backend & Data Generation Pipeline

This directory contains the offline geospatial data-generation scripts and the FastAPI backend powering the Lunar Landing Site Selection platform. It processes NASA LRO LOLA Digital Elevation Models (DEMs) and planetary science catalogs into fast spatial lookup tables, computes 360-degree spherical horizon profiles, interfaces with NASA's JPL Horizons API and Google Gemini, and dynamically renders 2D topographic rover traverse maps.

---

## 1. Architecture Overview

The backend is split into two layers:

1. **Offline Data Generation Pipeline** (run once to build `processed_data/`):
   * `generate_landing_sites.py` — Samples ~60,000 candidate landing sites across a dual-tier lunar grid (South Pole + Global) plus featured Artemis/global landmarks and computes local slope, elevation, and roughness.
   * `generate_horizons.py` — Ray-casts 360-degree local horizon elevation profiles out to $80\text{ km}$ for every candidate site using exact spherical curvature geometry.
   * `build_ssis.py` & `expanded_ssis.py` — Builds and expands a unified catalog of Sites of Scientific Interest (SSIs) with standardized intrinsic scientific value scores ($V_i \in [1.0, 5.0]$) and 3D Cartesian coordinates for fast spatial queries.
2. **Real-Time FastAPI Service (`main.py` & `api/analysis/`)**:
   * Serves batch Multi-Criteria Decision Making (MCDM) rankings, on-demand single-site terrain and horizon extraction (`inspect_point.py`), live JPL Horizons Sun/Earth ephemeris (`horizon.py`), Gemini AI mission-weight generation (`GeminiAPICaller.py`), and in-memory 2D hillshaded rover route maps (`render_route_map.py`).

---

## 2. Directory Structure

```text
.
├── main.py                                # FastAPI application & HTTP endpoints
├── generate_landing_sites.py              # Step 1: Build ~60k candidate landing sites
├── generate_horizons.py                   # Step 2: Ray-cast 360-deg horizon profiles
├── build_ssis.py                          # Step 3: Build base SSI catalog (PSRs + curated)
├── expanded_ssis.py                       # Step 4: Append micro-PSRs, IMPs, pits & extra targets
├── api/
│   └── analysis/
│       ├── inspect_point.py               # On-demand DEM patch & horizon extractor + caching
│       ├── horizon.py                     # NASA JPL Horizons API client (Sun & Earth)
│       ├── render_route_map.py            # 2D hillshaded DEM & rover traverse PNG renderer
│       ├── GeminiAPICaller.py             # Gemini API mission weight & site recommender
│       ├── score_calculator.py            # Single-site safety & science scoring logic
│       ├── scores_calculator.py           # Batch MCDM estimator & rover route solver
│       └── AstronomicalScore.py           # Standalone prototype for horizon/body visibility
├── raw_data/                              # Raw NASA DEMs & planetary catalogs (not tracked)
│   ├── ldem_64.tif                        # Global LOLA DEM (~474 m/px)
│   ├── south_pole/
│   │   ├── LDEM_80S_80MPP_ADJ.tiff        # South Pole stereographic DEM (80 m/px)
│   │   ├── LPSR_80S_20MPP_ADJ_1km2.csv    # NASA PGDA South Pole PSRs (>= 1 km^2)
│   │   └── LPSR_80S_20MPP_ADJ.csv         # Full unfiltered South Pole PSR catalog (optional)
│   └── expanded_ssis/
│       └── lroc_lunar_pits.csv            # LROC collapse pit catalog (optional)
└── processed_data/                        # Generated CSVs & runtime caches
    ├── landing_sites.csv
    ├── landing_site_horizons.csv
    ├── landing_site_horizons_cache.npz    # Auto-generated binary cache for fast lookup
    ├── ssis.csv
    ├── user_inspected_sites.csv           # Persisted custom user-clicked sites
    └── user_horizon_profiles.csv          # Persisted custom user-clicked horizons
```

---

## 3. Prerequisites & Installation

### Python Dependencies
Install the required packages (Python 3.10+ recommended):

```bash
pip install fastapi uvicorn numpy pandas scipy rasterio matplotlib requests python-dotenv google-genai
```

### Environment Variables
Create a `.env` file in the project root with your Google Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Raw Data Setup
Download the required NASA LRO LOLA / PGDA datasets into `raw_data/`:
* **Global DEM:** `raw_data/ldem_64.tif` (NASA SVS LOLA 64 ppd GeoTIFF). *Note: `inspect_point.py` also supports automatic `/vsicurl/` streaming fallback from NASA SVS if local DEM files are absent.*
* **South Pole DEM:** `raw_data/south_pole/LDEM_80S_80MPP_ADJ.tiff` ($80\text{ m/px}$ polar stereographic DEM).
* **South Pole PSR Catalog:** `raw_data/south_pole/LPSR_80S_20MPP_ADJ_1km2.csv` (and optionally `LPSR_80S_20MPP_ADJ.csv` for sub-km micro-PSRs).
* **Optional LROC Pits:** `raw_data/expanded_ssis/lroc_lunar_pits.csv`.

---

## 4. Data Generation Pipeline

Run the offline build scripts in the following order to populate `processed_data/`:

```bash
# 1. Generate ~60,000 candidate landing sites (South Pole + Equal-Area Global + 12 Featured)
python generate_landing_sites.py

# 2. Compute 360-degree horizon elevation profiles (out to 80 km) for all sites
python generate_horizons.py

# 3. Build the base Sites of Scientific Interest (SSI) catalog
python build_ssis.py

# 4. Expand the SSI catalog with micro-PSRs, IMPs, North Pole PSRs, and optional LROC pits
python expanded_ssis.py
```

### Key Technical Details
* **Dual-Tier Sampling (`generate_landing_sites.py`):**
  * **South Pole ($80^\circ\text{S}$–$90^\circ\text{S}$):** ~30,000 sites at $80\text{ m/px}$ using a $7\times 7$ px ($250\text{ m}$ radius) touchdown dispersion box for max/mean slope and a $51\times 51$ px ($\sim 2\text{ km}$) window for EVA roughness.
  * **Global ($80^\circ\text{S}$–$85^\circ\text{N}$):** ~30,000 sites on a cosine-corrected equal-area grid (~$474\text{ m/px}$) using fast patch-based $5\times 5$ extraction to avoid full-raster memory overhead.
* **Spherical Horizon Ray-Casting (`generate_horizons.py`):** Computes tangent-plane horizon elevation angles across 360 azimuths ($0^\circ$–$359^\circ$) out to $80\text{ km}$ accounting for lunar curvature ($R_{\text{Moon}} = 1,737.4\text{ km}$). South Pole sites merge far-field global rays with high-resolution $80\text{ m/px}$ near-field rays ($< 5\text{ km}$).
* **3D Cartesian Coordinates (`x_km`, `y_km`, `z_km`):** Both landing sites and SSIs are projected into Moon-centered 3D Cartesian space so rover range queries and spatial deduplication run in milliseconds via `scipy.spatial.cKDTree` without polar distortion.

---

## 5. Running the FastAPI Server

Start the development server (CORS is configured for `http://localhost:5173` by default):

```bash
uvicorn main:app --reload --port 8000
```

---

## 6. API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | **Full Site Analysis.** Evaluates terrain, 360° horizon profile, live JPL Horizons Sun & Earth ephemeris over `[time_start, time_end]`, exact `s_sun`, `s_earth`, `s_safety`, and `s_science` scores, plus per-step latency timings. |
| `GET` | `/api/site/inspect` | **Fast Site Inspection.** Returns terrain metadata plus `s_safety` and `s_science` without making external JPL Horizons API calls. |
| `GET` | `/api/site/route-map` | **2D Rover Route Map.** Returns an in-memory `image/png` hillshaded DEM patch with the true-to-scale landing dispersion zone, candidate SSIs, optimal rover path, and dynamic scale bar. Exposes `Server-Timing` and `X-Timings-Ms` headers. |
| `GET` | `/api/sites/scores` | **Batch MCDM Ranking.** Evaluates all precomputed sites locally using normalised weights (`w_safety`, `w_science`, `w_sun`, `w_earth`) and returns the top `top_n` (default `5`) candidate sites. |
| `GET` | `/api/ai` | **AI Mission Assistant.** Accepts a natural-language `query` describing mission constraints and returns Gemini-generated `weights`, recommended `(lat, lon)` `coords`, and `justifications`. |

### Standalone Route Map CLI
You can also test the 2D route renderer directly from the command line:

```bash
python api/analysis/render_route_map.py --lat -89.68 --lon -130.0 --range-km 12 --size-px 300 --out output/shackleton_route.png
```

---

## 7. Performance & Caching
* **Binary Horizon Cache:** On first lookup, `inspect_point.py` compiles `landing_site_horizons.csv` into `landing_site_horizons_cache.npz` (automatically invalidated if the CSV is modified) for near-instantaneous memory-mapped lookups.
* **Custom Site Persistence:** Newly inspected user coordinates outside the $0.15\text{ km}$ deduplication radius are computed on demand from the DEM and appended to `user_inspected_sites.csv` and `user_horizon_profiles.csv` for future cache hits.
* **Persistent Raster Handles:** Open `rasterio` dataset handles are cached in memory and cleanly closed on process exit via `atexit`.