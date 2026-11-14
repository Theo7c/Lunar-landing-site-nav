# Backend — Lunar Landing Site Navigator & MCDM Engine

Python data-processing pipeline and FastAPI service powering the **Lunar Landing Site Navigator & Multi-Criteria Decision-Making (MCDM) Engine**.

The backend handles two core workflows:
1. **Offline Geospatial Data Pipeline:** Processes raw NASA LOLA elevation rasters and Permanently Shadowed Region (PSR) catalogs into relational 3D-indexed CSV databases (`landing_sites.csv` and `ssis.csv`).
2. **Real-Time Spatial MCDM & On-Demand NASA Streaming API:** Ranks tens of thousands of candidate landing areas for any user-chosen rover travel radius ($R_{\max}$) in `<10 ms`, and streams native `20 m/px` NASA GeoTIFF tiles plus live PDS spacecraft observations on demand when a user clicks an arbitrary point on the 3D Moon.

---

## Directory Structure

    Lunar-landing-site-nav/
    ├── build_ssis.py             # Generates processed_data/ssis.csv (1,531 polar PSRs + 28 curated targets)
    ├── build_landing_sites.py    # Generates processed_data/landing_sites.csv (South Pole + Global + Artemis patches)
    ├── inspect_point.py          # On-demand NASA 20 m/px /vsicurl/ streamer, PDS API query & read-through cache
    ├── main.py                   # FastAPI application exposing /api/rank and /api/inspect endpoints
    ├── requirements.txt          # Python dependencies
    ├── .gitignore                # Excludes raw_data/ from version control
    ├── raw_data/                 # (Git-ignored) Raw NASA LOLA GeoTIFFs and PSR CSVs
    └── processed_data/           # Generated relational databases & schema README
        ├── README.md
        ├── landing_sites.csv
        ├── ssis.csv
        └── user_inspected_sites.csv

---

## Setup & Installation

### 1. Create a Virtual Environment & Install Dependencies

    python -m venv .venv
    # On Windows (Git Bash):
    source .venv/Scripts/activate
    # On macOS/Linux:
    # source .venv/bin/activate

    pip install fastapi uvicorn numpy pandas scipy rasterio requests

*(Note: On Windows, if `rasterio` fails to build from source, install via Conda with `conda install -c conda-forge rasterio` or use prebuilt wheels.)*

### 2. Raw Data Placement (`raw_data/`)
Ensure the following NASA datasets are placed inside `raw_data/` before running the build scripts:
* `raw_data/south_pole/LDEM_80S_80MPP_ADJ.tiff` — NASA PGDA `80 m/px` South Pole Stereographic DEM
* `raw_data/south_pole/LPSR_80S_20MPP_ADJ_1km2.csv` — NASA PGDA Permanently Shadowed Region WKT catalog
* `raw_data/ldem_16.tif` — NASA SVS Global Equirectangular LOLA DEM (`~1.9 km/px`, elevation in km)

---

## Data Generation Pipeline

Run the two preprocessing scripts in order to build the relational databases inside `processed_data/`:

    # 1. Build the unified Sites of Scientific Interest (SSI) catalog
    python build_ssis.py

    # 2. Build the feasible landing areas grid (hard-filtered to max_slope_deg <= 15.0 deg)
    python build_landing_sites.py

* **Teammate Ray-Tracing Integration:** `build_landing_sites.py` initializes `f_sun` and `f_earth` as `NaN`. Your teammate's horizon ray-tracing script populates these two columns in `processed_data/landing_sites.csv` using `LDEM_80S_80MPP_ADJ.tiff` (`tier == "south_pole"`) and `ldem_16.tif` (`tier == "global"`).
* **Self-Expanding Custom Cache:** Any valid coordinate inspected via `inspect_point.py` is automatically deduplicated (`150 m` radius) and appended to `processed_data/user_inspected_sites.csv` without overwriting the baseline grid.

---

## Running the FastAPI Server

Start the local API server with hot-reload:

    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Interactive Swagger documentation is available automatically at `http://localhost:8000/docs`.

---

## API Endpoints

### `GET /api/rank` — Dynamic Spatial MCDM Ranking
Computes live Euclidean distances between all candidate landing areas (`landing_sites.csv` + `user_inspected_sites.csv`) and all targets in `ssis.csv` within the user's chosen travel radius, returning the top-ranked sites.

| Query Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `r_max_km` | `float` | `10.0` | Maximum rover/EVA travel distance from touchdown site (km). |
| `w_safety` | `float` | `0.25` | Weight for terrain safety ($S_{\text{safety}}$). |
| `w_power` | `float` | `0.25` | Weight for solar illumination duty cycle ($f_{\text{sun}}$). |
| `w_comms` | `float` | `0.25` | Weight for Direct-to-Earth visibility ($f_{\text{earth}}$). |
| `w_science` | `float` | `0.25` | Weight for reachable scientific value ($S_{\text{science}}$). |
| `tier` | `string` | `"all"` | Filter by `"all"`, `"south_pole"`, or `"global"`. |
| `limit` | `int` | `50` | Number of top-ranked sites to return. |

### `GET /api/inspect` — Exact On-Demand NASA Coordinate Inspection
Calls `inspect_exact_lunar_point()` in `inspect_point.py` for any clicked `(lat, lon)` coordinate:
1. Streams only the required raster tile from NASA's public `20 m/px` (`LDEM_80S_20MPP_ADJ.TIF`) or global GeoTIFF via GDAL `/vsicurl/` HTTP Range Requests.
2. Queries NASA's live PDS Orbital Data Explorer REST API (`oderest.rsl.wustl.edu`) for overlapping LRO instrument products.
3. Evaluates reachable SSIs within `r_max_km` and caches valid touchdown points (`max_slope_deg <= 15.0°`) into `processed_data/user_inspected_sites.csv`.

| Query Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `lat` | `float` | *Required* | Target latitude (`-90.0` to `90.0`). |
| `lon` | `float` | *Required* | Target longitude (`-180.0` to `180.0`). |
| `r_max_km` | `float` | `10.0` | Rover travel radius for local SSI reachability (km). |
| `save_to_db` | `bool` | `true` | Persist valid points into `user_inspected_sites.csv`. |

---

## Architecture & Fallback Notes

* **Offline / Frontend Fallback:** If the Next.js frontend is deployed standalone on Vercel without a live Python backend, it falls back seamlessly to static JSON exports of `landing_sites.csv` and `ssis.csv` and executes the identical 3D Euclidean MCDM scoring loop client-side in TypeScript.
* **Serverless Filesystem Safety:** `inspect_point.py` wraps disk appends to `user_inspected_sites.csv` in a `try/except OSError` block so read-only cloud environments still return live NASA `/vsicurl/` and PDS API results without failing.