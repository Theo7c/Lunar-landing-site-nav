# Lunar Terrain Inspection & Runtime Scoring Architecture

This document covers the analytical pipeline across **`api/analysis/inspect_point.py`**, **`api/analysis/scores_calculator.py`**, and **`api/analysis/score_calculator.py`**.

---

## 1. System Overview & File Responsibilities

| Module | Role | Inputs | Outputs |
| :--- | :--- | :--- | :--- |
| `api/analysis/inspect_point.py` | On-demand DEM terrain & 360° horizon extractor for any `(lat, lon)` click | Local DEMs (`raw_data/`) or NASA `/vsicurl/` fallback | Appends to `user_inspected_sites.csv` & `user_horizon_profiles.csv`; returns `(site_row[12], horizons[360])` |
| `api/analysis/scores_calculator.py` | High-speed batch MCDM scorer across the entire candidate database (~60,000+ sites) | `landing_sites.csv`, `user_inspected_sites.csv`, `ssis.csv`, lander/rover constraints | `DataFrame` with `feasible`, `s_safety`, `reachable_ssi_count`, `s_science` |
| `api/analysis/score_calculator.py` | Single-point scorer for custom inspected sites | 12-element `site_row` list, `ssis.csv`, lander/rover constraints | `(s_safety, s_science)` tuple in `[0.0, 1.0]` |

---

## 2. Dynamic Point Inspection (`inspect_point.py`)

### A. Local-First Raster Resolution & Handle Caching
To avoid network latency on repeated map clicks, `get_dataset()` prioritizes local GeoTIFFs (`raw_data/south_pole/LDEM_80S_80MPP_ADJ.tiff` for $\text{lat} \le -80^\circ$, `raw_data/ldem_64.tif` or `raw_data/ldem_16.tif` for global coordinates) and caches open `rasterio.DatasetReader` handles in memory (`_OPEN_DATASETS`). If local files are absent, it seamlessly streams byte ranges via GDAL `/vsicurl/`.

### B. Single-Window Terrain & Horizon Extraction
For each `(lat, lon)`, a single `Window` covering $\pm 80\text{ km}$ around the target pixel is read from the DEM:

1. **Inner Touchdown Dispersion Patch (12 Static Columns):**
   * **South Pole ($\text{lat} \le -80^\circ$, `80 m/px` stereographic):** Uses a `250 m` radius window (`7 × 7` px) with uniform grid spacing $\Delta x = \Delta y = 80\text{ m}$.
   * **Global ($\text{lat} > -80^\circ$, `~474 m/px` equirectangular):** Uses a `3 × 3` pixel window (`~710 m` radius) with latitude-compensated longitudinal spacing $\Delta x = R_{\text{Moon}} \cos(\text{lat}) \, \Delta\lambda$.
   * Computes `elevation_m` (mean), `max_slope_deg`, `mean_slope_deg`, `roughness_m` (elevation standard deviation $\sigma_z$), and Moon-centered 3D Cartesian coordinates $(x_{\text{km}}, y_{\text{km}}, z_{\text{km}})$ with $R_{\text{Moon}} = 1737.4\text{ km}$.

2. **360° Horizon Elevation Profile (`az_000` to `az_359`):**
   * Casts 360 rays at $1^\circ$ azimuth increments ($\psi \in [0^\circ, 359^\circ]$) out to $R_{\text{horizon}} = 80\text{ km}$, using dense radial sampling near the lander (`150–250 m` steps under `5 km`) and coarser steps (`1,250 m`) out to `80 km`.
   * On equirectangular rasters, ray coordinates follow exact forward spherical geodesics to account for meridian convergence.
   * For each terrain sample at surface distance $d$ and elevation $z_1$ relative to observer elevation $z_0$ ($r_0 = R + z_0$, $r_1 = R + z_1$, $R = 1,737,400\text{ m}$), the exact tangent-plane elevation angle is:
     $$\alpha(d, \psi) = \operatorname{atan2}\!\Bigl(r_1 \cos(d/R) - r_0,\;\; r_1 \sin(d/R)\Bigr)$$
   * Taking the maximum across all radial steps yields the horizon obstruction angle $H(\psi) = \max_{d \le 80\text{ km}} \alpha(d, \psi)$ in degrees.

### C. Deduplication & Persistence
* Before writing to disk, the new $(x_{\text{km}}, y_{\text{km}}, z_{\text{km}})$ coordinate is checked against all existing sites in `landing_sites.csv` and `user_inspected_sites.csv`.
* If an existing site lies within `150 m` (`dedup_radius_km=0.15`), disk writes are skipped and the existing site's 12-element row is returned alongside the newly computed 360-element horizon list.
* Otherwise, a new `USER_XXXXXX` ID is generated and appended to:
  * `processed_data/user_inspected_sites.csv` (12 static terrain columns)
  * `processed_data/user_horizon_profiles.csv` (`site_id` + 360 columns `az_000`..`az_359`)

---

## 3. Site-Independent Score Calculation (`scores_calculator.py` & `score_calculator.py`)

Both modules evaluate every site strictly from its own terrain geometry, nearby Scientific Sites of Interest (`ssis.csv`), and the user's lander/rover hardware tolerances—**without** normalizing against other sites in the grid.

### A. Landing Safety Score ($S_{\text{safety}} \in [0.0, 1.0]$)

1. **Feasibility Gate:**
   $$\text{feasible}(x) = \bigl(\theta_{\max}(x) \le \theta_{\text{max\_limit}}\bigr) \;\land\; \bigl(\theta_{\text{mean}}(x) \le \theta_{\text{mean\_limit}}\bigr)$$
   If `feasible(x)` is `False`, $S_{\text{safety}}(x) = 0.0$ (and `calculate_scores()` filters the site out when `filter_infeasible=True`).

2. **Linear Margin Combination:**
   $$T_{\text{slope}}(x) = \text{clip}\!\left(1 - \frac{\theta_{\max}(x)}{\theta_{\text{max\_limit}}},\; 0,\; 1\right), \qquad T_{\text{rough}}(x) = \text{clip}\!\left(1 - \frac{\sigma_z(x)}{\sigma_{\text{limit}}},\; 0,\; 1\right)$$
   $$S_{\text{safety}}(x) = \alpha \, T_{\text{slope}}(x) + (1 - \alpha) \, T_{\text{rough}}(x)$$

### B. Scientific Accessibility Score ($S_{\text{science}} \in [0.0, 1.0]$)

1. **3D `cKDTree` Spatial Range Query:**
   Both `scores_calculator.py` and `score_calculator.py` index `ssis.csv` in a 3D `scipy.spatial.cKDTree` over $(x_{\text{km}}, y_{\text{km}}, z_{\text{km}})$ to retrieve all targets within rover/EVA sortie range $R_{\max}$ (`query_ball_point`).

2. **Inverse-Distance Decay & Rational Saturation:**
   Each reachable SSI $i$ ($d_i \le R_{\max}$) contributes its intrinsic value $V_i \in [1.0, 5.0]$ discounted by travel distance:
   $$R_{\text{raw}}(x) = \sum_{d_i(x) \le R_{\max}} \frac{V_i}{1 + k \, d_i(x)}$$
   $R_{\text{raw}}(x)$ is mapped smoothly into $[0.0, 1.0)$ via Michaelis-Menten / rational saturation with half-saturation constant $C$ (`science_half_sat`):
   $$S_{\text{science}}(x) = \frac{R_{\text{raw}}(x)}{R_{\text{raw}}(x) + C}$$
   * $R_{\text{raw}} = 0 \implies S_{\text{science}} = 0.0$
   * $R_{\text{raw}} = C \implies S_{\text{science}} = 0.5$
   * High-density science clusters asymptotically approach `1.0` without hard clipping, preserving differentiation between strong and exceptional sites.

### C. Configurable Mission Parameters

| Parameter | Default | Criterion | Description |
| :--- | :---: | :--- | :--- |
| `max_slope_limit_deg` | `15.0°` | Safety | Hard maximum single-pixel slope cutoff inside touchdown dispersion box. |
| `mean_slope_limit_deg` | `10.0°` | Safety | Hard maximum average slope cutoff across the landing patch. |
| `roughness_limit_m` | `25.0 m` | Safety | Local elevation standard deviation ($\sigma_z$) where $T_{\text{rough}}$ reaches `0.0`. |
| `slope_weight` ($\alpha$) | `0.60` | Safety | Convex weight between worst-case slope (`60%`) and roughness (`40%`). |
| `r_max_km` ($R_{\max}$) | `12.0 km` | Science | Maximum rover or EVA sortie radius from the landing site. |
| `decay_rate` ($k$) | `0.15 km⁻¹` | Science | Distance travel-cost penalty factor. |
| `science_half_sat` ($C$) | `5.0` | Science | Raw science value $R_{\text{raw}}$ that yields $S_{\text{science}} = 0.50$. |