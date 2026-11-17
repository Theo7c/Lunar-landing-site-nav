# Lunar Landing Site Navigator (`Lunar-landing-site-nav`)

**Lunar Navigator** is an interactive lunar mission planning, geospatial terrain analysis, orbital ephemeris, and Multi-Criteria Decision-Making (MCDM) platform designed to identify, evaluate, and compare candidate lunar landing sites and rover traverse routes.

---

## Official Project Links & Tutorial

* **Live Web Application:** [https://lunar.isaacbarker.net](https://lunar.isaacbarker.net)
* **Presentation & Visual Tutorial:** [Lunar Landing Navigator Presentation](https://docs.google.com/presentation/d/1L2ga50d0BYGZ_xTR7sy67_gp1S9Lelfc/edit?usp=sharing&ouid=114225838610489975589&rtpof=true&sd=true) — *Provides a complete visual walkthrough and user tutorial covering mission epoch selection, AI-assisted weight generation, interactive 3D globe navigation, on-demand coordinate inspection, and site comparison telemetry cards.*
* **GitHub Repository:** [https://github.com/Theo7c/Lunar-landing-site-nav](https://github.com/Theo7c/Lunar-landing-site-nav)

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Complete Repository Structure](#2-complete-repository-structure)
3. [Comprehensive Science & Mathematical Report](#3-comprehensive-science--mathematical-report)
   - [3.1 Lunar Geodesy & 3D Coordinate Transformations](#31-lunar-geodesy--3d-coordinate-transformations)
   - [3.2 Remote DEM Streaming & Topographic Hazard Mechanics](#32-remote-dem-streaming--topographic-hazard-mechanics)
   - [3.3 Landing Safety Scoring Model](#33-landing-safety-scoring-model)
   - [3.4 360° Spherical Geometry Horizon Ray-Casting](#34-360-spherical-geometry-horizon-ray-casting)
   - [3.5 Orbital Ephemeris & Line-of-Sight (LoS) Visibility Physics](#35-orbital-ephemeris--line-of-sight-los-visibility-physics)
   - [3.6 3D Spatial Indexing & Scientific Accessibility Model](#36-3d-spatial-indexing--scientific-accessibility-model)
   - [3.7 Rover Traverse Optimization & Lambertian Hillshade Rendering](#37-rover-traverse-optimization--lambertian-hillshade-rendering)
   - [3.8 Multi-Criteria Decision-Making (MCDM) & Gemini AI Weighting](#38-multi-criteria-decision-making-mcdm--gemini-ai-weighting)
   - [3.9 3D WebGL Globe Displacement & Raycasting Mathematics](#39-3d-webgl-globe-displacement--raycasting-mathematics)
4. [Offline Data Generation Pipeline](#4-offline-data-generation-pipeline)
5. [Performance, Caching & Persistence Architecture](#5-performance-caching--persistence-architecture)
6. [Detailed File-by-File Codebase Reference](#6-detailed-file-by-file-codebase-reference)
   - [6.1 Backend & Scientific Analysis Engine (`api/`)](#61-backend--scientific-analysis-engine-api)
   - [6.2 Interactive 3D Web Client (`frontend/`)](#62-interactive-3d-web-client-frontend)
7. [Data Schemas (`processed_data/`)](#7-data-schemas-processed_data)
8. [REST API Specification](#8-rest-api-specification)

---

## 1. System Architecture Overview

The platform is engineered around three tightly coupled subsystems:

1. **Offline Geospatial Data Generation Pipeline:**
   Ingests high-resolution **NASA LRO LOLA (Lunar Orbiter Laser Altimeter) Digital Elevation Models (DEMs)** and planetary feature catalogs to pre-compute baseline candidate landing site metrics, 360° topographic horizon elevation masks, and **Sites of Scientific Interest (SSIs)**.
2. **Real-Time FastAPI Scientific Backend (`api/`):**
   Hosted on **Render**, the backend executes vectorized MCDM batch rankings, $O(\log N)$ 3D `scipy.spatial.cKDTree` spatial range queries, live **NASA JPL Horizons** topocentric ephemeris visibility integration, **Google Gemini** natural-language mission parameter synthesis, in-memory **Matplotlib** hillshaded rover traverse rendering, and on-demand **GDAL `/vsicurl/`** HTTP byte-range DEM streaming for arbitrary user-selected lunar coordinates.
3. **Interactive 3D Globe Frontend (`frontend/`):**
   Hosted on **Vercel**, the **Svelte 5 + Vite + Three.js** client renders a displacement-mapped 3D lunar globe, interactive raycast coordinate picking, lander hardware preset selectors, mission epoch sliders, and side-by-side site telemetry cards.

---

## 2. Complete Repository Structure

```text
Lunar-landing-site-nav/
├── .gitignore                                      # Root Git ignore rules (Python bytecode, venv, OS files)
├── LICENSE                                         # Open-source project license
├── README.md                                       # Master project, architecture & mathematical report
├── Lunar Landing Navigator (1).pptx                # Official presentation slide deck & visual user tutorial
├── processed_data/
│   ├── landing_site_horizons.csv                   # Pre-computed 360° horizon profiles for baseline grid sites
│   └── landing_site_horizons_cache.npz             # Auto-compiled binary NumPy cache for instant horizon loading
├── api/                                            # FastAPI Backend & Geospatial/Ephemeris Engine (Render)
│   ├── __init__.py                                 # Python package initializer
│   ├── main.py                                     # FastAPI entry point, CORS setup, cache init & REST routes
│   ├── render.yaml                                 # Render.com cloud deployment Blueprint specification
│   ├── requirements.txt                            # Python dependencies (FastAPI, GDAL, SciPy, NumPy, Matplotlib, GenAI)
│   ├── processed_data/
│   │   ├── user_horizon_profiles.csv               # Persistent cache of 360° horizons for custom user-inspected points
│   │   └── user_inspected_sites.csv                # Persistent cache of 12-element terrain metadata for custom points
│   └── analysis/                                   # Core Mathematical, Physics, Geospatial & Rendering Modules
│       ├── AstronomicalScore.py                    # JPL Horizons API client & Sun/Earth LoS visibility integrator
│       ├── GeminiAPICaller.py                      # Google Gemini LLM client for natural-language mission weighting
│       ├── horizon.py                              # 360° azimuthal terrain ray-casting & spherical horizon solver
│       ├── inspect_point.py                        # On-demand GDAL /vsicurl/ DEM streamer & terrain hazard extractor
│       ├── landing_site_scoring_methodology.md     # Formal documentation of the MCDM & terrain scoring models
│       ├── notes.md                                # Engineering notes, NASA LOLA data sources & pipeline logs
│       ├── render_route_map.py                     # In-memory Matplotlib hillshade DEM & rover route map renderer
│       ├── score_calculator.py                     # Single-site safety and scientific accessibility evaluator
│       ├── scores_calculator.py                    # Vectorized batch MCDM ranker, cKDTree query engine & route solver
│       └── test.py                                 # Diagnostic & verification scripts for analysis pipelines
└── frontend/                                       # Svelte 5 + Vite + Three.js Interactive Web Client (Vercel)
    ├── .gitignore                                  # Frontend Git ignore rules (node_modules, build artifacts)
    ├── .vscode/
    │   └── extensions.json                         # Recommended VS Code workspace extensions (Svelte)
    ├── README.md                                   # Frontend-specific build & dev server documentation
    ├── index.html                                  # Root HTML document & viewport mount container
    ├── jsconfig.json                               # JavaScript/Svelte language server & path configuration
    ├── package-lock.json                           # Locked npm dependency tree
    ├── package.json                                # Frontend dependencies (Svelte 5, Three.js, Vite)
    ├── svelte.config.js                            # Svelte compiler & Vite preprocessor configuration
    ├── vite.config.js                              # Vite bundler & development server configuration
    ├── public/
    │   ├── clock.svg                               # Timeline clock vector icon
    │   ├── ldem_3_8bit.jpg                         # NASA LRO LOLA global lunar elevation displacement map texture
    │   └── lroc_color_poles_1k.jpg                 # NASA LROC WAC global color albedo map texture
    └── src/
        ├── App.svelte                              # Root application layout coordinating 3D globe, sidebar & modals
        ├── app.css                                 # Global stylesheet, dark space UI variables & typography
        ├── main.js                                 # Svelte 5 DOM mount entry point
        ├── assets/                                 # UI SVG icons for scoring dimensions and interactive controls
        │   ├── cross-icn.svg                       # Close/dismiss modal icon
        │   ├── earth-icn.svg                       # Earth DTE communication metric icon
        │   ├── earth.svg                           # Earth status badge graphic
        │   ├── reset-icn.svg                       # Parameter & camera reset icon
        │   ├── safety.svg                          # Landing safety metric icon
        │   ├── science.svg                         # Scientific accessibility metric icon
        │   ├── sun-icn.svg                         # Solar illumination metric icon
        │   ├── sun.svg                             # Sun status badge graphic
        │   └── total.svg                           # Composite MCDM total score icon
        ├── lib/                                    # Reusable UI & 3D WebGL Visualization Components
        │   ├── Moon.svelte                         # Three.js 3D lunar globe, displacement shaders, raycaster & pins
        │   ├── ParameterSelector.svelte            # Sliders for MCDM weights, lander limits, presets & AI prompt box
        │   ├── SiteCard.svelte                     # Detailed site telemetry modal, score bars & 2D route map viewer
        │   ├── SiteSelector.svelte                 # Live-ranked candidate list & custom coordinate inspection input
        │   └── TimeSlider.svelte                   # Mission epoch window selector for ephemeris calculations
        └── shared/                                 # Reactive Svelte 5 State Stores & Static Hardware/Site Catalogs
            ├── config.svelte.js                    # Backend API base URLs and environment configuration
            ├── inputs.svelte.js                    # Reactive ($state) mission weights, lander constraints & epochs
            ├── lander.js                           # Preset lander/rover hardware profiles (slope, roughness, range)
            ├── landing_sites.js                    # Baseline candidate landing sites catalog
            └── sites.svelte.js                     # Reactive store managing active sites, rankings & API fetches
```

---

## 3. Comprehensive Science & Mathematical Report

This section details the exact physical models, geodetic projections, numerical approximations, and optimization algorithms implemented across the codebase.

### 3.1 Lunar Geodesy & 3D Coordinate Transformations

The Moon is modeled as a reference sphere of mean volumetric radius:

$$R_{\text{moon}} = 1,737.4\text{ km} = 1,737,400\text{ m}$$

#### Selenographic to 3D Cartesian (LCLF) Projection
Standard 2D equirectangular metrics $(\Delta\phi, \Delta\lambda)$ suffer from severe meridian convergence near the lunar poles ($\cos\phi \to 0$ as $\phi \to \pm 90^\circ$) and wrap-around discontinuities at $\lambda = \pm 180^\circ$. To eliminate polar singularities, `scores_calculator.py` projects all selenographic latitudes $\phi \in [-90^\circ, 90^\circ]$ and longitudes $\lambda \in [-180^\circ, 180^\circ]$ (converted to radians) onto a 3D Lunar-Centered Lunar-Fixed (LCLF) Cartesian manifold $(X, Y, Z) \in \mathbb{R}^3$:

$$X = R_{\text{moon}} \cos(\phi)\cos(\lambda)$$

$$Y = R_{\text{moon}} \cos(\phi)\sin(\lambda)$$

$$Z = R_{\text{moon}} \sin(\phi)$$

#### Exact Conversion Between Geodesic Arc Length and 3D Chord Distance
Let $d_{\text{surface}}$ be the true great-circle surface traverse distance (in kilometers) between two points $\mathbf{p_1}, \mathbf{p_2}$ on the lunar sphere. The central angle $\Delta\sigma$ (in radians) subtended at the Moon's center is:

$$\Delta\sigma = \frac{d_{\text{surface}}}{R_{\text{moon}}}$$

By constructing an isosceles triangle with two sides of length $R_{\text{moon}}$ and included angle $\Delta\sigma$, the exact 3D Euclidean straight-line chord length $d_{\text{chord}} = \lVert \mathbf{p_1} - \mathbf{p_2} \rVert_2$ through the lunar interior is:

$$d_{\text{chord}} = 2 R_{\text{moon}} \sin\left(\frac{d_{\text{surface}}}{2 R_{\text{moon}}}\right)$$

Because $\sin(\Delta\sigma / 2)$ is strictly monotonically increasing on $\Delta\sigma \in [0, \pi]$, a spherical cap of geodesic radius $R_{\max}$ on the lunar surface is mathematically identical to the intersection of the lunar sphere with a 3D Euclidean ball of radius $d_{\text{chord}}(R_{\max})$. Once points inside the ball are retrieved via 3D `cKDTree`, their Euclidean chord distances $d_{\text{chord}}$ are inverted back into exact surface geodesic distances $d_{\text{surface}}$ via:

$$d_{\text{surface}} = 2 R_{\text{moon}} \arcsin\left(\frac{d_{\text{chord}}}{2 R_{\text{moon}}}\right)$$

#### Map Projections & GDAL Affine Geotransforms
When `inspect_point.py` and `render_route_map.py` sample NASA LRO LOLA DEM rasters (such as the $20\text{ m/px}$ South Polar Stereographic DEMs or Equirectangular cylindrical tiles), geographic coordinates $(\phi, \lambda)$ are transformed via `osgeo.osr.CoordinateTransformation` into planar map coordinates $(x_{\text{proj}}, y_{\text{proj}})$ in meters.

Each GeoTIFF/JP2 raster defines a 6-element affine geotransform vector $\mathcal{G} = [x_0, \Delta x, R_x, y_0, R_y, \Delta y]$, where $(x_0, y_0)$ is the top-left corner origin, $\Delta x > 0$ is the West-East pixel width, $\Delta y < 0$ is the North-South pixel height, and $R_x = R_y = 0$ for north-up/polar-aligned rasters. The forward mapping from pixel indices $(\text{col}, \text{row})$ to projected planar coordinates is:

$$x_{\text{proj}} = x_0 + \text{col} \cdot \Delta x + \text{row} \cdot R_x, \qquad y_{\text{proj}} = y_0 + \text{col} \cdot R_y + \text{row} \cdot \Delta y$$

Inverting this linear system yields the exact raster array indices $(\text{row}, \text{col})$ for any lunar coordinate:

$$\text{col} = \left\lfloor \frac{x_{\text{proj}} - x_0}{\Delta x} \right\rfloor, \qquad \text{row} = \left\lfloor \frac{y_{\text{proj}} - y_0}{\Delta y} \right\rfloor$$

---

### 3.2 Remote DEM Streaming & Topographic Hazard Mechanics

When inspecting an arbitrary coordinate $(\phi_0, \lambda_0)$, `inspect_point.py` uses GDAL's `/vsicurl/` virtual filesystem driver (`gdal.Open('/vsicurl/https://...')`) and `band.ReadAsArray(xoff, yoff, xsize, ysize)` to issue HTTP `Range` requests. This streams only the small $N \times N$ pixel subarray surrounding $(\text{col}, \text{row})$ directly into RAM over HTTPS.

To capture terrain hazards across both spacecraft touchdown and guidance scales, two concentric windows are evaluated:
1. **Local Window ($\Omega_{\text{local}}$):** Represents the immediate lander footprint and rocket plume interaction zone (e.g., $60\text{--}100\text{ m}$ radius).
2. **Regional Window ($\Omega_{\text{regional}}$):** Represents the 3-sigma guidance navigation & control (GN&C) landing dispersion ellipse and initial rover egress zone (e.g., $500\text{--}1,000\text{ m}$ radius).

#### 2D Finite-Difference Surface Slope Calculation
Let $z(i, j)$ be the elevation in meters at row $i$, column $j$ of a DEM window with uniform spatial resolution $\Delta s = \vert{}\Delta x\vert{} = \vert{}\Delta y\vert{}$ (in meters/pixel). `inspect_point.py` computes the orthogonal spatial partial derivatives using a second-order central finite-difference stencil (`numpy.gradient`) at interior nodes (and first-order forward/backward differences at window boundaries):

$$\frac{\partial z}{\partial x}(i, j) \approx \frac{z(i, j+1) - z(i, j-1)}{2\Delta s}, \qquad \frac{\partial z}{\partial y}(i, j) \approx \frac{z(i+1, j) - z(i-1, j)}{2\Delta s}$$

The magnitude of the surface gradient vector $\nabla z(i, j)$ represents the tangent of the local tilt angle:

$$\lVert \nabla z(i, j) \rVert_2 = \sqrt{\left(\frac{\partial z}{\partial x}(i, j)\right)^2 + \left(\frac{\partial z}{\partial y}(i, j)\right)^2}$$

The local topographic slope angle $\theta(i, j)$ in degrees is therefore:

$$\theta(i, j) = \arctan\left(\sqrt{\left(\frac{\partial z}{\partial x}(i, j)\right)^2 + \left(\frac{\partial z}{\partial y}(i, j)\right)^2}\right) \times \frac{180^\circ}{\pi}$$

From the 2D slope matrix over window $\Omega$, the code extracts the **mean slope** $\bar{\theta_{\Omega}}$ and the **maximum (worst-case) slope** $\theta_{\max, \Omega}$:

$$\bar{\theta_{\Omega}} = \frac{1}{\vert{}\Omega\vert{}} \sum_{(i,j)\in\Omega} \theta(i, j), \qquad \theta_{\max, \Omega} = \max_{(i,j)\in\Omega} \theta(i, j)$$

#### Terrain Roughness (RMS Elevation Dispersion)
Terrain roughness $\sigma_{\Omega}$ (in meters) quantifies vertical surface irregularity (boulders, small impact craters, hummocky ejecta) within window $\Omega$ as the root-mean-square (RMS) standard deviation of elevation residuals relative to the window's mean elevation $\bar{z_{\Omega}}$:

$$\bar{z_{\Omega}} = \frac{1}{\vert{}\Omega\vert{}} \sum_{(i,j)\in\Omega} z(i, j), \qquad \sigma_{\Omega} = \sqrt{\frac{1}{\vert{}\Omega\vert{}} \sum_{(i,j)\in\Omega} \big(z(i, j) - \bar{z_{\Omega}}\big)^2}$$

---

### 3.3 Landing Safety Scoring Model

Implemented in `score_calculator.py` and vectorized across all sites in `scores_calculator.py`, the **Landing Safety Score** ($S_{\text{safety}} \in [0, 100]$) evaluates whether a candidate site satisfies the mechanical stability limits of the selected lander hardware (`frontend/src/shared/lander.js`).

1. **Worst-Case Hazard Envelope:**
   To prevent a smooth regional average from masking a lethal local crater wall (or vice versa), the algorithm takes the conservative worst-case slope $\theta_{\text{worst}}$ and worst-case roughness $\sigma_{\text{worst}}$ across the site's telemetry vector:

$$\theta_{\text{worst}} = \max\left(\bar{\theta_{\text{local}}}, \bar{\theta_{\text{regional}}}, \theta_{\max, \text{local}}\right)$$

$$\sigma_{\text{worst}} = \max\left(\sigma_{\text{local}}, \sigma_{\text{regional}}\right)$$

2. **Hardware-Constrained Linear Penalty Functions:**
   Let $S_{\max}$ be the lander's maximum allowable touchdown tilt angle (in degrees, e.g., $5^\circ\text{--}15^\circ$) and $\sigma_{\max}$ be the lander's maximum allowable vertical roughness/clearance limit (in meters). Normalized dimensionless safety factors $f_{\text{slope}}, f_{\text{rough}} \in [0, 1]$ are computed via clamped linear decay:

$$f_{\text{slope}} = \text{clip}\left(1 - \frac{\theta_{\text{worst}}}{S_{\max}}, 0, 1\right) = \max\left(0, \min\left(1, 1 - \frac{\theta_{\text{worst}}}{S_{\max}}\right)\right)$$

$$f_{\text{rough}} = \text{clip}\left(1 - \frac{\sigma_{\text{worst}}}{\sigma_{\max}}, 0, 1\right) = \max\left(0, \min\left(1, 1 - \frac{\sigma_{\text{worst}}}{\sigma_{\max}}\right)\right)$$

3. **Composite Safety Aggregation:**
   The overall safety score $S_{\text{safety}} \in [0, 100]$ combines the slope and roughness survival factors:

$$S_{\text{safety}} = 100 \times \big(0.5 \, f_{\text{slope}} + 0.5 \, f_{\text{rough}}\big)$$

---

### 3.4 360° Spherical Geometry Horizon Ray-Casting

At lunar polar latitudes ($\vert{}\phi\vert{} > 80^\circ$), the Sun and Earth remain at grazing elevations near $0^\circ$. Furthermore, because the Moon's radius ($1,737.4\text{ km}$) is only $\sim 27\%$ of Earth's, planetary curvature causes rapid horizon drop-off ($\Delta h \approx \frac{d^2}{2 R_{\text{moon}}}$, or $\approx 720\text{ m}$ of vertical drop at $d = 50\text{ km}$). Flat-Earth tangent-plane ray-casting severely overestimates shadow lengths from distant massifs. `horizon.py` and `inspect_point.py` implement **exact spherical trigonometry ray-casting**.

```text
       Observer (r0 = R + z0 + h_mast)    Target Terrain Peak (r1 = R + z1)
                *  _ - - - - - - - - - - - - *
                | \   alpha (Elevation)     /
                |   \                      /
                |     *-------------------*  <-- Local Horizontal Tangent Plane (ENU)
                |    /                   /
             r0 |   /                   / r1
                |  /  gamma = d / R    /
                | /                   /
                *--------------------*
                    Lunar Center (0,0,0)
```

1. **Observer Radial Distance ($r_0$):**
   For an observer at surface elevation $z_0$ with a solar array or communication antenna mounted at height $h_{\text{mast}}$ above the surface, the radial distance from the Moon's center of mass is:

$$r_0 = R_{\text{moon}} + z_0 + h_{\text{mast}}$$

2. **Azimuthal Ray Marching:**
   For each discrete azimuth angle $\psi \in \lbrace 0^\circ, 1^\circ, 2^\circ, \dots, 359^\circ \rbrace$ (measured clockwise from local lunar North), a geodesic ray is marched outward in radial distance steps $d_m \in [d_{\min}, d_{\max}]$.
   In local planar polar stereographic coordinates centered at $(x_0, y_0)$ with local grid meridian convergence offset, step $(d_m, \psi)$ samples DEM coordinates:

$$x_m = x_0 + d_m \sin(\psi), \qquad y_m = y_0 + d_m \cos(\psi)$$

   yielding sampled terrain elevation $z_1(d_m, \psi)$ and target radial distance:

$$r_1(d_m, \psi) = R_{\text{moon}} + z_1(d_m, \psi)$$

3. **Exact Spherical Elevation Angle ($\alpha$):**
   A surface arc distance $d_m$ subtends a central angle $\gamma_m = \frac{d_m}{R_{\text{moon}}}$ (in radians) at the center of the Moon. In the 2D plane spanned by the lunar center, the observer, and the target point:
   - The component of $\mathbf{r_1}$ parallel to the observer's zenith vector $\hat{\mathbf{u_0}}$ is $r_1 \cos(\gamma_m)$, giving a vertical height difference relative to the observer's local horizontal plane of:

$$\Delta v = r_1 \cos(\gamma_m) - r_0$$

   - The component of $\mathbf{r_1}$ perpendicular to $\hat{\mathbf{u_0}}$ (parallel to the observer's local horizontal tangent plane) is:

$$\Delta h = r_1 \sin(\gamma_m)$$

   The exact apparent horizon elevation angle $\alpha(\psi, d_m)$ in degrees is therefore:

$$\alpha(\psi, d_m) = \text{atan2}\Big(r_1 \cos(\gamma_m) - r_0, r_1 \sin(\gamma_m)\Big) \times \frac{180^\circ}{\pi}$$

4. **Horizon Profile Envelope ($H(\psi)$):**
   Taking the maximum apparent elevation angle over all radial steps $d_m$ along azimuth $\psi$ yields the horizon mask $H(\psi)$:

$$H(\psi) = \max_{d_m \in [d_{\min}, d_{\max}]} \alpha(\psi, d_m)$$

   Repeating this across all 360 azimuths produces the 360-element horizon vector $\mathbf{H} = [H(0^\circ), H(1^\circ), \dots, H(359^\circ)]$.

---

### 3.5 Orbital Ephemeris & Line-of-Sight (LoS) Visibility Physics

`AstronomicalScore.py` evaluates real-time **Solar Illumination** (governing photovoltaic power generation and thermal survival) and **Earth Visibility** (governing Direct-to-Earth X/Ka-band radio communication without requiring lunar relay satellites) across any user-selected mission window $[t_{\text{start}}, t_{\text{end}}]$.

1. **NASA JPL Horizons Topocentric Ephemeris Query:**
   The module queries `https://ssd.jpl.nasa.gov/api/horizons.api` with parameters:
   - `CENTER='coord@301'` (topocentric coordinates on the Moon, NAIF ID `301`)
   - `SITE_COORD='lon,lat,elevation_km'`
   - `COMMAND='10'` (Sun) and `COMMAND='399'` (Earth)
   - `QUANTITIES='4'` (Apparent Azimuth $\psi_{\text{body}}$ and Elevation $\theta_{\text{body}}$ in degrees)
   - `START_TIME`, `STOP_TIME`, `STEP_SIZE`

2. **Azimuthal Horizon Lookup & Line-of-Sight Occlusion:**
   For each discrete ephemeris timestep $t_k$ ($k = 1, \dots, K$), the celestial body's apparent azimuth $\psi_{\text{body}}(t_k) \in [0^\circ, 360^\circ)$ is matched against the site's 360° horizon profile $H(\psi)$ (via nearest-bin indexing $\lfloor \psi_{\text{body}}(t_k) \rfloor \bmod 360$ or linear interpolation between adjacent $1^\circ$ bins).
   Using the Heaviside step function $\Theta(\cdot)$, the instantaneous Line-of-Sight visibility state $V_{\text{body}}(t_k) \in \lbrace 0, 1 \rbrace$ is:

$$
V_{\text{body}}(t_k) = \Theta\Big(\theta_{\text{body}}(t_k) - H\big(\psi_{\text{body}}(t_k)\big)\Big) = \begin{cases} 1 & \text{if } \theta_{\text{body}}(t_k) > H\big(\psi_{\text{body}}(t_k)\big) \quad \text{(Unobstructed LoS)} \\ 0 & \text{if } \theta_{\text{body}}(t_k) \le H\big(\psi_{\text{body}}(t_k)\big) \quad \text{(Terrain Occluded)} \end{cases}
$$

3. **Time-Integrated Visibility Scores ($S_{\text{sun}}, S_{\text{earth}}$):**
   Integrating $V_{\text{body}}(t_k)$ over all $K$ timesteps yields the percentage of total mission duration with direct solar illumination and direct Earth communication:

$$
S_{\text{sun}} = \frac{100}{K} \sum_{k=1}^{K} V_{\odot}(t_k), \qquad S_{\text{earth}} = \frac{100}{K} \sum_{k=1}^{K} V_{\oplus}(t_k)
$$

---

### 3.6 3D Spatial Indexing & Scientific Accessibility Model

Implemented in `scores_calculator.py`, the **Scientific Accessibility Score** ($S_{\text{science}} \in [0, 100]$) measures the cumulative, distance-weighted scientific value of all **Sites of Scientific Interest (SSIs)** reachable within the rover's maximum traverse radius $R_{\max}$ (in kilometers).

1. **3D `cKDTree` Ball Range Query:**
   All $N_{\text{SSI}}$ cataloged scientific targets are converted into 3D LCLF vectors $\mathbf{q_j} \in \mathbb{R}^3$ and indexed in a `scipy.spatial.cKDTree`. Each target $j$ carries an intrinsic scientific priority weight $V_j > 0$ (based on volatile/water-ice potential, PSR proximity, or geologic significance).
   Given a candidate landing site at $\mathbf{p_i} \in \mathbb{R}^3$ and maximum rover surface range $R_{\max}$, the code queries `tree.query_ball_point(p_i, d_chord(R_max))` to retrieve the reachable target set:

$$
\mathcal{N}_i(R_{\max}) = \left\lbrace j \in \lbrace 1, \dots, N_{\text{SSI}} \rbrace \mid d_{\text{surface}}(\mathbf{p_i}, \mathbf{q_j}) \le R_{\max} \right\rbrace
$$

2. **Linear Distance-Decayed Utility Accumulation:**
   Because rover energy consumption, wheel wear, and mission risk scale with traverse distance, targets located closer to the landing site contribute higher effective utility. For each reachable target $j \in \mathcal{N}_i(R_{\max})$ at surface geodesic distance $d_{ij} = d_{\text{surface}}(\mathbf{p_i}, \mathbf{q_j})$, the distance-decayed raw scientific utility $U_i$ is:

$$
U_i = \sum_{j \in \mathcal{N}_i(R_{\max})} V_j \left(1 - \frac{d_{ij}}{R_{\max}}\right)
$$

   Targets at the exact touchdown point ($d_{ij} = 0$) retain $100\%$ of their intrinsic value $V_j$, decaying linearly to $0$ at the maximum range boundary ($d_{ij} = R_{\max}$).

3. **Convex Quadratic Max-Scaling ($S_{\text{science}}$):**
   Let $U_{\max} = \max_{m \in \lbrace 1, \dots, M \rbrace} U_m$ be the highest raw scientific utility achieved across all $M$ candidate landing sites for the current $R_{\max}$. To create strong selective contrast that rewards elite scientific hubs (sites situated inside dense clusters of high-value targets) over isolated mediocre sites, raw utilities are mapped to $[0, 100]$ via a convex quadratic transformation:

$$
S_{\text{science}, i} = 100 \times \left(\frac{U_i}{U_{\max}}\right)^2
$$

---

### 3.7 Rover Traverse Optimization & Lambertian Hillshade Rendering

#### Prize-Collecting Orienteering Route Solver (`solve_best_rover_route`)
When a user inspects a specific landing site, `scores_calculator.py` plans an optimal multi-stop rover excursion starting at the lander $v_0 = \mathbf{p_{\text{site}}}$ and visiting an ordered sequence of reachable SSIs $\pi = (v_0, v_1, \dots, v_L)$ subject to the total traverse range budget $R_{\max}$. Mathematically, this is the **Prize-Collecting Orienteering Problem (PCOP)**:

$$
\max_{\pi = (v_0, v_1, \dots, v_L)} \sum_{k=1}^{L} V_{v_k} \quad \text{subject to} \quad \sum_{k=1}^{L} d_{\text{surface}}(v_{k-1}, v_k) \le R_{\max}
$$

The solver uses a greedy marginal-efficiency heuristic: from the current rover position $v_{\text{curr}}$ with remaining distance budget $B_{\text{rem}}$, it evaluates all unvisited candidate targets $j \in \mathcal{N}(R_{\max}) \setminus \pi$ satisfying $d_{\text{surface}}(v_{\text{curr}}, j) \le B_{\text{rem}}$ and selects the target maximizing scientific reward per distance cost:

$$
v_{\text{next}} = \arg\max_{j \notin \pi, d(v_{\text{curr}}, j) \le B_{\text{rem}}} \frac{V_j}{\big(d_{\text{surface}}(v_{\text{curr}}, j) + \epsilon\big)^\beta}
$$

updating $B_{\text{rem}} \leftarrow B_{\text{rem}} - d_{\text{surface}}(v_{\text{curr}}, v_{\text{next}})$ until no further targets can be reached within the budget.

#### Lambertian Hillshade Reflectance Mathematics (`render_route_map.py`)
To generate a visual 2D traverse map in memory, `render_route_map.py` extracts the local DEM window bounding the lander and reachable SSIs, and computes a shaded relief surface using **Lambert's Cosine Law** (`matplotlib.colors.LightSource`).
Given synthetic illumination azimuth $\psi_L = 315^\circ$ (North-West) and altitude angle $\theta_L = 45^\circ$ (zenith angle $Z_L = 90^\circ - \theta_L$), and local DEM partial derivatives $\left(\frac{\partial z}{\partial x}, \frac{\partial z}{\partial y}\right)$:
- The local surface slope angle in radians is:

$$
\theta_{\text{slope}} = \arctan\left(z_{\text{factor}}\sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2}\right)
$$

- The local down-slope aspect angle is:

$$
\psi_{\text{aspect}} = \text{atan2}\left(-\frac{\partial z}{\partial y}, -\frac{\partial z}{\partial x}\right)
$$

- The normalized Lambertian reflectance intensity $I(x, y) \in [0, 1]$ is the dot product between the unit surface normal $\hat{\mathbf{n}}$ and unit light vector $\hat{\mathbf{s}}$:

$$
I(x, y) = \hat{\mathbf{n}}(x,y) \cdot \hat{\mathbf{s}} = \sin(\theta_L)\cos(\theta_{\text{slope}}) + \cos(\theta_L)\sin(\theta_{\text{slope}})\cos(\psi_L - \psi_{\text{aspect}})
$$

The renderer composites the hillshade intensity with a hypsometric elevation colormap, overlays elevation contour lines, plots the SSI markers scaled by $V_j$ and the ordered traverse polyline $\pi$, and writes the figure directly into an in-memory `io.BytesIO` PNG stream.

---

### 3.8 Multi-Criteria Decision-Making (MCDM) & Gemini AI Weighting

#### Composite MCDM Score ($S_{\text{total}}$)
For every candidate site $i$, the four normalized sub-scores—Landing Safety ($S_{\text{safety}, i}$), Scientific Accessibility ($S_{\text{science}, i}$), Solar Illumination ($S_{\text{sun}, i}$), and Earth Communication ($S_{\text{earth}, i}$)—each span $[0, 100]$. Given user-defined preference weights $\mathbf{w} = [w_{\text{safety}}, w_{\text{science}}, w_{\text{sun}}, w_{\text{earth}}]^T \in [0, 1]^4$, `scores_calculator.py` computes the composite **Total Mission Score** $S_{\text{total}, i} \in [0, 100]$ via Normalized Weighted Linear Combination (WLC):

$$
S_{\text{total}, i} = \frac{w_{\text{safety}} S_{\text{safety}, i} + w_{\text{science}} S_{\text{science}, i} + w_{\text{sun}} S_{\text{sun}, i} + w_{\text{earth}} S_{\text{earth}, i}}{w_{\text{safety}} + w_{\text{science}} + w_{\text{sun}} + w_{\text{earth}}}
$$

*(If $\sum w_k = 0$, the denominator defaults to equal weights $w_k = 0.25$ to prevent division by zero).*

#### AI-Assisted Parameter Synthesis (`GeminiAPICaller.py`)
When a user submits a natural-language mission description in the UI, `GeminiAPICaller.py` calls the **Google Gemini API** (`google-genai`) with a structured JSON output schema. The model analyzes the mission's power architecture (e.g., solar vs. RTG), communication architecture (Direct-to-Earth vs. orbital relay), landing system hazard tolerance, and science objectives to output:
- Normalized weight vector $\mathbf{w} = (w_{\text{safety}}, w_{\text{science}}, w_{\text{sun}}, w_{\text{earth}})$
- Hardware constraint vector $(S_{\max}, \sigma_{\max}, R_{\max})$
- Engineering justification for the selected parameters.

---

### 3.9 3D WebGL Globe Displacement & Raycasting Mathematics

In `frontend/src/lib/Moon.svelte`, the lunar globe is rendered in **Three.js** (`THREE.SphereGeometry`) using NASA LRO textures:
1. **Vertex Displacement Mapping:**
   Each vertex on the unit sphere with normal $\hat{\mathbf{n}}$ and UV coordinates $(u, v)$ samples the 8-bit LOLA elevation map `ldem_3_8bit.jpg` ($D(u,v) \in [0, 1]$) and shifts its radial position along the surface normal:

$$
\mathbf{v_{\text{displaced}}} = \big(R_{\text{globe}} + s_{\text{disp}} \cdot (D(u, v) - b_{\text{disp}})\big) \hat{\mathbf{n}}
$$

   while mapping `lroc_color_poles_1k.jpg` to diffuse surface albedo.
2. **Selenographic Marker Placement & Inverse Raycasting:**
   Three.js uses a right-handed Y-up coordinate system (`Y` = North Pole, `X/Z` = Equatorial Plane). Converting $(\phi, \lambda)$ to 3D globe coordinates $(x_{\text{gl}}, y_{\text{gl}}, z_{\text{gl}})$ at radius $r$ follows:

$$
y_{\text{gl}} = r \sin(\phi), \qquad x_{\text{gl}} = r \cos(\phi)\cos(\lambda), \qquad z_{\text{gl}} = -r \cos(\phi)\sin(\lambda)
$$

   When a user clicks on the 3D globe, `THREE.Raycaster` computes the intersection point $\mathbf{p_{\text{hit}}} = (x_{\text{gl}}, y_{\text{gl}}, z_{\text{gl}})$ on the mesh and inverts the transformation to recover selenographic $(\phi, \lambda)$:

$$
\phi = \arcsin\left(\frac{y_{\text{gl}}}{\lVert \mathbf{p_{\text{hit}}} \rVert_2}\right) \times \frac{180^\circ}{\pi}, \qquad \lambda = \text{atan2}(-z_{\text{gl}}, x_{\text{gl}}) \times \frac{180^\circ}{\pi}
$$

   which is then dispatched to `/api/site/inspect`.

---

## 4. Offline Data Generation Pipeline

The baseline datasets in `processed_data/` are generated by four offline data-engineering scripts:

1. **`generate_landing_sites.py`**: Generates the baseline grid of candidate lunar landing sites, samples NASA LRO LOLA DEMs, and computes the 12-element terrain hazard vector (`elevation_m`, local/regional mean and max slopes, local/regional roughness) for each candidate.
2. **`generate_horizons.py`**: Runs batch 360° spherical horizon ray-casting across the LOLA DEMs for every candidate landing site, writing `processed_data/landing_site_horizons.csv`.
3. **`build_ssis.py`**: Compiles the primary catalog of Sites of Scientific Interest (Permanently Shadowed Regions, water-ice detections, volcanic pits, and historic/Artemis reference sites) with intrinsic scientific value scores $V_j$.
4. **`expanded_ssis.py`**: Expands and densifies the SSI catalog with secondary geological targets and localized polar cold traps for high-resolution rover route planning.

---

## 5. Performance, Caching & Persistence Architecture

* **Binary Horizon Cache (`landing_site_horizons_cache.npz`):** On initial server startup, the backend compiles `processed_data/landing_site_horizons.csv` into a binary NumPy archive (`landing_site_horizons_cache.npz`). Subsequent startups load the 360° horizon matrix directly from the `.npz` binary in milliseconds rather than parsing large CSV text files.
* **Persistent User Inspection Cache (`user_inspected_sites.csv` & `user_horizon_profiles.csv`):** Every custom coordinate inspected via `/api/site/inspect` has its 12-element terrain metadata and 360° horizon profile appended to `api/processed_data/` and hot-loaded into the in-memory data structures, making repeat queries for that coordinate instantaneous.
* **GDAL `/vsicurl/` Chunk Caching:** Remote NASA DEM reads use GDAL's internal block cache so adjacent pixel queries or route-map renders over a recently inspected region reuse cached HTTP byte ranges.
* **Pre-Built 3D `cKDTree`:** Constructed once in memory when `scores_calculator.py` initializes, allowing hundreds of candidate sites to be batch-queried against the SSI catalog in $< 5\text{ ms}$.

---

## 6. Detailed File-by-File Codebase Reference

### 6.1 Backend & Scientific Analysis Engine (`api/`)

| File Path | Detailed Responsibilities |
| :--- | :--- |
| `api/main.py` | FastAPI application server. Configures CORS middleware, initializes cached site/horizon datasets, and defines the REST API endpoints for batch MCDM ranking, single-point inspection, hillshaded route map streaming, and Gemini AI recommendations. |
| `api/render.yaml` | Infrastructure-as-Code Blueprint for deploying the Python FastAPI service on Render. |
| `api/requirements.txt` | Specifies backend Python package dependencies (`fastapi`, `uvicorn`, `numpy`, `scipy`, `matplotlib`, `gdal`, `requests`, `google-genai`, `pydantic`). |
| `api/analysis/inspect_point.py` | Opens remote NASA LOLA DEMs via GDAL `/vsicurl/`, projects $(\phi, \lambda)$ to raster pixel offsets, computes local and regional finite-difference slopes and elevation roughness, invokes 360° horizon ray-casting, and appends new records to `user_inspected_sites.csv` and `user_horizon_profiles.csv`. |
| `api/analysis/horizon.py` | Implements the 360° spherical geometry ray-casting engine over local and regional DEM windows to produce the 360-bin horizon elevation profile $H(\psi)$. |
| `api/analysis/AstronomicalScore.py` | Interfaces with the NASA JPL Horizons REST API (`ssd.jpl.nasa.gov/api/horizons.api`) to fetch topocentric Sun and Earth ephemerides and integrates Line-of-Sight visibility against $H(\psi)$. |
| `api/analysis/score_calculator.py` | Computes single-site Landing Safety and Scientific Accessibility scores from raw 12-element terrain vectors and mission constraints. |
| `api/analysis/scores_calculator.py` | Vectorized batch MCDM ranking engine. Builds the 3D LCLF `scipy.spatial.cKDTree` over all SSIs, computes distance-decayed scientific utility with quadratic max-scaling, aggregates weighted MCDM scores, and implements `solve_best_rover_route`. |
| `api/analysis/render_route_map.py` | Streams local DEM topography, computes Lambertian hillshading, plots SSI targets and the optimal rover traverse path, and returns an in-memory PNG stream (`io.BytesIO`). |
| `api/analysis/GeminiAPICaller.py` | Sends natural-language mission prompts to Google Gemini with a structured response schema to automatically configure MCDM weights and lander hardware limits. |
| `api/analysis/landing_site_scoring_methodology.md` | Formal mathematical specification of the safety, scientific accessibility, and MCDM normalization equations. |
| `api/analysis/notes.md` | Developer notes detailing NASA LOLA data URLs, projection parameters, and pipeline design decisions. |
| `api/analysis/test.py` | Standalone test harness for verifying horizon ray-casting, ephemeris scoring, and MCDM ranking outputs. |

### 6.2 Interactive 3D Web Client (`frontend/`)

| File Path | Detailed Responsibilities |
| :--- | :--- |
| `frontend/src/App.svelte` | Top-level Svelte 5 component orchestrating the 3D lunar globe viewport, control panels, timeline slider, and site inspection modals. |
| `frontend/src/lib/Moon.svelte` | Three.js WebGL component rendering the 3D lunar globe with `lroc_color_poles_1k.jpg` albedo and `ldem_3_8bit.jpg` vertex displacement, orbital camera controls, 3D site pins, and click-to-inspect raycasting. |
| `frontend/src/lib/ParameterSelector.svelte` | Interactive control panel for adjusting the 4 MCDM weights, lander hardware tolerances ($S_{\max}, \sigma_{\max}, R_{\max}$), preset lander profiles, and the Gemini AI mission prompt input. |
| `frontend/src/lib/SiteSelector.svelte` | Displays the live-sorted leaderboard of ranked landing sites and provides manual $(\text{lat}, \text{lon})$ coordinate entry for custom site inspection. |
| `frontend/src/lib/SiteCard.svelte` | Telemetry modal displaying a selected site's 12-element terrain metrics, score bars, reachable SSIs, and the dynamically streamed 2D hillshaded rover traverse map. |
| `frontend/src/lib/TimeSlider.svelte` | Interactive mission timeline control for selecting mission start/end dates and stepping through ephemeris epochs. |
| `frontend/src/shared/config.svelte.js` | Centralizes API endpoint base URLs and environment configuration. |
| `frontend/src/shared/inputs.svelte.js` | Reactive Svelte 5 `$state` store holding user MCDM weights, hardware limits, and mission epoch timestamps. |
| `frontend/src/shared/lander.js` | Defines preset lander and rover hardware specifications (maximum tilt angle, maximum surface roughness, and maximum rover traverse radius). |
| `frontend/src/shared/landing_sites.js` | Static fallback and baseline catalog of candidate lunar landing sites. |
| `frontend/src/shared/sites.svelte.js` | Reactive Svelte 5 store managing the active site list, selected site state, and asynchronous requests to the FastAPI backend. |

---

## 7. Data Schemas (`processed_data/`)

### 12-Element Terrain Metadata Schema (`user_inspected_sites.csv`)
| Index | Column Name | Type | Units | Mathematical Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `0` | `site_id` | `int`/`str` | — | Unique identifier for the baseline or custom landing site |
| `1` | `lat` | `float` | Degrees | Selenographic latitude $\phi \in [-90^\circ, 90^\circ]$ |
| `2` | `lon` | `float` | Degrees | Selenographic longitude $\lambda \in [-180^\circ, 180^\circ]$ |
| `3` | `elevation_m` | `float` | Meters | Surface elevation $z_0$ relative to the $1,737.4\text{ km}$ reference sphere |
| `4` | `local_mean_slope_deg` | `float` | Degrees | Mean slope $\bar{\theta_{\text{local}}}$ within the local touchdown window |
| `5` | `local_max_slope_deg` | `float` | Degrees | Maximum slope $\theta_{\max, \text{local}}$ within the local touchdown window |
| `6` | `local_roughness_m` | `float` | Meters | Elevation standard deviation $\sigma_{\text{local}}$ within the local window |
| `7` | `regional_mean_slope_deg` | `float` | Degrees | Mean slope $\bar{\theta_{\text{regional}}}$ across the regional dispersion window |
| `8` | `regional_max_slope_deg` | `float` | Degrees | Maximum slope $\theta_{\max, \text{regional}}$ across the regional dispersion window |
| `9` | `regional_roughness_m` | `float` | Meters | Elevation standard deviation $\sigma_{\text{regional}}$ across the regional window |
| `10` | `is_custom` | `bool` | — | `True` if dynamically inspected by a user via `/vsicurl/`, `False` if baseline grid |
| `11` | `label` | `str` | — | Human-readable site or lunar region designation |

### 360° Horizon Profile Schema (`landing_site_horizons.csv`, `user_horizon_profiles.csv`)
* **`site_id`**: Matches the `site_id` in the site metadata table.
* **`az_0` ... `az_359`**: 360 floating-point columns storing the maximum spherical horizon elevation angle $H(\psi)$ (in degrees) for each $1^\circ$ azimuth bin $\psi \in \lbrace 0^\circ, 1^\circ, \dots, 359^\circ \rbrace$.

---

## 8. REST API Specification

| Method | Endpoint | Query / JSON Payload | Response Format |
| :--- | :--- | :--- | :--- |
| `GET` / `POST` | `/api/sites/rank` | `w_safety`, `w_science`, `w_sun`, `w_earth` ($\in [0,1]$), `max_slope` (deg), `max_roughness` (m), `rover_range_km` (km), `start_time`, `end_time` (ISO-8601) | JSON list of all sites enriched with `safety_score`, `science_score`, `sun_score`, `earth_score`, `total_score`, and reachable SSIs. |
| `POST` | `/api/site/inspect` | `{"lat": float, "lon": float}` | Streams NASA DEM via GDAL `/vsicurl/`, computes 12-element terrain vector & 360° horizon profile, caches to CSV, and returns full scored site object. |
| `GET` | `/api/site/route-map` | `lat` (float), `lon` (float), `rover_range_km` (float) | `image/png` binary stream (`StreamingResponse`) of the hillshaded DEM and optimal rover traverse route. |
| `POST` | `/api/gemini/recommend` | `{"prompt": str}` | JSON containing Gemini-synthesized MCDM weights, hardware tolerances, and mission rationale. |
