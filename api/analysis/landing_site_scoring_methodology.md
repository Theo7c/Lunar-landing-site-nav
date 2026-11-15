# Lunar Landing Site Scoring Methodology

Each candidate landing site is evaluated across four independent, normalized metrics in the range $[0.0, 1.0]$ (rounded to 4 decimal places):

1. **Terrain Safety Score ($s_{\text{safety}}$)**
2. **Rover Traverse Science Score ($s_{\text{science}}$)**
3. **Solar Illumination Visibility Score ($s_{\text{sun}}$)**
4. **Earth Line-of-Sight Communication Score ($s_{\text{earth}}$)**

---

## 1. Terrain Safety Score ($s_{\text{safety}}$)

The safety score evaluates whether a landing site satisfies spacecraft touchdown stability constraints based on local slope and surface roughness derived from Lunar Reconnaissance Orbiter (LRO) LOLA Digital Elevation Models (DEMs).

### Feasibility Gate
A site must first pass two hard engineering thresholds:

$$\text{max\_slope\_deg} \le \theta_{\max, \text{lim}} \quad \text{and} \quad \text{mean\_slope\_deg} \le \theta_{\text{mean}, \text{lim}}$$

If either condition is violated, the site is marked infeasible (`feasible = False`) and immediately assigned:

$$s_{\text{safety}} = 0.0$$

### Normalized Calculation
For feasible sites, normalized sub-scores for maximum slope ($t_{\text{slope}}$) and terrain roughness ($t_{\text{rough}}$) are computed as:

$$t_{\text{slope}} = \text{clip}\left(1 - \frac{\text{max\_slope\_deg}}{\theta_{\max, \text{lim}}},\; 0.0,\; 1.0\right)$$

$$t_{\text{rough}} = \text{clip}\left(1 - \frac{\text{roughness\_m}}{R_{\text{lim}}},\; 0.0,\; 1.0\right)$$

These two components are combined via a convex weighted sum:

$$s_{\text{safety}} = \alpha \cdot t_{\text{slope}} + (1 - \alpha) \cdot t_{\text{rough}}$$

### Default Parameters
* $\theta_{\max, \text{lim}}$ (`max_slope_limit_deg`): $15.0^\circ$
* $\theta_{\text{mean}, \text{lim}}$ (`mean_slope_limit_deg`): $10.0^\circ$
* $R_{\text{lim}}$ (`roughness_limit_m`): $25.0\text{ m}$
* $\alpha$ (`slope_weight`): $0.60$ (60% slope weight, 40% roughness weight)

---

## 2. Rover Traverse Science Score ($s_{\text{science}}$)

The science score quantifies the cumulative scientific value a surface rover can collect from nearby **Sites of Scientific Interest (SSIs)** within a maximum traverse distance budget $r_{\max}$.

### Step 1: Spatial Candidate Filtering
Using a 3D Euclidean `cKDTree` over lunar body-fixed coordinates $(x_{\text{km}}, y_{\text{km}}, z_{\text{km}})$, all SSIs within radius $r_{\max}$ of the landing site $\mathbf{p}_0$ are identified:

$$\|\mathbf{p}_i - \mathbf{p}_0\|_2 \le r_{\max}$$

### Step 2: Distance-Decayed Multi-Stop Route Optimization
Rather than summing all nearby SSIs blindly, the solver finds an ordered route $\pi = (i_1, i_2, \dots, i_L)$ starting at $\mathbf{p}_0$ that maximizes cumulative decayed scientific value subject to the rover's total range limit $r_{\max}$:

$$W_{\text{raw}}(\pi) = \sum_{j=1}^{L} \frac{v_{i_j}}{1 + k_{\text{decay}} \cdot d_{\text{cum}, j}}$$

where:
* $v_{i_j}$ is the intrinsic scientific priority value of the $j$-th visited SSI.
* $k_{\text{decay}}$ (`decay_rate`) penalizes targets visited later in the traverse.
* $d_{\text{cum}, j} = \|\mathbf{p}_{i_1} - \mathbf{p}_0\|_2 + \sum_{m=2}^{j} \|\mathbf{p}_{i_m} - \mathbf{p}_{i_{m-1}}\|_2$ is the cumulative path distance up to stop $j$.

The total traverse distance must satisfy:

$$D_{\text{total}}(\pi) = d_{\text{cum}, L} + d_{\text{return}} \le r_{\max}$$

where $d_{\text{return}} = \|\mathbf{p}_{i_L} - \mathbf{p}_0\|_2$ if `return_to_lander=True`, and $d_{\text{return}} = 0$ for one-way traverses.

* **Exact Solver ($m \le 12$ reachable SSIs):** Solved to global optimality via Bitmask Dynamic Programming over $2^m$ subsets.
* **Heuristic Solver ($m > 12$ reachable SSIs):** Solved via multi-tradeoff distance-penalized Beam Search followed by 2-Opt path shortening and greedy node insertion.

### Step 3: Hyperbolic Saturation Normalization
To map the unbounded raw route weight $W_{\text{raw}} \in [0, \infty)$ into $[0.0, 1.0)$ with diminishing returns, a Michaelis-Menten / hyperbolic saturation curve is applied:

$$s_{\text{science}} = \frac{W_{\text{raw}}}{W_{\text{raw}} + H_{1/2}}$$

where $H_{1/2}$ (`science_half_sat`) is the raw science weight required to achieve a score of $0.50$.

### Default Parameters
* $r_{\max}$ (`r_max_km`): $12.0\text{ km}$
* $k_{\text{decay}}$ (`decay_rate`): $0.15\text{ km}^{-1}$
* $H_{1/2}$ (`science_half_sat`): $5.0$
* `return_to_lander`: `False`

---

## 3. Solar & Earth Visibility Scores ($s_{\text{sun}}$ and $s_{\text{earth}}$)

Both astronomical visibility scores measure the fraction of time over a mission window that the target body (**Sun** for solar power and thermal stability; **Earth** for direct-to-Earth line-of-sight communications) clears the surrounding lunar topography.

### Step 1: 360-Degree Local Horizon Profile
For a site at $(\text{lat}, \text{lon})$, `inspect_site_horizon` ray-casts outward across the lunar DEM in $1^\circ$ azimuth increments to build a 360-element terrain horizon elevation array:

$$H(\text{Az}) \quad \text{for } \text{Az} \in \{0, 1, 2, \dots, 359\}$$

where $H(\text{Az})$ is the minimum elevation angle (in degrees) required to clear surrounding crater rims and ridges along azimuth $\text{Az}$.

### Step 2: Topocentric Ephemeris Query
Using NASA JPL Horizons (`GetHorizonsEphemeris`) at observer coordinates $(\text{lon}, \text{lat}, \text{alt}_{\text{km}})$ where $\text{alt}_{\text{km}} = \text{elevation\_m} / 1000$, topocentric azimuth $\text{Az}_b(t) \in [0, 360)$ and elevation angle $\text{Alt}_b(t)$ are retrieved for $N$ discrete timestamps $t = 1, \dots, N$.

### Step 3: Horizon Clearance & Visibility Fraction
At each timestamp $t$, the body's azimuth is rounded to the nearest integer degree index:

$$\text{Az}_{\text{idx}}(t) = \lfloor \text{Az}_b(t) + 0.5 \rfloor \bmod 360$$

The angular clearance above the local terrain horizon is:

$$\Delta \theta(t) = \text{Alt}_b(t) - H\big(\text{Az}_{\text{idx}}(t)\big)$$

The final score ($s_{\text{sun}}$ for `target="Sun"`, $s_{\text{earth}}$ for `target="Earth"`) is the proportion of time steps with positive line-of-sight clearance:

$$s_{\text{body}} = \frac{1}{N} \sum_{t=1}^{N} \mathbb{I}\big(\Delta \theta(t) > 0\big)$$

where $\mathbb{I}(\Delta \theta(t) > 0) = 1$ if $\Delta \theta(t) > 0$, and $0$ otherwise.

### Default Parameters
* `time_range`: `("2027-01-01", "2027-01-30")` (one full lunar synodic period, ~29.5 days)
* `time_resolution`: `(1, "hour")`

---

## Summary Table

| Score | Primary Inputs | Physical Meaning | Range |
| :--- | :--- | :--- | :--- |
| **`s_safety`** | `max_slope_deg`, `mean_slope_deg`, `roughness_m` | Touchdown hazard & lander tilt safety | $[0.0, 1.0]$ |
| **`s_science`** | Site $(x,y,z)$, `ssis.csv` coordinates & $v_i$ values | Optimal range-constrained rover science return | $[0.0, 1.0]$ |
| **`s_sun`** | Site $(\text{lat}, \text{lon}, \text{elev})$, DEM horizon, Sun ephemeris | Fraction of mission window in direct sunlight | $[0.0, 1.0]$ |
| **`s_earth`** | Site $(\text{lat}, \text{lon}, \text{elev})$, DEM horizon, Earth ephemeris | Fraction of mission window with Direct-to-Earth LOS | $[0.0, 1.0]$ |