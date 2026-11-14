# Score Calculation & Grid Standardization (`scores_calculator.py` & `score_calculator.py`)

This document describes how **Landing Safety ($S_{\text{safety}}$)** and **Scientific Accessibility ($S_{\text{science}}$)** are computed dynamically from user-configured lander/rover parameters and standardized onto a unified $[0.0, 1.0]$ scale across the lunar grid.

---

## 1. Two-Stage Architecture

1. **Batch Grid & Calibration Engine (`api/analysis/scores_calculator.py`):**
   * Runs whenever the user updates their lander or rover hardware parameters (e.g., max tilt angle, roughness clearance, rover range).
   * Evaluates all ~60,000 candidate sites in `landing_sites.csv` + `user_inspected_sites.csv`, standardizes the resulting score distributions across $[0.0, 1.0]$, and writes the fitted parameters to `processed_data/score_calibration.json`.
2. **Single-Point Scorer (`api/analysis/score_calculator.py`):**
   * Runs when a user clicks an arbitrary coordinate on the 3D Moon (`inspect_point.main()`).
   * Accepts a **12-element CSV row list** (`[site_id, tier, lat, lon, x_km, y_km, z_km, area_radius_m, elevation_m, max_slope_deg, mean_slope_deg, roughness_m]`), evaluates raw safety and science using the active mission parameters, and maps both onto $[0.0, 1.0]$ using `score_calibration.json`.

---

## 2. Raw Metric Computation

### A. Raw Landing Safety ($R_{\text{safety}}$)
Each site $x$ is first checked against the lander's slope limits:
$$\text{feasible}(x) = \bigl(\theta_{\max}(x) \le \theta_{\text{max\_limit}}\bigr) \;\land\; \bigl(\theta_{\text{mean}}(x) \le \theta_{\text{mean\_limit}}\bigr)$$

If `feasible(x)` is `False`, $R_{\text{safety}}(x) = 0.0$ (and the site is filtered out in batch mode when `filter_infeasible=True`). Otherwise:

$$T_{\text{slope}}(x) = \text{clip}\!\left(1 - \frac{\theta_{\max}(x)}{\theta_{\text{max\_limit}}},\; 0,\; 1\right), \qquad T_{\text{rough}}(x) = \text{clip}\!\left(1 - \frac{\sigma_z(x)}{\sigma_{\text{limit}}},\; 0,\; 1\right)$$

$$R_{\text{safety}}(x) = \alpha \, T_{\text{slope}}(x) + (1 - \alpha) \, T_{\text{rough}}(x)$$

### B. Raw Scientific Accessibility ($R_{\text{science}}$)
Using Moon-centered 3D Cartesian coordinates $(x_{\text{km}}, y_{\text{km}}, z_{\text{km}})$ with $R_{\text{Moon}} = 1737.4\text{ km}$, the straight-line distance $d_i(x)$ is computed to every target $i$ in `processed_data/ssis.csv` ($V_i \in [1.0, 5.0]$):

$$d_i(x) = \sqrt{(x - x_i)^2 + (y - y_i)^2 + (z - z_i)^2}$$

All targets within the rover/EVA range $R_{\max}$ contribute an inverse-distance weighted value:

$$R_{\text{science}}(x) = \sum_{d_i(x) \le R_{\max}} \frac{V_i}{1 + k \, d_i(x)}$$

*(If no SSIs lie within $R_{\max}$, $R_{\text{science}}(x) = 0.0$.)*

---

## 3. Grid-Dependent Standardization ($[0.0, 1.0]$)

Raw safety and raw science naturally span different numerical ranges (for example, $R_{\text{safety}} \in [0.4, 0.95]$ while $R_{\text{science}} \in [0.0, 18.0]$). To ensure user MCDM slider weights ($w_{\text{safety}}, w_{\text{science}}$) reflect true relative preference without scale bias, positive raw values ($R > 0$) across the active grid are mapped into $[0.0, 1.0]$ using one of two methods (while $R \le 0$ stays pinned at `0.0`):

1. **Uniform / Empirical CDF (`method="uniform"` — Default):**
   * Fits **201 quantile knots** ($0.0\%, 0.5\%, \dots, 100.0\%$) over all positive values on the grid and applies piecewise linear interpolation (`np.interp`).
   * Produces a uniform percentile distribution across feasible sites so every slider increment yields equal marginal selectivity.
2. **Z-Score + Sigmoid (`method="sigmoid"`):**
   * Computes grid mean $\mu$ and standard deviation $\sigma$ over positive values, then applies:
     $$S(x) = \frac{1}{1 + \exp\!\left(-\frac{R(x) - \mu}{\sigma}\right)}$$
   * Centers the grid average at `0.5` while preserving smooth spacing among extreme outliers.

---

## 4. Configurable Mission Parameters

| Parameter | Default | Used In | Description |
| :--- | :---: | :--- | :--- |
| `max_slope_limit_deg` | `15.0°` | Safety | Hard maximum single-pixel slope cutoff inside touchdown dispersion box. |
| `mean_slope_limit_deg` | `10.0°` | Safety | Hard maximum average slope cutoff across the landing patch. |
| `roughness_limit_m` | `25.0 m` | Safety | Local elevation standard deviation ($\sigma_z$) where $T_{\text{rough}}$ reaches `0.0`. |
| `slope_weight` ($\alpha$) | `0.60` | Safety | Convex weight between worst-case slope (`60%`) and roughness (`40%`). |
| `r_max_km` ($R_{\max}$) | `12.0 km` | Science | Maximum rover or EVA sortie radius from the landing site. |
| `decay_rate` ($k$) | `0.15 km⁻¹` | Science | Distance travel-cost penalty factor. |
| `method` | `"uniform"` | Both | Standardization curve (`"uniform"` ECDF quantiles or `"sigmoid"`). |