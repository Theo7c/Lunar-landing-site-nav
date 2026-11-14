import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE_SITES_CSV = "processed_data/landing_sites.csv"
CUSTOM_SITES_CSV = "processed_data/user_inspected_sites.csv"
SSIS_CSV = "processed_data/ssis.csv"


def load_sites_and_ssis(
    base_csv: str = BASE_SITES_CSV,
    custom_csv: str = CUSTOM_SITES_CSV,
    ssis_csv: str = SSIS_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    site_dfs = []
    if os.path.exists(base_csv):
        site_dfs.append(pd.read_csv(base_csv))
    if os.path.exists(custom_csv):
        site_dfs.append(pd.read_csv(custom_csv))
    sites_df = pd.concat(site_dfs, ignore_index=True) if site_dfs else pd.DataFrame()
    ssis_df = pd.read_csv(ssis_csv) if os.path.exists(ssis_csv) else pd.DataFrame()
    return sites_df, ssis_df


def calculate_scores(
    sites_df: pd.DataFrame | None = None,
    ssis_df: pd.DataFrame | None = None,
    max_slope_limit_deg: float = 15.0,
    mean_slope_limit_deg: float = 10.0,
    roughness_limit_m: float = 25.0,
    slope_weight: float = 0.60,
    r_max_km: float = 12.0,
    decay_rate: float = 0.15,
    science_half_sat: float = 5.0,
    filter_infeasible: bool = True,
) -> pd.DataFrame:
    """
    Fast, vectorized batch scoring across all candidate sites.
    Scores are scaled independently into [0.0, 1.0] per site with zero dependence
    on other sites in the grid.
    """
    if sites_df is None or ssis_df is None:
        loaded_sites, loaded_ssis = load_sites_and_ssis()
        sites_df = loaded_sites if sites_df is None else sites_df
        ssis_df = loaded_ssis if ssis_df is None else ssis_df

    if sites_df.empty:
        return sites_df.copy()

    max_slp = sites_df["max_slope_deg"].to_numpy(dtype=np.float64)
    mean_slp = sites_df["mean_slope_deg"].to_numpy(dtype=np.float64)
    rough = sites_df["roughness_m"].to_numpy(dtype=np.float64)

    max_lim = max(1e-6, float(max_slope_limit_deg))
    rough_lim = max(1e-6, float(roughness_limit_m))
    alpha = float(np.clip(slope_weight, 0.0, 1.0))

    # 1. Feasibility & Independent Safety Score in [0.0, 1.0]
    feasible = (max_slp <= max_lim) & (mean_slp <= mean_slope_limit_deg)

    if filter_infeasible:
        df = sites_df.loc[feasible].reset_index(drop=True)
        max_slp = max_slp[feasible]
        rough = rough[feasible]
        feasible_out = np.ones(len(df), dtype=bool)
    else:
        df = sites_df.copy()
        feasible_out = feasible

    n_sites = len(df)
    if n_sites == 0:
        df["feasible"] = pd.Series(dtype=bool)
        df["s_safety"] = pd.Series(dtype=float)
        df["reachable_ssi_count"] = pd.Series(dtype=int)
        df["s_science"] = pd.Series(dtype=float)
        return df

    t_slope = np.clip(1.0 - (max_slp / max_lim), 0.0, 1.0)
    t_rough = np.clip(1.0 - (rough / rough_lim), 0.0, 1.0)
    s_safety = np.where(feasible_out, alpha * t_slope + (1.0 - alpha) * t_rough, 0.0)

    # 2. Fast 3D cKDTree Radius Query for Independent Science Score in [0.0, 1.0]
    reachable_counts = np.zeros(n_sites, dtype=np.int32)
    raw_science = np.zeros(n_sites, dtype=np.float64)

    if not ssis_df.empty and r_max_km > 0.0:
        site_xyz = df[["x_km", "y_km", "z_km"]].to_numpy(dtype=np.float64)
        ssi_xyz = ssis_df[["x_km", "y_km", "z_km"]].to_numpy(dtype=np.float64)
        ssi_vi = ssis_df["v_i"].to_numpy(dtype=np.float64)

        ssi_tree = cKDTree(ssi_xyz)
        neighbor_lists = ssi_tree.query_ball_point(site_xyz, r=float(r_max_km), workers=-1)

        # Flatten matched (site_idx, ssi_idx) pairs for vectorized distance & decay math
        counts = np.fromiter((len(lst) for lst in neighbor_lists), dtype=np.int32, count=n_sites)
        reachable_counts = counts
        total_pairs = int(counts.sum())

        if total_pairs > 0:
            site_idx_flat = np.repeat(np.arange(n_sites, dtype=np.int32), counts)
            ssi_idx_flat = np.fromiter(
                (idx for lst in neighbor_lists for idx in lst),
                dtype=np.int32,
                count=total_pairs,
            )

            diffs = site_xyz[site_idx_flat] - ssi_xyz[ssi_idx_flat]
            dists = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
            weights = ssi_vi[ssi_idx_flat] / (1.0 + float(decay_rate) * dists)

            np.add.at(raw_science, site_idx_flat, weights)

    half_sat = max(1e-6, float(science_half_sat))
    s_science = raw_science / (raw_science + half_sat)

    df["feasible"] = feasible_out
    df["s_safety"] = np.round(np.clip(s_safety, 0.0, 1.0), 4)
    df["reachable_ssi_count"] = reachable_counts
    df["s_science"] = np.round(np.clip(s_science, 0.0, 1.0), 4)
    return df


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    scored_df = calculate_scores()
    dt = (time.perf_counter() - t0) * 1000.0
    print(f"Scored {len(scored_df):,} feasible sites in {dt:.1f} ms")
    print(scored_df[["site_id", "tier", "s_safety", "reachable_ssi_count", "s_science"]].head(5).to_string(index=False))