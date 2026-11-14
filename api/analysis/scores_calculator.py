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


def solve_best_rover_route(
    site_xyz: np.ndarray,
    cand_xyz: np.ndarray,
    cand_vi: np.ndarray,
    r_max_km: float,
    decay_rate: float = 0.15,
    return_to_lander: bool = False,
    exact_limit: int = 12,
    beam_width: int = 32,
) -> tuple[float, int]:
    """
    Find the route starting at site_xyz visiting a subset of candidate SSIs
    that maximizes total distance-decayed weight without exceeding r_max_km.

    Returns (best_raw_weight, visited_stop_count).
    """
    m = len(cand_vi)
    if m == 0 or r_max_km <= 0.0:
        return 0.0, 0

    # Distances from lander (node 0) to each candidate SSI
    d0 = np.linalg.norm(cand_xyz - site_xyz[None, :], axis=1)
    roundtrip_mult = 2.0 if return_to_lander else 1.0

    # Fast path: single candidate SSI in range
    if m == 1:
        if d0[0] * roundtrip_mult <= r_max_km:
            return float(cand_vi[0] / (1.0 + decay_rate * d0[0])), 1
        return 0.0, 0

    # Pairwise Euclidean distance matrix between candidate SSIs: shape (m, m)
    diff = cand_xyz[:, None, :] - cand_xyz[None, :, :]
    dist_mat = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))

    # 1. Exact Bitmask DP for up to `exact_limit` SSIs
    if m <= exact_limit:
        n_states = 1 << m
        # dp_dist[mask, last] = minimum cumulative path distance to reach `last` visiting `mask`
        # dp_score[mask, last] = maximum cumulative decayed weight for that state
        dp_dist = np.full((n_states, m), np.inf, dtype=np.float64)
        dp_score = np.full((n_states, m), -1.0, dtype=np.float64)

        best_score = 0.0
        best_stops = 0

        for i in range(m):
            leg = float(d0[i])
            total_req = leg + (leg if return_to_lander else 0.0)
            if total_req <= r_max_km:
                mask = 1 << i
                sc = float(cand_vi[i] / (1.0 + decay_rate * leg))
                dp_dist[mask, i] = leg
                dp_score[mask, i] = sc
                if sc > best_score:
                    best_score = sc
                    best_stops = 1

        for mask in range(1, n_states):
            # Skip single-bit masks already initialized
            if (mask & (mask - 1)) == 0:
                continue
            stops = mask.bit_count()

            # Iterate over set bits for `last`
            temp_last = mask
            while temp_last:
                lsb_last = temp_last & -temp_last
                last = lsb_last.bit_length() - 1
                temp_last ^= lsb_last

                prev_mask = mask ^ lsb_last
                vi_last = float(cand_vi[last])
                d_ret = float(d0[last]) if return_to_lander else 0.0

                best_state_score = -1.0
                best_state_dist = np.inf

                temp_prev = prev_mask
                while temp_prev:
                    lsb_prev = temp_prev & -temp_prev
                    prev = lsb_prev.bit_length() - 1
                    temp_prev ^= lsb_prev

                    prev_d = dp_dist[prev_mask, prev]
                    if prev_d == np.inf:
                        continue

                    new_d = prev_d + float(dist_mat[prev, last])
                    if new_d + d_ret <= r_max_km:
                        new_sc = dp_score[prev_mask, prev] + vi_last / (1.0 + decay_rate * new_d)
                        if new_sc > best_state_score or (
                            new_sc == best_state_score and new_d < best_state_dist
                        ):
                            best_state_score = new_sc
                            best_state_dist = new_d

                if best_state_score >= 0.0:
                    dp_dist[mask, last] = best_state_dist
                    dp_score[mask, last] = best_state_score
                    if best_state_score > best_score:
                        best_score = best_state_score
                        best_stops = stops

        return best_score, best_stops

    # 2. Bounded Beam Search for dense clusters (m > exact_limit)
    # State tuple: (score, cum_dist, last_idx, visited_mask)
    beam: list[tuple[float, float, int, int]] = []
    best_score = 0.0
    best_stops = 0

    for i in range(m):
        leg = float(d0[i])
        d_ret = leg if return_to_lander else 0.0
        if leg + d_ret <= r_max_km:
            sc = float(cand_vi[i] / (1.0 + decay_rate * leg))
            beam.append((sc, leg, i, 1 << i))
            if sc > best_score:
                best_score = sc
                best_stops = 1

    beam.sort(key=lambda x: x[0], reverse=True)
    beam = beam[:beam_width]

    for step in range(2, m + 1):
        next_states: dict[tuple[int, int], tuple[float, float, int, int]] = {}
        for sc, cum_d, last, mask in beam:
            for nxt in range(m):
                if mask & (1 << nxt):
                    continue
                new_d = cum_d + float(dist_mat[last, nxt])
                d_ret = float(d0[nxt]) if return_to_lander else 0.0
                if new_d + d_ret <= r_max_km:
                    new_sc = sc + float(cand_vi[nxt]) / (1.0 + decay_rate * new_d)
                    new_mask = mask | (1 << nxt)
                    key = (new_mask, nxt)
                    prev_best = next_states.get(key)
                    if prev_best is None or new_sc > prev_best[0]:
                        next_states[key] = (new_sc, new_d, nxt, new_mask)
                        if new_sc > best_score:
                            best_score = new_sc
                            best_stops = step

        if not next_states:
            break
        beam = sorted(next_states.values(), key=lambda x: x[0], reverse=True)[:beam_width]

    return best_score, best_stops


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
    return_to_lander: bool = False,
    filter_infeasible: bool = True,
) -> pd.DataFrame:
    """
    Fast batch scoring across all candidate sites.
    Finds all SSIs within r_max_km and computes the highest-weight rover route
    whose total travel distance does not exceed r_max_km.
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
        df["visited_ssi_count"] = pd.Series(dtype=int)
        df["s_science"] = pd.Series(dtype=float)
        return df

    t_slope = np.clip(1.0 - (max_slp / max_lim), 0.0, 1.0)
    t_rough = np.clip(1.0 - (rough / rough_lim), 0.0, 1.0)
    s_safety = np.where(feasible_out, alpha * t_slope + (1.0 - alpha) * t_rough, 0.0)

    # 2. Optimal Max-Weight Rover Route within r_max_km budget
    reachable_counts = np.zeros(n_sites, dtype=np.int32)
    visited_counts = np.zeros(n_sites, dtype=np.int32)
    raw_science = np.zeros(n_sites, dtype=np.float64)

    if not ssis_df.empty and r_max_km > 0.0:
        site_xyz = df[["x_km", "y_km", "z_km"]].to_numpy(dtype=np.float64)
        ssi_xyz = ssis_df[["x_km", "y_km", "z_km"]].to_numpy(dtype=np.float64)
        ssi_vi = ssis_df["v_i"].to_numpy(dtype=np.float64)

        ssi_tree = cKDTree(ssi_xyz)
        neighbor_lists = ssi_tree.query_ball_point(site_xyz, r=float(r_max_km), workers=-1)

        r_budget = float(r_max_km)
        k_decay = float(decay_rate)

        for idx, nbrs in enumerate(neighbor_lists):
            n_nbrs = len(nbrs)
            reachable_counts[idx] = n_nbrs
            if n_nbrs == 0:
                continue

            nbr_arr = np.asarray(nbrs, dtype=np.int32)
            best_w, n_visited = solve_best_rover_route(
                site_xyz=site_xyz[idx],
                cand_xyz=ssi_xyz[nbr_arr],
                cand_vi=ssi_vi[nbr_arr],
                r_max_km=r_budget,
                decay_rate=k_decay,
                return_to_lander=return_to_lander,
            )
            raw_science[idx] = best_w
            visited_counts[idx] = n_visited

    half_sat = max(1e-6, float(science_half_sat))
    s_science = raw_science / (raw_science + half_sat)

    df["feasible"] = feasible_out
    df["s_safety"] = np.round(np.clip(s_safety, 0.0, 1.0), 4)
    df["reachable_ssi_count"] = reachable_counts
    df["visited_ssi_count"] = visited_counts
    df["s_science"] = np.round(np.clip(s_science, 0.0, 1.0), 4)
    return df


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    scored_df = calculate_scores()
    dt = (time.perf_counter() - t0) * 1000.0
    print(f"Scored {len(scored_df):,} feasible sites with optimal multi-stop routing in {dt:.1f} ms")
    cols = ["site_id", "tier", "s_safety", "reachable_ssi_count", "visited_ssi_count", "s_science"]
    print(scored_df[cols].head(5).to_string(index=False))