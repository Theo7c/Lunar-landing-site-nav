import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
for rel_path in ("", "..", os.path.join("..", "..")):
    candidate_dir = os.path.abspath(os.path.join(CURRENT_DIR, rel_path))
    if candidate_dir not in sys.path:
        sys.path.insert(0, candidate_dir)

try:
    from api.analysis.inspect_point import CSV_COLUMNS, LANDING_SITES_CSV, CUSTOM_SITES_CSV
    from api.analysis.score_calculator import (
        calculate_safety_score_vectorized,
        calculate_astronomical_scores_batch,
    )
except ImportError:
    from inspect_point import CSV_COLUMNS, LANDING_SITES_CSV, CUSTOM_SITES_CSV
    from score_calculator import (
        calculate_safety_score_vectorized,
        calculate_astronomical_scores_batch,
    )

SSIS_CSV = "processed_data/ssis.csv"
DEFAULT_MAX_STOPS = 20


def _resolve_csv_path(rel_path: str) -> str:
    if os.path.exists(rel_path):
        return rel_path
    for base in (CURRENT_DIR, os.path.dirname(CURRENT_DIR), os.path.dirname(os.path.dirname(CURRENT_DIR))):
        candidate = os.path.join(base, rel_path)
        if os.path.exists(candidate):
            return candidate
    return rel_path


def _route_score(
    order: list[int],
    dist_from_lander: np.ndarray,
    pairwise_dist: np.ndarray,
    cand_vi: np.ndarray,
    decay_rate: float,
) -> float:
    """Compute distance-decayed cumulative science value along an ordered sequence of stops."""
    total_val = 0.0
    cum_dist = 0.0
    prev = -1
    for idx in order:
        step = dist_from_lander[idx] if prev == -1 else pairwise_dist[prev, idx]
        cum_dist += step
        total_val += cand_vi[idx] * np.exp(-decay_rate * cum_dist)
        prev = idx
    return float(total_val)


def _route_length(
    order: list[int],
    dist_from_lander: np.ndarray,
    pairwise_dist: np.ndarray,
    return_to_lander: bool,
) -> float:
    """Compute total path length in km for a candidate stop sequence."""
    if not order:
        return 0.0
    total = float(dist_from_lander[order[0]])
    for i in range(1, len(order)):
        total += float(pairwise_dist[order[i - 1], order[i]])
    if return_to_lander:
        total += float(dist_from_lander[order[-1]])
    return total


def solve_best_rover_route_with_path(
    site_xyz: np.ndarray,
    cand_xyz: np.ndarray,
    cand_vi: np.ndarray,
    r_max_km: float = 12.0,
    decay_rate: float = 0.15,
    return_to_lander: bool = False,
    max_stops: int = DEFAULT_MAX_STOPS,
) -> tuple[float, list[int]]:
    """
    Solve the distance-constrained lunar rover orienteering problem up to `max_stops`
    (default 20) science targets using value-per-km greedy construction + 2-opt
    local search. Returns `(raw_science_weight, visited_indices)`.
    """
    n_cand = len(cand_vi)
    stop_cap = max(1, int(max_stops))
    if n_cand == 0 or r_max_km <= 0.0:
        return 0.0, []

    dist_from_lander = np.linalg.norm(cand_xyz - site_xyz[None, :], axis=1)
    min_roundtrip = 2.0 * dist_from_lander if return_to_lander else dist_from_lander
    feasible_mask = min_roundtrip <= r_max_km
    if not np.any(feasible_mask):
        return 0.0, []

    diff = cand_xyz[:, None, :] - cand_xyz[None, :, :]
    pairwise_dist = np.linalg.norm(diff, axis=2)

    visited: list[int] = []
    unvisited = set(np.where(feasible_mask)[0].tolist())
    curr_idx = -1
    cum_dist = 0.0

    while unvisited and len(visited) < stop_cap:
        best_next = -1
        best_ratio = -1.0

        for idx in unvisited:
            step_dist = dist_from_lander[idx] if curr_idx == -1 else pairwise_dist[curr_idx, idx]
            new_cum_dist = cum_dist + step_dist
            return_leg = dist_from_lander[idx] if return_to_lander else 0.0

            if new_cum_dist + return_leg > r_max_km:
                continue

            marginal_gain = cand_vi[idx] * np.exp(-decay_rate * new_cum_dist)
            ratio = marginal_gain / max(0.05, step_dist)
            if ratio > best_ratio:
                best_ratio = ratio
                best_next = idx

        if best_next == -1:
            break

        step_dist = dist_from_lander[best_next] if curr_idx == -1 else pairwise_dist[curr_idx, best_next]
        cum_dist += step_dist
        visited.append(best_next)
        unvisited.remove(best_next)
        curr_idx = best_next

    # Lightweight 2-opt swap pass to untangle crossings and improve decayed score
    if len(visited) >= 3:
        best_score = _route_score(visited, dist_from_lander, pairwise_dist, cand_vi, decay_rate)
        improved = True
        while improved:
            improved = False
            for i in range(len(visited) - 1):
                for j in range(i + 1, len(visited)):
                    candidate = visited[:i] + visited[i : j + 1][::-1] + visited[j + 1 :]
                    if _route_length(candidate, dist_from_lander, pairwise_dist, return_to_lander) <= r_max_km:
                        cand_score = _route_score(candidate, dist_from_lander, pairwise_dist, cand_vi, decay_rate)
                        if cand_score > best_score + 1e-6:
                            visited = candidate
                            best_score = cand_score
                            improved = True

    raw_weight = _route_score(visited, dist_from_lander, pairwise_dist, cand_vi, decay_rate)
    return round(raw_weight, 4), visited


def scores_calculator(
    max_slope_limit_deg: float = 15.0,
    mean_slope_limit_deg: float = 10.0,
    roughness_limit_m: float = 25.0,
    slope_weight: float = 0.60,
    r_max_km: float = 12.0,
    decay_rate: float = 0.15,
    science_half_sat: float = 5.0,
    return_to_lander: bool = False,
    max_stops: int = DEFAULT_MAX_STOPS,
    time_range: tuple[str, str] = ("2027-01-01 00:00", "2027-01-30 00:00"),
    time_resolution: tuple[int, str] = (1, "day"),
    filter_infeasible: bool = True,
) -> pd.DataFrame:
    """
    Batch-evaluate all candidate sites in `landing_sites.csv` and `user_inspected_sites.csv`,
    capping rover traverse stops at `max_stops` (default 20).
    """
    frames = []
    for rel_csv in (LANDING_SITES_CSV, CUSTOM_SITES_CSV):
        resolved = _resolve_csv_path(rel_csv)
        if os.path.exists(resolved):
            frames.append(pd.read_csv(resolved))

    if not frames:
        return pd.DataFrame(columns=CSV_COLUMNS + ["s_safety", "s_science", "s_sun", "s_earth", "reachable_ssi_count", "visited_ssi_count"])

    sites_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["site_id"], keep="last").reset_index(drop=True)

    # 1. Vectorized Terrain Safety Score
    s_safety = calculate_safety_score_vectorized(
        max_slope_deg=sites_df["max_slope_deg"].to_numpy(dtype=np.float64),
        mean_slope_deg=sites_df["mean_slope_deg"].to_numpy(dtype=np.float64),
        roughness_m=sites_df["roughness_m"].to_numpy(dtype=np.float64),
        max_slope_limit_deg=max_slope_limit_deg,
        mean_slope_limit_deg=mean_slope_limit_deg,
        roughness_limit_m=roughness_limit_m,
        slope_weight=slope_weight,
    )
    sites_df["s_safety"] = np.round(s_safety, 4)

    if filter_infeasible:
        sites_df = sites_df[sites_df["s_safety"] > 0.0].reset_index(drop=True)
        if sites_df.empty:
            return sites_df

    # 2. Rover Science Orienteering Score (capped at max_stops)
    n_sites = len(sites_df)
    s_science = np.zeros(n_sites, dtype=np.float64)
    reachable_counts = np.zeros(n_sites, dtype=np.int32)
    visited_counts = np.zeros(n_sites, dtype=np.int32)

    ssis_path = _resolve_csv_path(SSIS_CSV)
    if os.path.exists(ssis_path) and r_max_km > 0.0:
        ssis_df = pd.read_csv(ssis_path)
        if not ssis_df.empty:
            ssi_xyz = ssis_df[["x_km", "y_km", "z_km"]].to_numpy(dtype=np.float64)
            ssi_vi = ssis_df["v_i"].to_numpy(dtype=np.float64)
            ssi_tree = cKDTree(ssi_xyz)

            site_xyz_all = sites_df[["x_km", "y_km", "z_km"]].to_numpy(dtype=np.float64)
            search_r = float(r_max_km) * (0.5 if return_to_lander else 1.0)
            nbr_lists = ssi_tree.query_ball_point(site_xyz_all, r=search_r)

            half_sat = max(1e-6, float(science_half_sat))
            for i, nbrs in enumerate(nbr_lists):
                if not nbrs:
                    continue
                nbr_arr = np.asarray(nbrs, dtype=np.int32)
                reachable_counts[i] = len(nbr_arr)
                raw_w, visited_order = solve_best_rover_route_with_path(
                    site_xyz=site_xyz_all[i],
                    cand_xyz=ssi_xyz[nbr_arr],
                    cand_vi=ssi_vi[nbr_arr],
                    r_max_km=float(r_max_km),
                    decay_rate=float(decay_rate),
                    return_to_lander=return_to_lander,
                    max_stops=max_stops,
                )
                visited_counts[i] = len(visited_order)
                s_science[i] = 1.0 - np.exp(-np.log(2.0) * raw_w / half_sat)

    sites_df["s_science"] = np.round(s_science, 4)
    sites_df["reachable_ssi_count"] = reachable_counts
    sites_df["visited_ssi_count"] = visited_counts

    # 3. Sun & Earth Visibility Scores
    s_sun, s_earth = calculate_astronomical_scores_batch(
        sites_df=sites_df,
        time_range=time_range,
        time_resolution=time_resolution,
    )
    sites_df["s_sun"] = np.round(s_sun, 4)
    sites_df["s_earth"] = np.round(s_earth, 4)

    return sites_df