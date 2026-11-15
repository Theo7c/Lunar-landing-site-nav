import inspect
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
    from api.analysis.horizon import GetHorizonsEphemeris
except ImportError:
    from horizon import GetHorizonsEphemeris

try:
    from api.analysis.AstronomicalScore import FindBodyHorizonAngles
except ImportError:
    try:
        from AstronomicalScore import FindBodyHorizonAngles
    except ImportError:
        from AstronomicalScore import FindBodyHorizonAngles

try:
    from api.analysis.inspect_point import inspect_site_horizon
except ImportError:
    try:
        from inspect_point import inspect_site_horizon
    except ImportError:
        inspect_site_horizon = None


BASE_SITES_CSV = "processed_data/landing_sites.csv"
CUSTOM_SITES_CSV = "processed_data/user_inspected_sites.csv"
SSIS_CSV = "processed_data/ssis.csv"

_HORIZON_CACHE: dict[tuple[float, float], np.ndarray] = {}
_EPHEMERIS_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _resolve_csv_path(csv_path: str) -> str:
    if os.path.exists(csv_path):
        return csv_path
    for rel_path in ("..", os.path.join("..", "..")):
        candidate = os.path.abspath(os.path.join(CURRENT_DIR, rel_path, csv_path))
        if os.path.exists(candidate):
            return candidate
    return csv_path


def load_sites_and_ssis(
    base_csv: str = BASE_SITES_CSV,
    custom_csv: str = CUSTOM_SITES_CSV,
    ssis_csv: str = SSIS_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_path = _resolve_csv_path(base_csv)
    custom_path = _resolve_csv_path(custom_csv)
    ssis_path = _resolve_csv_path(ssis_csv)

    site_dfs = []
    if os.path.exists(base_path):
        df_base = pd.read_csv(base_path)
        if "source_csv" not in df_base.columns:
            df_base["source_csv"] = os.path.basename(base_path)
        site_dfs.append(df_base)
    if os.path.exists(custom_path):
        df_custom = pd.read_csv(custom_path)
        if "source_csv" not in df_custom.columns:
            df_custom["source_csv"] = os.path.basename(custom_path)
        site_dfs.append(df_custom)

    sites_df = pd.concat(site_dfs, ignore_index=True) if site_dfs else pd.DataFrame()
    ssis_df = pd.read_csv(ssis_path) if os.path.exists(ssis_path) else pd.DataFrame()
    return sites_df, ssis_df


def _get_site_horizon_profile(lat: float, lon: float) -> np.ndarray:
    cache_key = (round(float(lat), 6), round(float(lon), 6))
    if cache_key in _HORIZON_CACHE:
        return _HORIZON_CACHE[cache_key]

    if inspect_site_horizon is None:
        raise ImportError(
            "Could not import `inspect_site_horizon` from `inspect_point.py`."
        )

    params = inspect.signature(inspect_site_horizon).parameters
    if "lat" in params and "lon" in params:
        result = inspect_site_horizon(lat=float(lat), lon=float(lon))
    else:
        result = inspect_site_horizon(float(lat), float(lon))

    horizon_arr = None
    if isinstance(result, np.ndarray) and result.shape == (360,):
        horizon_arr = result
    elif isinstance(result, (list, tuple)):
        if len(result) == 360 and not isinstance(result[0], (list, tuple, dict, np.ndarray)):
            horizon_arr = np.asarray(result, dtype=np.float64)
        else:
            for item in result:
                if isinstance(item, np.ndarray) and item.shape == (360,):
                    horizon_arr = item
                    break
                if isinstance(item, (list, tuple)) and len(item) == 360:
                    horizon_arr = np.asarray(item, dtype=np.float64)
                    break
    elif isinstance(result, dict):
        for key in ("horizonAlt_Az", "horizon_profile", "horizon", "horizon_angles", "elevations", "profile"):
            if key in result:
                horizon_arr = np.asarray(result[key], dtype=np.float64)
                break
        if horizon_arr is None:
            for val in result.values():
                if isinstance(val, (list, tuple, np.ndarray)):
                    arr = np.asarray(val)
                    if arr.shape == (360,):
                        horizon_arr = arr.astype(np.float64)
                        break

    if horizon_arr is None or horizon_arr.shape != (360,):
        raise ValueError(f"Could not extract 360-element horizon profile for ({lat}, {lon}).")

    horizon_arr = np.asarray(horizon_arr, dtype=np.float64)
    _HORIZON_CACHE[cache_key] = horizon_arr
    return horizon_arr


def _get_clean_ephemeris(
    target: str,
    lon: float,
    lat: float,
    alt_km: float,
    time_range: tuple[str, str],
    time_resolution: tuple[int, str],
) -> tuple[np.ndarray, np.ndarray]:
    # Round coordinates to ~0.01 deg (~300m on Moon) and alt to 0.1 km to avoid duplicate Horizons HTTP calls
    cache_key = (
        target,
        round(float(lon), 2),
        round(float(lat), 2),
        round(float(alt_km), 1),
        time_range,
        time_resolution,
    )
    if cache_key in _EPHEMERIS_CACHE:
        return _EPHEMERIS_CACHE[cache_key]

    coords = (float(lon), float(lat), float(alt_km))
    eph_data = GetHorizonsEphemeris(target, coords, time_range, time_resolution)

    if eph_data is None or len(eph_data) == 0:
        empty = (np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64))
        _EPHEMERIS_CACHE[cache_key] = empty
        return empty

    az_series = pd.to_numeric(eph_data["azimuth"], errors="coerce")
    alt_series = pd.to_numeric(eph_data["altitude"], errors="coerce")
    valid_mask = az_series.notna() & alt_series.notna()

    body_az = az_series[valid_mask].to_numpy(dtype=np.float64) % 360.0
    body_alt = alt_series[valid_mask].to_numpy(dtype=np.float64)
    body_az = np.where(np.round(body_az) >= 360, 0.0, body_az)

    cleaned = (body_az, body_alt)
    _EPHEMERIS_CACHE[cache_key] = cleaned
    return cleaned


def _evaluate_path(
    path: list[int],
    d0: np.ndarray,
    dist_mat: np.ndarray,
    cand_vi: np.ndarray,
    r_max_km: float,
    decay_rate: float,
    return_to_lander: bool,
) -> tuple[float, float]:
    """Compute (total_score, total_distance_km) for an ordered sequence of candidate indices."""
    if not path:
        return 0.0, 0.0
    cum_d = float(d0[path[0]])
    score = float(cand_vi[path[0]]) / (1.0 + decay_rate * cum_d)
    for k in range(1, len(path)):
        cum_d += float(dist_mat[path[k - 1], path[k]])
        score += float(cand_vi[path[k]]) / (1.0 + decay_rate * cum_d)
    total_d = cum_d + (float(d0[path[-1]]) if return_to_lander else 0.0)
    if total_d > r_max_km + 1e-9:
        return -1.0, total_d
    return score, total_d


def _refine_path_2opt_and_insert(
    path: list[int],
    d0: np.ndarray,
    dist_mat: np.ndarray,
    cand_vi: np.ndarray,
    r_max_km: float,
    decay_rate: float,
    return_to_lander: bool,
) -> tuple[float, list[int]]:
    """Shorten path via 2-opt swaps, then greedily insert unvisited SSIs within budget."""
    m = len(cand_vi)
    cur_path = list(path)
    cur_score, _ = _evaluate_path(cur_path, d0, dist_mat, cand_vi, r_max_km, decay_rate, return_to_lander)
    if cur_score < 0.0:
        return 0.0, []

    improved = True
    while improved and len(cur_path) >= 3:
        improved = False
        n = len(cur_path)
        for i in range(n - 1):
            for j in range(i + 2, n + 1):
                candidate = cur_path[:i] + cur_path[i:j][::-1] + cur_path[j:]
                sc, _ = _evaluate_path(candidate, d0, dist_mat, cand_vi, r_max_km, decay_rate, return_to_lander)
                if sc > cur_score + 1e-9:
                    cur_path = candidate
                    cur_score = sc
                    improved = True

    visited = set(cur_path)
    while len(visited) < m:
        best_gain = 0.0
        best_new_path = None
        best_new_score = cur_score
        best_node = -1

        for node in range(m):
            if node in visited:
                continue
            for pos in range(len(cur_path) + 1):
                candidate = cur_path[:pos] + [node] + cur_path[pos:]
                sc, _ = _evaluate_path(candidate, d0, dist_mat, cand_vi, r_max_km, decay_rate, return_to_lander)
                if sc > best_new_score + 1e-9:
                    best_gain = sc - cur_score
                    best_new_score = sc
                    best_new_path = candidate
                    best_node = node

        if best_new_path is None or best_gain <= 0.0:
            break
        cur_path = best_new_path
        cur_score = best_new_score
        visited.add(best_node)

    return cur_score, cur_path


def solve_best_rover_route_with_path(
    site_xyz: np.ndarray,
    cand_xyz: np.ndarray,
    cand_vi: np.ndarray,
    r_max_km: float,
    decay_rate: float = 0.15,
    return_to_lander: bool = False,
    exact_limit: int = 12,
    beam_width: int = 96,
) -> tuple[float, list[int]]:
    """
    Solve for the maximum-weight rover traverse starting at `site_xyz` whose total
    travel distance does not exceed `r_max_km`.
    Returns (best_raw_weight, ordered_candidate_indices).
    """
    m = len(cand_vi)
    if m == 0 or r_max_km <= 0.0:
        return 0.0, []

    d0 = np.linalg.norm(cand_xyz - site_xyz[None, :], axis=1)
    roundtrip_mult = 2.0 if return_to_lander else 1.0

    if m == 1:
        if d0[0] * roundtrip_mult <= r_max_km:
            return float(cand_vi[0] / (1.0 + decay_rate * d0[0])), [0]
        return 0.0, []

    diff = cand_xyz[:, None, :] - cand_xyz[None, :, :]
    dist_mat = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))

    # 1. Exact Bitmask DP with parent backtracking (m <= exact_limit)
    if m <= exact_limit:
        n_states = 1 << m
        dp_dist = np.full((n_states, m), np.inf, dtype=np.float64)
        dp_score = np.full((n_states, m), -1.0, dtype=np.float64)
        parent = np.full((n_states, m), -1, dtype=np.int32)

        best_score = 0.0
        best_state = (-1, -1)

        for i in range(m):
            leg = float(d0[i])
            if leg * roundtrip_mult <= r_max_km:
                mask = 1 << i
                sc = float(cand_vi[i] / (1.0 + decay_rate * leg))
                dp_dist[mask, i] = leg
                dp_score[mask, i] = sc
                if sc > best_score:
                    best_score = sc
                    best_state = (mask, i)

        for mask in range(1, n_states):
            if (mask & (mask - 1)) == 0:
                continue

            temp_last = mask
            while temp_last:
                lsb_last = temp_last & -temp_last
                last = lsb_last.bit_length() - 1
                temp_last ^= lsb_last

                prev_mask = mask ^ lsb_last
                vi_last = float(cand_vi[last])
                d_ret = float(d0[last]) if return_to_lander else 0.0

                best_st_sc = -1.0
                best_st_d = np.inf
                best_prev = -1

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
                        if new_sc > best_st_sc or (new_sc == best_st_sc and new_d < best_st_d):
                            best_st_sc = new_sc
                            best_st_d = new_d
                            best_prev = prev

                if best_st_sc >= 0.0:
                    dp_dist[mask, last] = best_st_d
                    dp_score[mask, last] = best_st_sc
                    parent[mask, last] = best_prev
                    if best_st_sc > best_score:
                        best_score = best_st_sc
                        best_state = (mask, last)

        if best_state[0] == -1:
            return 0.0, []

        path: list[int] = []
        cur_mask, cur_last = best_state
        while cur_last != -1:
            path.append(cur_last)
            prev_last = int(parent[cur_mask, cur_last])
            cur_mask ^= 1 << cur_last
            cur_last = prev_last
        path.reverse()
        return best_score, path

    # 2. Multi-Tradeoff Distance-Aware Beam Search + 2-Opt & Insertion (m > exact_limit)
    best_score = 0.0
    best_path: list[int] = []
    avg_vi = float(np.mean(cand_vi))

    for dist_penalty in (0.0, 0.5 * avg_vi / max(1.0, r_max_km / 6.0), 1.5 * avg_vi / max(1.0, r_max_km / 6.0)):
        beam: list[tuple[float, float, float, int, int, list[int]]] = []
        pass_best_sc = 0.0
        pass_best_path: list[int] = []

        for i in range(m):
            leg = float(d0[i])
            if leg * roundtrip_mult <= r_max_km:
                sc = float(cand_vi[i] / (1.0 + decay_rate * leg))
                rank_key = sc - dist_penalty * leg
                beam.append((rank_key, sc, leg, i, 1 << i, [i]))
                if sc > pass_best_sc:
                    pass_best_sc = sc
                    pass_best_path = [i]

        beam.sort(key=lambda x: x[0], reverse=True)
        beam = beam[:beam_width]

        for _ in range(2, m + 1):
            next_states: dict[tuple[int, int], tuple[float, float, float, int, int, list[int]]] = {}
            for _, sc, cum_d, last, mask, path in beam:
                for nxt in range(m):
                    if mask & (1 << nxt):
                        continue
                    new_d = cum_d + float(dist_mat[last, nxt])
                    d_ret = float(d0[nxt]) if return_to_lander else 0.0
                    if new_d + d_ret <= r_max_km:
                        new_sc = sc + float(cand_vi[nxt]) / (1.0 + decay_rate * new_d)
                        rank_key = new_sc - dist_penalty * new_d
                        new_mask = mask | (1 << nxt)
                        key = (new_mask, nxt)
                        prev_best = next_states.get(key)
                        if prev_best is None or new_sc > prev_best[1]:
                            new_path = path + [nxt]
                            next_states[key] = (rank_key, new_sc, new_d, nxt, new_mask, new_path)
                            if new_sc > pass_best_sc:
                                pass_best_sc = new_sc
                                pass_best_path = new_path

            if not next_states:
                break
            beam = sorted(next_states.values(), key=lambda x: x[0], reverse=True)[:beam_width]

        if pass_best_path:
            polished_sc, polished_path = _refine_path_2opt_and_insert(
                pass_best_path, d0, dist_mat, cand_vi, r_max_km, decay_rate, return_to_lander
            )
            if polished_sc > pass_best_sc:
                pass_best_sc = polished_sc
                pass_best_path = polished_path

        if pass_best_sc > best_score:
            best_score = pass_best_sc
            best_path = pass_best_path

    return best_score, best_path


def solve_best_rover_route(
    site_xyz: np.ndarray,
    cand_xyz: np.ndarray,
    cand_vi: np.ndarray,
    r_max_km: float,
    decay_rate: float = 0.15,
    return_to_lander: bool = False,
    exact_limit: int = 12,
    beam_width: int = 96,
) -> tuple[float, int]:
    """Return (best_raw_weight, visited_stop_count) for the optimal route within r_max_km."""
    best_weight, path = solve_best_rover_route_with_path(
        site_xyz=site_xyz,
        cand_xyz=cand_xyz,
        cand_vi=cand_vi,
        r_max_km=r_max_km,
        decay_rate=decay_rate,
        return_to_lander=return_to_lander,
        exact_limit=exact_limit,
        beam_width=beam_width,
    )
    return best_weight, len(path)


def scores_calculator(
    sites_df: pd.DataFrame | None = None,
    ssis_df: pd.DataFrame | None = None,
    base_csv: str = BASE_SITES_CSV,
    custom_csv: str = CUSTOM_SITES_CSV,
    ssis_csv: str = SSIS_CSV,
    max_slope_limit_deg: float = 15.0,
    mean_slope_limit_deg: float = 10.0,
    roughness_limit_m: float = 25.0,
    slope_weight: float = 0.60,
    r_max_km: float = 12.0,
    decay_rate: float = 0.15,
    science_half_sat: float = 5.0,
    return_to_lander: bool = False,
    time_range: tuple[str, str] = ("2027-01-01", "2027-01-30"),
    time_resolution: tuple[int, str] = (1, "hour"),
    filter_infeasible: bool = False,
) -> pd.DataFrame:
    """
    Main batch scoring function for all landing sites in both CSV files
    (`landing_sites.csv` and `user_inspected_sites.csv`).

    Computes all 4 scores (`s_safety`, `s_science`, `s_sun`, `s_earth`) for every site.
    """
    if sites_df is None or ssis_df is None:
        loaded_sites, loaded_ssis = load_sites_and_ssis(
            base_csv=base_csv,
            custom_csv=custom_csv,
            ssis_csv=ssis_csv,
        )
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
        df = sites_df.copy().reset_index(drop=True)
        feasible_out = feasible

    n_sites = len(df)
    if n_sites == 0:
        df["feasible"] = pd.Series(dtype=bool)
        df["s_safety"] = pd.Series(dtype=float)
        df["reachable_ssi_count"] = pd.Series(dtype=int)
        df["visited_ssi_count"] = pd.Series(dtype=int)
        df["s_science"] = pd.Series(dtype=float)
        df["s_sun"] = pd.Series(dtype=float)
        df["s_earth"] = pd.Series(dtype=float)
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

    # 3. Astronomical Visibility Scores for Sun & Earth
    lats = df["lat"].to_numpy(dtype=np.float64)
    lons = df["lon"].to_numpy(dtype=np.float64)
    alt_kms = (
        df["elevation_m"].to_numpy(dtype=np.float64) / 1000.0
        if "elevation_m" in df.columns
        else np.zeros(n_sites, dtype=np.float64)
    )

    s_sun = np.zeros(n_sites, dtype=np.float64)
    s_earth = np.zeros(n_sites, dtype=np.float64)

    for idx in range(n_sites):
        horizon_alt_az = _get_site_horizon_profile(lats[idx], lons[idx])

        sun_az, sun_alt = _get_clean_ephemeris(
            target="Sun",
            lon=lons[idx],
            lat=lats[idx],
            alt_km=alt_kms[idx],
            time_range=time_range,
            time_resolution=time_resolution,
        )
        if sun_az.size > 0:
            sun_angles = FindBodyHorizonAngles(horizon_alt_az, (sun_az, sun_alt))
            if sun_angles.size > 0:
                s_sun[idx] = float(np.sum(sun_angles > 0) / sun_angles.size)

        earth_az, earth_alt = _get_clean_ephemeris(
            target="Earth",
            lon=lons[idx],
            lat=lats[idx],
            alt_km=alt_kms[idx],
            time_range=time_range,
            time_resolution=time_resolution,
        )
        if earth_az.size > 0:
            earth_angles = FindBodyHorizonAngles(horizon_alt_az, (earth_az, earth_alt))
            if earth_angles.size > 0:
                s_earth[idx] = float(np.sum(earth_angles > 0) / earth_angles.size)

    df["feasible"] = feasible_out
    df["s_safety"] = np.round(np.clip(s_safety, 0.0, 1.0), 4)
    df["reachable_ssi_count"] = reachable_counts
    df["visited_ssi_count"] = visited_counts
    df["s_science"] = np.round(np.clip(s_science, 0.0, 1.0), 4)
    df["s_sun"] = np.round(np.clip(s_sun, 0.0, 1.0), 4)
    df["s_earth"] = np.round(np.clip(s_earth, 0.0, 1.0), 4)
    return df





if __name__ == "__main__":
    import time

    t0 = time.perf_counter()
    scored_df = scores_calculator()
    dt = (time.perf_counter() - t0) * 1000.0
    print(f"Scored {len(scored_df):,} landing sites across both CSVs in {dt:.1f} ms")
    cols = [
        "site_id",
        "tier",
        "s_safety",
        "reachable_ssi_count",
        "visited_ssi_count",
        "s_science",
        "s_sun",
        "s_earth",
    ]
    available_cols = [c for c in cols if c in scored_df.columns]
    print(scored_df[available_cols].head(10).to_string(index=False))