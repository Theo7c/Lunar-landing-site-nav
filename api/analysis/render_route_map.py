import argparse
import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import transform
from rasterio.windows import Window
from scipy.spatial import cKDTree

from inspect_point import (
    main as inspect_point,
    get_dataset,
    normalize_lon,
    SP_SOURCES,
    GLOBAL_SOURCES,
    R_MOON_M,
)
from scores_calculator import SSIS_CSV

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


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

    # 1. 2-opt local search to eliminate crossings and reduce cumulative distance
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

    # 2. Iterative best-insertion of remaining unvisited SSIs within r_max_km
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

    # Run beam search across 3 distance-penalty weights so short-hop dense tours aren't crowded out
    for dist_penalty in (0.0, 0.5 * avg_vi / max(1.0, r_max_km / 6.0), 1.5 * avg_vi / max(1.0, r_max_km / 6.0)):
        beam: list[tuple[float, float, float, int, int, list[int]]] = []
        for i in range(m):
            leg = float(d0[i])
            if leg * roundtrip_mult <= r_max_km:
                sc = float(cand_vi[i] / (1.0 + decay_rate * leg))
                rank_key = sc - dist_penalty * leg
                beam.append((rank_key, sc, leg, i, 1 << i, [i]))
                if sc > best_score:
                    best_score = sc
                    best_path = [i]

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
                            if new_sc > best_score:
                                best_score = new_sc
                                best_path = new_path

            if not next_states:
                break
            beam = sorted(next_states.values(), key=lambda x: x[0], reverse=True)[:beam_width]

        # Polish best route from this pass with 2-opt + cheapest-insertion
        if best_path:
            polished_sc, polished_path = _refine_path_2opt_and_insert(
                best_path, d0, dist_mat, cand_vi, r_max_km, decay_rate, return_to_lander
            )
            if polished_sc > best_score:
                best_score = polished_sc
                best_path = polished_path

    return best_score, best_path


def _latlon_to_window_pixels(
    lats: np.ndarray,
    lons: np.ndarray,
    src: rasterio.DatasetReader,
    is_projected: bool,
    r0: int,
    c0: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert arrays of (lat, lon) to fractional (col, row) coordinates inside the DEM window."""
    if is_projected:
        xs, ys = transform("+proj=longlat +R=1737400 +no_defs", src.crs, lons.tolist(), lats.tolist())
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        return np.asarray(cols, dtype=np.float64) - c0, np.asarray(rows, dtype=np.float64) - r0

    if src.crs is not None:
        q_lons = np.where((src.bounds.left >= 0) & (lons < 0), lons + 360.0, lons)
        rows, cols = rasterio.transform.rowcol(src.transform, q_lons.tolist(), lats.tolist())
        return np.asarray(cols, dtype=np.float64) - c0, np.asarray(rows, dtype=np.float64) - r0

    rows = (90.0 - lats) / 180.0 * src.height - 0.5
    cols = ((lons + 180.0) / 360.0 * src.width - 0.5) % src.width
    return cols - c0, rows - r0


def render_rover_route_image(
    lat: float | None = None,
    lon: float | None = None,
    site_row: list | None = None,
    r_max_km: float = 12.0,
    decay_rate: float = 0.15,
    return_to_lander: bool = False,
    output_path: str = "output/rover_route.png",
    ssis_csv: str = SSIS_CSV,
    dpi: int = 300,
) -> dict:
    """
    Generate a clean 2D surface image (no labels, axes, or key) showing the lunar
    topography around the inputted point, the shaded landing dispersion zone, and
    the highest-weight rover route within `r_max_km`.
    """
    if site_row is not None and len(site_row) >= 8:
        lat_val = float(site_row[2])
        lon_val = normalize_lon(float(site_row[3]))
        site_xyz = np.array([float(site_row[4]), float(site_row[5]), float(site_row[6])], dtype=np.float64)
        area_radius_m = float(site_row[7])
    elif lat is not None and lon is not None:
        inspected_row, _ = inspect_point(lat=float(lat), lon=float(lon))
        if not inspected_row:
            raise ValueError(f"No valid DEM elevation data at (lat={lat}, lon={lon}).")
        lat_val = float(inspected_row[2])
        lon_val = float(inspected_row[3])
        site_xyz = np.array([float(inspected_row[4]), float(inspected_row[5]), float(inspected_row[6])], dtype=np.float64)
        area_radius_m = float(inspected_row[7])
    else:
        raise ValueError("Provide either `(lat, lon)` or a valid 12-element `site_row`.")

    # 1. Load SSIs and find optimal route within r_max_km
    ssis_df = pd.read_csv(ssis_csv) if os.path.exists(ssis_csv) else pd.DataFrame()
    cand_df = pd.DataFrame()
    visited_order: list[int] = []
    raw_weight = 0.0

    if not ssis_df.empty and r_max_km > 0.0:
        ssi_xyz = ssis_df[["x_km", "y_km", "z_km"]].to_numpy(dtype=np.float64)
        ssi_vi = ssis_df["v_i"].to_numpy(dtype=np.float64)
        tree = cKDTree(ssi_xyz)
        nbrs = tree.query_ball_point(site_xyz, r=float(r_max_km))
        if nbrs:
            nbr_arr = np.asarray(nbrs, dtype=np.int32)
            raw_weight, visited_order = solve_best_rover_route_with_path(
                site_xyz=site_xyz,
                cand_xyz=ssi_xyz[nbr_arr],
                cand_vi=ssi_vi[nbr_arr],
                r_max_km=float(r_max_km),
                decay_rate=float(decay_rate),
                return_to_lander=return_to_lander,
            )
            cand_df = ssis_df.iloc[nbr_arr].reset_index(drop=True)

    # 2. Extract 2D DEM window covering 1.35x the rover range
    is_south_pole = lat_val <= -80.0
    src, _ = get_dataset(SP_SOURCES if is_south_pole else GLOBAL_SOURCES)
    is_projected = is_south_pole and (src.crs is not None)

    view_radius_m = max(2_500.0, float(r_max_km) * 1_000.0 * 1.35)

    if is_projected:
        res_m = abs(src.res[0])
        xs, ys = transform("+proj=longlat +R=1737400 +no_defs", src.crs, [lon_val], [lat_val])
        row_c, col_c = src.index(xs[0], ys[0])
        half_r = int(np.ceil(view_radius_m / res_m))
        half_c = half_r
    else:
        res_m = R_MOON_M * (np.pi / src.height)
        if src.crs is not None:
            q_lon = lon_val + 360.0 if (src.bounds.left >= 0 and lon_val < 0) else lon_val
            row_c, col_c = src.index(q_lon, lat_val)
        else:
            row_c = int(np.clip((90.0 - lat_val) / 180.0 * src.height, 0, src.height - 1))
            col_c = int(np.clip((lon_val + 180.0) / 360.0 * src.width, 0, src.width - 1))
        cos_lat = max(0.15, np.cos(np.radians(lat_val)))
        half_r = int(np.ceil(view_radius_m / res_m))
        half_c = int(np.ceil(view_radius_m / (res_m * cos_lat)))

    r0 = max(0, int(row_c) - half_r)
    c0 = max(0, int(col_c) - half_c)
    r1 = min(src.height, int(row_c) + half_r + 1)
    c1 = min(src.width, int(col_c) + half_c + 1)

    dem_patch = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)
    if src.nodata is not None:
        dem_patch[dem_patch == src.nodata] = np.nan
    if np.nanmax(np.abs(dem_patch)) < 50.0:
        dem_patch *= 1000.0

    if np.isnan(dem_patch).any():
        dem_patch = np.where(np.isnan(dem_patch), float(np.nanmedian(dem_patch)), dem_patch)

    # 3. Render shaded relief + shaded landing zone + rover path (no labels/axes/key)
    dx_m = res_m if is_projected else max(50.0, res_m * np.cos(np.radians(lat_val)))
    ls = LightSource(azdeg=315, altdeg=35)
    shaded = ls.shade(
        dem_patch,
        cmap=plt.cm.gray,
        vert_exag=1.8,
        dx=dx_m,
        dy=res_m,
        blend_mode="overlay",
    )

    h_px, w_px = dem_patch.shape
    fig_w = 6.0
    fig_h = max(3.0, fig_w * (h_px / max(1, w_px)))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(shaded, origin="upper", interpolation="bilinear")

    lander_x, lander_y = _latlon_to_window_pixels(
        np.array([lat_val]), np.array([lon_val]), src, is_projected, r0, c0
    )

    # Shaded landing zone ellipse (true physical radius, with minimum visual floor at wide zooms)
    min_vis_px = max(7.0, min(w_px, h_px) * 0.025)
    lz_rx = max(area_radius_m / dx_m, min_vis_px)
    lz_ry = max(area_radius_m / res_m, min_vis_px)
    landing_zone_patch = Ellipse(
        xy=(float(lander_x[0]), float(lander_y[0])),
        width=2.0 * lz_rx,
        height=2.0 * lz_ry,
        facecolor="#ef4444",
        alpha=0.28,
        edgecolor="#fca5a5",
        linewidth=1.3,
        zorder=3.5,
    )
    ax.add_patch(landing_zone_patch)

    # Candidate SSIs in range
    if not cand_df.empty:
        cand_x, cand_y = _latlon_to_window_pixels(
            cand_df["lat"].to_numpy(dtype=np.float64),
            cand_df["lon"].to_numpy(dtype=np.float64),
            src,
            is_projected,
            r0,
            c0,
        )
        unvisited_mask = np.ones(len(cand_df), dtype=bool)
        if visited_order:
            unvisited_mask[visited_order] = False
        if np.any(unvisited_mask):
            ax.scatter(
                cand_x[unvisited_mask],
                cand_y[unvisited_mask],
                s=22,
                c="#94a3b8",
                edgecolors="#0f172a",
                linewidths=0.6,
                alpha=0.8,
                zorder=3,
            )

        if visited_order:
            route_x = np.concatenate([[lander_x[0]], cand_x[visited_order]])
            route_y = np.concatenate([[lander_y[0]], cand_y[visited_order]])
            if return_to_lander:
                route_x = np.append(route_x, lander_x[0])
                route_y = np.append(route_y, lander_y[0])

            ax.plot(route_x, route_y, color="#020617", linewidth=3.4, solid_capstyle="round", zorder=4)
            ax.plot(route_x, route_y, color="#38bdf8", linewidth=1.9, solid_capstyle="round", zorder=5)

            ax.scatter(
                cand_x[visited_order],
                cand_y[visited_order],
                s=48,
                c="#facc15",
                edgecolors="#020617",
                linewidths=1.0,
                zorder=6,
            )

    # Lander center marker
    ax.scatter(
        lander_x,
        lander_y,
        s=90,
        marker="^",
        c="#ef4444",
        edgecolors="#ffffff",
        linewidths=1.1,
        zorder=7,
    )

    ax.set_xlim(0, w_px - 1)
    ax.set_ylim(h_px - 1, 0)
    ax.set_axis_off()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return {
        "image_path": output_path,
        "lat": lat_val,
        "lon": lon_val,
        "raw_science_weight": round(raw_weight, 4),
        "visited_ssi_ids": cand_df.iloc[visited_order]["ssi_id"].tolist() if (visited_order and "ssi_id" in cand_df.columns) else visited_order,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a label-free 2D lunar surface rover route map.")
    parser.add_argument("--lat", type=float, default=-89.6800, help="Latitude (-90 to 90)")
    parser.add_argument("--lon", type=float, default=-130.0000, help="Longitude (-180 to 180)")
    parser.add_argument("--range-km", type=float, default=120.0, help="Max rover travel distance (km)")
    parser.add_argument("--return-to-lander", action="store_true", help="Require closed loop back to lander")
    parser.add_argument("--out", type=str, default="output/rover_route.png", help="Output PNG path")
    args = parser.parse_args()

    res = render_rover_route_image(
        lat=args.lat,
        lon=args.lon,
        r_max_km=args.range_km,
        return_to_lander=args.return_to_lander,
        output_path=args.out,
    )
    print(f"Saved route image to {res['image_path']} (visited={len(res['visited_ssi_ids'])}, weight={res['raw_science_weight']})")