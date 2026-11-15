import argparse
import io
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
for rel_path in ("", "..", os.path.join("..", "..")):
    candidate_dir = os.path.abspath(os.path.join(CURRENT_DIR, rel_path))
    if candidate_dir not in sys.path:
        sys.path.insert(0, candidate_dir)

try:
    from api.analysis.inspect_point import (
        inspect_site_terrain,
        get_dataset,
        normalize_lon,
        SP_SOURCES,
        GLOBAL_SOURCES,
        R_MOON_M,
    )
except ImportError:
    from inspect_point import (
        inspect_site_terrain,
        get_dataset,
        normalize_lon,
        SP_SOURCES,
        GLOBAL_SOURCES,
        R_MOON_M,
    )

try:
    from api.analysis.scores_calculator import (
        SSIS_CSV,
        DEFAULT_MAX_STOPS,
        solve_best_rover_route_with_path,
        _resolve_csv_path,
    )
except ImportError:
    from scores_calculator import (
        SSIS_CSV,
        DEFAULT_MAX_STOPS,
        solve_best_rover_route_with_path,
        _resolve_csv_path,
    )

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

_NICE_SCALE_STEPS_M = np.array(
    [100, 200, 250, 500, 1_000, 2_000, 2_500, 5_000, 10_000, 20_000, 25_000, 50_000, 100_000],
    dtype=np.float64,
)


def _latlon_to_window_pixels(
    lats: np.ndarray,
    lons: np.ndarray,
    src: rasterio.DatasetReader,
    is_projected: bool,
    r0: int,
    c0: int,
) -> tuple[np.ndarray, np.ndarray]:
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


def _draw_scale_bar(ax: plt.Axes, w_px: int, h_px: int, dx_m: float) -> None:
    if w_px <= 10 or h_px <= 10 or dx_m <= 0.0:
        return

    target_m = (w_px * dx_m) * 0.22
    idx = int(np.argmin(np.abs(_NICE_SCALE_STEPS_M - target_m)))
    scale_m = float(_NICE_SCALE_STEPS_M[idx])
    bar_px = scale_m / dx_m

    if bar_px > w_px * 0.40 and idx > 0:
        scale_m = float(_NICE_SCALE_STEPS_M[idx - 1])
        bar_px = scale_m / dx_m
    elif bar_px < w_px * 0.10 and idx < len(_NICE_SCALE_STEPS_M) - 1:
        scale_m = float(_NICE_SCALE_STEPS_M[idx + 1])
        bar_px = scale_m / dx_m

    label = f"{int(round(scale_m / 1000.0))} km" if scale_m >= 1000.0 else f"{int(round(scale_m))} m"

    x0 = w_px * 0.06
    x1 = x0 + bar_px
    y0 = h_px * 0.92
    tick_h = max(2.0, h_px * 0.015)

    stroke = [path_effects.Stroke(linewidth=2.2, foreground="#020617"), path_effects.Normal()]

    ax.plot([x0, x1], [y0, y0], color="#ffffff", linewidth=1.2, solid_capstyle="butt", path_effects=stroke, zorder=9)
    ax.plot([x0, x0], [y0 - tick_h, y0 + tick_h], color="#ffffff", linewidth=1.1, path_effects=stroke, zorder=9)
    ax.plot([x1, x1], [y0 - tick_h, y0 + tick_h], color="#ffffff", linewidth=1.1, path_effects=stroke, zorder=9)

    txt = ax.text(
        0.5 * (x0 + x1),
        y0 - tick_h * 1.5,
        label,
        color="#ffffff",
        fontsize=6.5,
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=10,
    )
    txt.set_path_effects([path_effects.Stroke(linewidth=1.8, foreground="#020617"), path_effects.Normal()])


def render_rover_route_image(
    lat: float | None = None,
    lon: float | None = None,
    site_row: list | None = None,
    area_radius_m: float | None = None,
    r_max_km: float = 12.0,
    decay_rate: float = 0.15,
    return_to_lander: bool = False,
    max_stops: int = DEFAULT_MAX_STOPS,
    show_scale_bar: bool = True,
    output_path: str | None = None,
    ssis_csv: str = SSIS_CSV,
    size_px: int = 150,
    dpi: int = 100,
) -> dict:
    """
    Generate a borderless `size_px x size_px` (default 150x150) 2D surface image showing
    the lunar topography around the inputted point, the true-to-scale shaded landing
    dispersion zone, and the highest-weight rover route (capped at `max_stops`, default 20).
    """
    t_start = time.perf_counter()

    t_sec = time.perf_counter()
    if site_row is not None and len(site_row) >= 8:
        lat_val = float(site_row[2])
        lon_val = normalize_lon(float(site_row[3]))
        site_xyz = np.array([float(site_row[4]), float(site_row[5]), float(site_row[6])], dtype=np.float64)
        resolved_radius_m = float(area_radius_m) if area_radius_m is not None else float(site_row[7])
    elif lat is not None and lon is not None:
        inspected_row = inspect_site_terrain(lat=float(lat), lon=float(lon), area_radius_m=area_radius_m)
        if not inspected_row:
            raise ValueError(f"No valid DEM elevation data at (lat={lat}, lon={lon}).")
        lat_val = float(inspected_row[2])
        lon_val = float(inspected_row[3])
        site_xyz = np.array([float(inspected_row[4]), float(inspected_row[5]), float(inspected_row[6])], dtype=np.float64)
        resolved_radius_m = float(area_radius_m) if area_radius_m is not None else float(inspected_row[7])
    else:
        raise ValueError("Provide either `(lat, lon)` or a valid 12-element `site_row`.")
    site_lookup_ms = (time.perf_counter() - t_sec) * 1000.0

    t_sec = time.perf_counter()
    resolved_ssis = _resolve_csv_path(ssis_csv)
    ssis_df = pd.read_csv(resolved_ssis) if os.path.exists(resolved_ssis) else pd.DataFrame()
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
                max_stops=max_stops,
            )
            cand_df = ssis_df.iloc[nbr_arr].reset_index(drop=True)
    route_solve_ms = (time.perf_counter() - t_sec) * 1000.0

    t_sec = time.perf_counter()
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
    dem_read_ms = (time.perf_counter() - t_sec) * 1000.0

    t_sec = time.perf_counter()
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
    fig_inches = float(size_px) / float(dpi)

    fig, ax = plt.subplots(figsize=(fig_inches, fig_inches), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(shaded, origin="upper", interpolation="bilinear", aspect="auto")

    lander_x, lander_y = _latlon_to_window_pixels(
        np.array([lat_val]), np.array([lon_val]), src, is_projected, r0, c0
    )

    lz_rx = max(0.5, resolved_radius_m / dx_m)
    lz_ry = max(0.5, resolved_radius_m / res_m)
    landing_zone_patch = Ellipse(
        xy=(float(lander_x[0]), float(lander_y[0])),
        width=2.0 * lz_rx,
        height=2.0 * lz_ry,
        facecolor="#ef4444",
        alpha=0.28,
        edgecolor="#fca5a5",
        linewidth=0.9,
        zorder=3.5,
    )
    ax.add_patch(landing_zone_patch)

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
                s=9,
                c="#94a3b8",
                edgecolors="#0f172a",
                linewidths=0.4,
                alpha=0.85,
                zorder=3,
            )

        if visited_order:
            route_x = np.concatenate([[lander_x[0]], cand_x[visited_order]])
            route_y = np.concatenate([[lander_y[0]], cand_y[visited_order]])
            if return_to_lander:
                route_x = np.append(route_x, lander_x[0])
                route_y = np.append(route_y, lander_y[0])

            ax.plot(route_x, route_y, color="#020617", linewidth=2.2, solid_capstyle="round", zorder=4)
            ax.plot(route_x, route_y, color="#38bdf8", linewidth=1.2, solid_capstyle="round", zorder=5)

            ax.scatter(
                cand_x[visited_order],
                cand_y[visited_order],
                s=18,
                c="#facc15",
                edgecolors="#020617",
                linewidths=0.6,
                zorder=6,
            )

    ax.scatter(
        lander_x,
        lander_y,
        s=18,
        marker="^",
        c="#ef4444",
        edgecolors="#ffffff",
        linewidths=0.5,
        zorder=7,
    )

    if show_scale_bar:
        _draw_scale_bar(ax, w_px=w_px, h_px=h_px, dx_m=dx_m)

    ax.set_xlim(0, w_px - 1)
    ax.set_ylim(h_px - 1, 0)
    ax.set_axis_off()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    png_bytes = buf.getvalue()
    matplotlib_render_ms = (time.perf_counter() - t_sec) * 1000.0

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(png_bytes)

    total_ms = (time.perf_counter() - t_start) * 1000.0
    timings_ms = {
        "site_lookup_ms": round(site_lookup_ms, 2),
        "route_solve_ms": round(route_solve_ms, 2),
        "dem_read_ms": round(dem_read_ms, 2),
        "matplotlib_render_ms": round(matplotlib_render_ms, 2),
        "total_ms": round(total_ms, 2),
    }
    print(
        f"[render_route_map] ({lat_val}, {lon_val}, {size_px}x{size_px}px, "
        f"r={r_max_km}km, stops={len(visited_order)}/{max_stops}) -> {timings_ms}"
    )

    return {
        "image_bytes": png_bytes,
        "image_path": output_path,
        "width_px": size_px,
        "height_px": size_px,
        "lat": lat_val,
        "lon": lon_val,
        "area_radius_m": resolved_radius_m,
        "raw_science_weight": round(raw_weight, 4),
        "visited_ssi_ids": (
            cand_df.iloc[visited_order]["ssi_id"].tolist()
            if (visited_order and "ssi_id" in cand_df.columns)
            else visited_order
        ),
        "timings_ms": timings_ms,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a 150x150 2D lunar surface rover route map.")
    parser.add_argument("--lat", type=float, default=-89.6800, help="Latitude (-90 to 90)")
    parser.add_argument("--lon", type=float, default=-130.0000, help="Longitude (-180 to 180)")
    parser.add_argument("--area-radius-m", type=float, default=None, help="Landing zone dispersion radius (m)")
    parser.add_argument("--range-km", type=float, default=12.0, help="Max rover travel distance (km)")
    parser.add_argument("--max-stops", type=int, default=DEFAULT_MAX_STOPS, help="Max science stops per mission")
    parser.add_argument("--size-px", type=int, default=150, help="Output image width and height in pixels")
    parser.add_argument("--return-to-lander", action="store_true", help="Require closed loop back to lander")
    parser.add_argument("--no-scale-bar", action="store_true", help="Hide the scale bar")
    parser.add_argument("--out", type=str, default="output/rover_route.png", help="Output PNG path")
    args = parser.parse_args()

    res = render_rover_route_image(
        lat=args.lat,
        lon=args.lon,
        area_radius_m=args.area_radius_m,
        r_max_km=args.range_km,
        max_stops=args.max_stops,
        size_px=args.size_px,
        return_to_lander=args.return_to_lander,
        show_scale_bar=not args.no_scale_bar,
        output_path=args.out,
    )
    print(f"Saved {res['width_px']}x{res['height_px']} route image to {res['image_path']}")