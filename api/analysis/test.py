import argparse
import io
import os
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

# Resolve repository root so "processed_data/..." and "raw_data/..." paths work from any directory
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(
    (p for p in CURRENT_FILE.parents if (p / "processed_data").exists() or (p / ".git").exists()),
    CURRENT_FILE.parents[2],
)
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.analysis.inspect_point import (  # noqa: E402
    inspect_site_terrain,
    inspect_site_horizon,
    CSV_COLUMNS,
)
from api.analysis.score_calculator import score_calculator  # noqa: E402
from api.analysis.render_route_map import render_rover_route_image  # noqa: E402


def test_score_and_show_image(
    lat: float = -89.6800,
    lon: float = -130.0000,
    r_max_km: float = 12.0,
    decay_rate: float = 0.15,
    science_half_sat: float = 2.5,
    return_to_lander: bool = False,
    time_range: tuple[str, str] = ("2027-01-01 00:00", "2027-01-30 00:00"),
    time_resolution: tuple[int, str] = (1, "hour"),
    output_path: str | None = None,
    show_image: bool = True,
) -> dict:
    """
    Inspects (lat, lon), calculates all 4 mission scores (safety, science, sun, earth),
    renders the 2D lunar surface landing zone & rover route map, and displays the image.
    """
    # 1. Inspect terrain & 360-degree horizon profile
    t0 = time.perf_counter()
    site_row = inspect_site_terrain(lat=lat, lon=lon)
    if not site_row:
        raise ValueError(f"No valid DEM elevation data at (lat={lat}, lon={lon}).")
    terrain_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    horizons_list = inspect_site_horizon(lat=lat, lon=lon, site_id=site_row[0])
    horizons_arr = np.asarray(horizons_list, dtype=np.float64)
    horizon_ms = (time.perf_counter() - t1) * 1000.0

    # 2. Calculate all 4 normalized scores in [0.0, 1.0]
    t2 = time.perf_counter()
    s_safety, s_science, s_sun, s_earth = score_calculator(
        site_row=site_row,
        r_max_km=r_max_km,
        decay_rate=decay_rate,
        science_half_sat=science_half_sat,
        return_to_lander=return_to_lander,
        time_range=time_range,
        time_resolution=time_resolution,
        horizon_profile=horizons_arr,
    )
    score_ms = (time.perf_counter() - t2) * 1000.0

    # 3. Render 2D landing zone & rover path map in memory
    t3 = time.perf_counter()
    render_res = render_rover_route_image(
        site_row=site_row,
        r_max_km=r_max_km,
        decay_rate=decay_rate,
        return_to_lander=return_to_lander,
        output_path=output_path,
    )
    render_ms = (time.perf_counter() - t3) * 1000.0

    # 4. Print summary diagnostics
    print(f"=== Lunar Site Evaluation ({lat}, {lon}) ===")
    print(
        f"Timings -> Terrain: {terrain_ms:.1f} ms | Horizon: {horizon_ms:.1f} ms | "
        f"Scores: {score_ms:.1f} ms | Render: {render_ms:.1f} ms"
    )

    print("\n[1] Site Terrain Metadata:")
    for col, val in zip(CSV_COLUMNS, site_row):
        print(f"  {col:<15}: {val}")

    print("\n[2] Mission Scores [0.0 - 1.0]:")
    print(f"  s_safety        : {s_safety:.4f}")
    print(f"  s_science       : {s_science:.4f}")
    print(f"  s_sun           : {s_sun:.4f}")
    print(f"  s_earth         : {s_earth:.4f}")

    print("\n[3] Rover Traverse Summary:")
    print(f"  Range Budget    : {r_max_km} km (return_to_lander={return_to_lander})")
    print(f"  Raw Science Wt  : {render_res['raw_science_weight']}")
    print(f"  Visited SSIs ({len(render_res['visited_ssi_ids'])}): {render_res['visited_ssi_ids']}")
    if output_path:
        print(f"  Saved PNG       : {output_path}")

    # 5. Display image directly from RAM buffer
    if show_image:
        img = Image.open(io.BytesIO(render_res["image_bytes"]))
        img.show(title=f"Rover Route ({lat}, {lon})")

    return {
        "site": dict(zip(CSV_COLUMNS, site_row)),
        "scores": {
            "s_safety": s_safety,
            "s_science": s_science,
            "s_sun": s_sun,
            "s_earth": s_earth,
        },
        "visited_ssi_ids": render_res["visited_ssi_ids"],
        "raw_science_weight": render_res["raw_science_weight"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a lunar site and display its 2D landing zone & route map.")
    parser.add_argument("--lat", type=float, default=-89.6800, help="Latitude (-90 to 90)")
    parser.add_argument("--lon", type=float, default=-130.0000, help="Longitude (-180 to 180)")
    parser.add_argument("--range-km", type=float, default=12.0, help="Max rover travel distance (km)")
    parser.add_argument("--return-to-lander", action="store_true", help="Require closed loop back to lander")
    parser.add_argument("--out", type=str, default=None, help="Optional file path to also save the PNG")
    parser.add_argument("--no-show", action="store_true", help="Skip opening the image viewer window")
    args = parser.parse_args()

    test_score_and_show_image(
        lat=args.lat,
        lon=args.lon,
        r_max_km=args.range_km,
        return_to_lander=args.return_to_lander,
        output_path=args.out,
        show_image=not args.no_show,
    )