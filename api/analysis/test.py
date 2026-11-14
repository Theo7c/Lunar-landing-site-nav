import argparse
import os
import sys
import time
from pathlib import Path
import pandas as pd

# Resolve repository root so "processed_data/..." and "raw_data/..." paths work from any directory
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(
    (p for p in CURRENT_FILE.parents if (p / "processed_data").exists() or (p / ".git").exists()),
    CURRENT_FILE.parents[2],
)
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.analysis.inspect_point import CSV_COLUMNS, BASE_SITES_CSV  # noqa: E402
from api.analysis.render_route_map import render_rover_route_image  # noqa: E402


def test_render_route_map(
    lat: float = -89.6800,
    lon: float = -130.0000,
    site_id: str | None = None,
    r_max_km: float = 120.0,
    decay_rate: float = 0.15,
    return_to_lander: bool = False,
    output_path: str = "output/test_rover_route.png",
) -> dict:
    """
    Tests `render_rover_route_image` using either a `site_id` from landing_sites.csv
    or a custom `(lat, lon)` coordinate, verifying that the PNG is created on disk.
    """
    site_row = None

    if site_id is not None:
        sites_path = PROJECT_ROOT / BASE_SITES_CSV
        if not sites_path.exists():
            raise FileNotFoundError(f"Cannot look up site_id='{site_id}'; missing {BASE_SITES_CSV}.")
        df = pd.read_csv(sites_path)
        match = df[df["site_id"] == site_id]
        if match.empty:
            raise ValueError(f"site_id='{site_id}' not found in {BASE_SITES_CSV}.")
        site_row = [match.iloc[0][col] for col in CSV_COLUMNS]
        print(f"Rendering route map for grid site '{site_id}' (lat={site_row[2]}, lon={site_row[3]})...")
    else:
        print(f"Rendering route map for coordinate (lat={lat}, lon={lon})...")

    t0 = time.perf_counter()
    result = render_rover_route_image(
        lat=None if site_row is not None else lat,
        lon=None if site_row is not None else lon,
        site_row=site_row,
        r_max_km=r_max_km,
        decay_rate=decay_rate,
        return_to_lander=return_to_lander,
        output_path=output_path,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    img_file = Path(result["image_path"])
    assert img_file.exists(), f"Expected output image was not created at {img_file}"
    file_size_kb = img_file.stat().st_size / 1024.0
    assert file_size_kb > 0.0, f"Output image at {img_file} is empty (0 bytes)."

    print(f"Completed in {elapsed_ms:.1f} ms")
    print(f"  Image Path        : {result['image_path']} ({file_size_kb:.1f} KB)")
    print(f"  Coordinates       : ({result['lat']}, {result['lon']})")
    print(f"  Raw Science Weight: {result['raw_science_weight']}")
    print(f"  Visited SSIs ({len(result['visited_ssi_ids'])}): {result['visited_ssi_ids']}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test api/analysis/render_route_map.py")
    parser.add_argument("--lat", type=float, default=-89.6800, help="Latitude (-90 to 90)")
    parser.add_argument("--lon", type=float, default=-130.0000, help="Longitude (-180 to 180)")
    parser.add_argument("--site-id", type=str, default=None, help="Optional site_id from landing_sites.csv")
    parser.add_argument("--range-km", type=float, default=12.0, help="Max rover travel distance (km)")
    parser.add_argument("--return-to-lander", action="store_true", help="Require closed loop back to lander")
    parser.add_argument("--out", type=str, default="output/test_rover_route.png", help="Output PNG path")
    args = parser.parse_args()

    test_render_route_map(
        lat=args.lat,
        lon=args.lon,
        site_id=args.site_id,
        r_max_km=args.range_km,
        return_to_lander=args.return_to_lander,
        output_path=args.out,
    )