import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Walk up from api/analysis/test.py to repository root so "processed_data/..." & "raw_data/..." resolve cleanly
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(
    (p for p in CURRENT_FILE.parents if (p / "processed_data").exists() or (p / ".git").exists()),
    CURRENT_FILE.parents[2],
)
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.analysis.inspect_point import (  # noqa: E402
    main,
    CSV_COLUMNS,
    CUSTOM_SITES_CSV,
    CUSTOM_HORIZONS_CSV,
)

DEFAULT_LAT = 8.3800
DEFAULT_LON = 33.2500


def run_test(lat: float, lon: float) -> None:
    print(f"Inspecting lunar point (lat={lat}, lon={lon})...")
    t0 = time.perf_counter()
    site_row, horizons = main(lat=lat, lon=lon)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if not site_row:
        print("No valid elevation data returned for this coordinate.")
        return

    print(f"Completed in {elapsed_ms:.1f} ms")

    print("\n--- Returned List 1: 12-Element Site Row ---")
    print(site_row)
    for col, val in zip(CSV_COLUMNS, site_row):
        print(f"  {col:<15}: {val}")

    h_arr = np.asarray(horizons, dtype=float)
    print(f"\n--- Returned List 2: 360-Degree Horizon Profile (len={len(horizons)}) ---")
    print(f"  Min angle (deg) : {h_arr.min():.2f}")
    print(f"  Mean angle (deg): {h_arr.mean():.2f}")
    print(f"  Max angle (deg) : {h_arr.max():.2f}")
    print(f"  First 12 az     : {horizons[:12]}")

    sites_path = PROJECT_ROOT / CUSTOM_SITES_CSV
    if sites_path.exists():
        df = pd.read_csv(sites_path)
        print(f"\n--- {CUSTOM_SITES_CSV} (Total rows: {len(df)}) ---")
        print(df.tail(2).to_string(index=False))

    horizons_path = PROJECT_ROOT / CUSTOM_HORIZONS_CSV
    if horizons_path.exists():
        hdf = pd.read_csv(horizons_path)
        preview_cols = ["site_id", "az_000", "az_090", "az_180", "az_270", "az_359"]
        print(f"\n--- {CUSTOM_HORIZONS_CSV} (Total rows: {len(hdf)}) ---")
        print(hdf[preview_cols].tail(2).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test api/analysis/inspect_point.py")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT, help="Latitude (-90 to 90)")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON, help="Longitude (-180 to 180)")
    args = parser.parse_args()

    run_test(lat=args.lat, lon=args.lon)