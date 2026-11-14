import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd

# Ensure the project root is on sys.path and set as the working directory
# so relative paths ("processed_data/...") inside inspect_point.py always work.
PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "backend" else Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.inspect_point import inspect_and_save_point, CUSTOM_SITES_CSV  # noqa: E402


# ==============================================================================
# DEFAULT TEST COORDINATE (Change these or pass --lat / --lon in terminal)
# ==============================================================================
# Example 1 (South Pole, 20 m/px NASA DEM): Shackleton-de Gerlache Ridge
DEFAULT_LAT = -78.4521
DEFAULT_LON = -130.4890

# Example 2 (Global, ldem_16 DEM): Mare Tranquillitatis Pit -> lat=8.380, lon=33.250


def run_test(lat: float, lon: float) -> None:
    print(f"Streaming exact NASA terrain data for (lat={lat}, lon={lon})...")
    result = inspect_and_save_point(lat=lat, lon=lon)

    print("\n--- NASA Inspection Result ---")
    print(json.dumps(result, indent=2, default=str))

    # Verify the file on disk in processed_data/user_inspected_sites.csv[cite: 3]
    csv_path = PROJECT_ROOT / CUSTOM_SITES_CSV
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"\n--- {CUSTOM_SITES_CSV} (Total saved custom sites: {len(df)}) ---")
        print(df.tail(3).to_string(index=False))
    else:
        print(f"\nNote: {CUSTOM_SITES_CSV} was not created (likely because max_slope_deg > 15.0° or site already existed).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test on-demand NASA lunar point inspection.")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT, help="Target latitude (-90 to 90)")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON, help="Target longitude (-180 to 180)")
    args = parser.parse_args()

    run_test(lat=args.lat, lon=args.lon)