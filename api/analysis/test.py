import os
import sys
import argparse
from pathlib import Path
import pandas as pd

# Walk up from api/analysis/test.py to the repository root so "processed_data/..." resolves cleanly
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(
    (p for p in CURRENT_FILE.parents if (p / "processed_data").exists() or (p / ".git").exists()),
    CURRENT_FILE.parents[2],
)
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.analysis.inspect_point import main, CSV_COLUMNS, CUSTOM_SITES_CSV  # noqa: E402


# Default coordinate: Mare Tranquillitatis (streams directly from NASA SVS ldem_64.tif over HTTPS)
DEFAULT_LAT = 8.3800
DEFAULT_LON = 33.2500


def run_test(lat: float, lon: float) -> None:
    print(f"Inspecting lunar point (lat={lat}, lon={lon}) via remote NASA stream...")
    row_list = main(lat=lat, lon=lon)

    if not row_list:
        print("No valid elevation data returned for this coordinate.")
        return

    print("\n--- Returned 12-Element List ---")
    print(row_list)

    print("\n--- Mapped Columns ---")
    for col, val in zip(CSV_COLUMNS, row_list):
        print(f"  {col:<15}: {val}")

    csv_path = PROJECT_ROOT / CUSTOM_SITES_CSV
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"\n--- {CUSTOM_SITES_CSV} (Total rows: {len(df)}) ---")
        print(df.tail(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test api/analysis/inspect_point.py")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT, help="Latitude (-90 to 90)")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON, help="Longitude (-180 to 180)")
    args = parser.parse_args()

    run_test(lat=args.lat, lon=args.lon)