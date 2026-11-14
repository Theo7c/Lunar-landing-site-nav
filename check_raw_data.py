import os
import json
import numpy as np
import pandas as pd
import rasterio
from PIL import Image

print("=== 1. CHECKING RASTER DEMs ===")
dem_files = [
    "raw_data/south_pole/LDEM_80S_80MPP_ADJ.tiff",
    "raw_data/ldem_16.tif",
]

for path in dem_files:
    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        continue
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        print(f"\nFile: {path}")
        print(f"  Shape: {arr.shape} | Res: {src.res} | CRS: {src.crs}")
        print(f"  Bounds: {src.bounds}")
        print(f"  Min/Max Value: {np.nanmin(arr):.3f} to {np.nanmax(arr):.3f}")
        if np.nanmax(np.abs(arr)) < 50:
            print("  -> UNITS DETECTED: Kilometers (multiply by 1000.0 for meters)")
        else:
            print("  -> UNITS DETECTED: Meters")

print("\n=== 2. CHECKING POLAR PSR CSV ===")
psr_path = "raw_data/south_pole/LPSR_80S_20MPP_ADJ_1km2.csv"
if os.path.exists(psr_path):
    df = pd.read_csv(psr_path)
    print(f"File: {psr_path} | Rows: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")
    print("  First 2 rows:")
    print(df.head(2).to_string(index=False))
else:
    print(f"[MISSING] {psr_path}")

print("\n=== 3. CHECKING TEXTURE & GEOJSON ===")
tex_path = "raw_data/lroc_color_2k.jpg"
if os.path.exists(tex_path):
    with Image.open(tex_path) as img:
        print(f"File: {tex_path} | Size: {img.size} | Mode: {img.mode}")

geo_path = "raw_data/my-features.geojson"
if os.path.exists(geo_path):
    with open(geo_path) as f:
        gj = json.load(f)
    print(f"File: {geo_path} | Features: {len(gj.get('features', []))}")