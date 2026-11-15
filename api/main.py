from datetime import datetime
import os
import sys
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from analysis.horizon import GetHorizonsEphemeris
from analysis.inspect_point import (
    inspect_site_terrain,
    inspect_site_horizon,
    CSV_COLUMNS,
)
from analysis.score_calculator import (
    calculate_safety_score,
    calculate_ssi_score,
    calculate_astronomical_scores,
)
from analysis.scores_calculator import scores_calculator
from analysis.render_route_map import render_rover_route_image

app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _format_time_window(time_start: str, time_end: str) -> tuple[str, str, str]:
    dt0 = datetime.fromisoformat(time_start)
    t0 = dt0.strftime("%Y-%m-%d %H:%M")

    dt1 = datetime.fromisoformat(time_end)
    t1 = dt1.strftime("%Y-%m-%d %H:%M")

    delta_ms = abs((dt1 - dt0).total_seconds() * 1_000)

    MINUTE_MS = 60 * 1000
    HOUR_MS = 60 * MINUTE_MS
    DAY_MS = 24 * HOUR_MS
    MONTH_MS = 30 * DAY_MS
    YEAR_MS = 12 * MONTH_MS

    if delta_ms < 2 * HOUR_MS:
        resolution = "minute"
    elif delta_ms < 7 * DAY_MS:
        resolution = "hour"
    elif delta_ms < 3 * MONTH_MS:
        resolution = "day"
    elif delta_ms < 3 * YEAR_MS:
        resolution = "month"
    else:
        resolution = "year"

    return t0, t1, resolution


# ==============================================================================
# 1. MAIN GRAPH & FULL SITE DATA ENDPOINT
# ==============================================================================
@app.get("/")
async def root(
    lat: float,
    lng: float,
    time_start: str,
    time_end: str,
    max_slope_limit_deg: float = 15.0,
    mean_slope_limit_deg: float = 10.0,
    roughness_limit_m: float = 25.0,
    slope_weight: float = 0.60,
    r_max_km: float = 12.0,
    decay_rate: float = 0.15,
    science_half_sat: float = 5.0,
    return_to_lander: bool = False,
):
    t0, t1, resolution = _format_time_window(time_start, time_end)

    # 1. Terrain row (12-element flat list; index [8] is elevation_m)
    site_row = inspect_site_terrain(lat, lng)
    if not site_row:
        raise HTTPException(status_code=404, detail="No valid DEM elevation data at this coordinate.")

    alt = float(site_row[8]) / 1000.0

    # 2. 360-degree horizon profile (0 to 359 deg) wrapped to 361 points for circular interpolation
    horizons_list = inspect_site_horizon(lat, lng, site_id=site_row[0])
    horizons = np.asarray(horizons_list, dtype=float)
    az_grid = np.arange(361, dtype=float)
    horizon_wrapped = np.append(horizons, horizons[0])

    # 3. Earth & Sun ephemeris time-series for frontend graphs
    df = GetHorizonsEphemeris("Earth", (lng, lat, alt), (t0, t1), (1, resolution))
    sun_df = GetHorizonsEphemeris("Sun", (lng, lat, alt), (t0, t1), (1, resolution))

    df["sunAz"] = sun_df["azimuth"]
    df["sunEl"] = sun_df["altitude"]
    df["earthAz"] = df["azimuth"]
    df["earthEl"] = df["altitude"]

    # Map [-180, 180] azimuths onto [0, 360) for circular horizon interpolation
    df["sunHorizonEl"] = np.round(np.interp(np.mod(df["sunAz"], 360.0), az_grid, horizon_wrapped), 2)
    df["earthHorizonEl"] = np.round(np.interp(np.mod(df["earthAz"], 360.0), az_grid, horizon_wrapped), 2)
    df = df.drop(columns=["azimuth", "altitude"])

    # 4. Calculate all 4 normalized scores [0.0, 1.0] for this site
    s_safety = calculate_safety_score(
        site_row=site_row,
        max_slope_limit_deg=max_slope_limit_deg,
        mean_slope_limit_deg=mean_slope_limit_deg,
        roughness_limit_m=roughness_limit_m,
        slope_weight=slope_weight,
    )
    s_science = calculate_ssi_score(
        site_row=site_row,
        r_max_km=r_max_km,
        decay_rate=decay_rate,
        science_half_sat=science_half_sat,
        return_to_lander=return_to_lander,
    )
    s_sun, s_earth = calculate_astronomical_scores(
        site_row=site_row,
        time_range=(t0, t1),
        time_resolution=(1, resolution),
        horizon_profile=horizons,
    )

    data = df.to_dict(orient="records")

    return {
        "lat": lat,
        "lng": lng,
        "alt": alt,
        "data": data,
        "resolution": resolution,
        "site": dict(zip(CSV_COLUMNS, site_row)),
        "horizon_profile": horizons_list,
        "scores": {
            "s_safety": s_safety,
            "s_science": s_science,
            "s_sun": s_sun,
            "s_earth": s_earth,
        },
    }


# ==============================================================================
# 2. FAST SINGLE-SITE TERRAIN + SAFETY & SCIENCE ENDPOINT (~5-15 ms on click)
# ==============================================================================
@app.get("/api/site/inspect")
async def inspect_site_fast(
    lat: float,
    lng: float,
    max_slope_limit_deg: float = 15.0,
    mean_slope_limit_deg: float = 10.0,
    roughness_limit_m: float = 25.0,
    slope_weight: float = 0.60,
    r_max_km: float = 12.0,
    decay_rate: float = 0.15,
    science_half_sat: float = 5.0,
    return_to_lander: bool = False,
):
    """
    Fast click endpoint: skips 80 km horizon ray-casting and JPL Horizons HTTP calls
    so the UI can immediately render terrain attributes, safety score, and science score.
    """
    site_row = inspect_site_terrain(lat, lng)
    if not site_row:
        raise HTTPException(status_code=404, detail="No valid DEM elevation data at this coordinate.")

    s_safety = calculate_safety_score(
        site_row=site_row,
        max_slope_limit_deg=max_slope_limit_deg,
        mean_slope_limit_deg=mean_slope_limit_deg,
        roughness_limit_m=roughness_limit_m,
        slope_weight=slope_weight,
    )
    s_science = calculate_ssi_score(
        site_row=site_row,
        r_max_km=r_max_km,
        decay_rate=decay_rate,
        science_half_sat=science_half_sat,
        return_to_lander=return_to_lander,
    )

    return {
        "lat": lat,
        "lng": lng,
        "alt": float(site_row[8]) / 1000.0,
        "site": dict(zip(CSV_COLUMNS, site_row)),
        "scores": {
            "s_safety": s_safety,
            "s_science": s_science,
        },
    }


# ==============================================================================
# 3. IN-MEMORY 2D ROVER ROUTE & LANDING ZONE IMAGE ENDPOINT (Zero disk writes)
# ==============================================================================
@app.get("/api/site/route-map")
async def get_route_map(
    lat: float,
    lng: float,
    r_max_km: float = 12.0,
    decay_rate: float = 0.15,
    return_to_lander: bool = False,
    dpi: int = 200,
):
    """
    Generates and streams a borderless 2D lunar surface route map directly from RAM
    without saving any image files locally.
    """
    site_row = inspect_site_terrain(lat=lat, lon=lng)
    if not site_row:
        raise HTTPException(status_code=404, detail="No valid DEM elevation data at this coordinate.")

    result = render_rover_route_image(
        site_row=site_row,
        r_max_km=r_max_km,
        decay_rate=decay_rate,
        return_to_lander=return_to_lander,
        output_path=None,
        dpi=dpi,
    )

    return Response(content=result["image_bytes"], media_type="image/png")


# ==============================================================================
# 4. BATCH ALL-SITES SCORING ENDPOINT (For Globe Markers & Leaderboard Ranking)
# ==============================================================================
@app.get("/api/sites/scores")
async def all_sites_scores(
    time_start: str = "2027-01-01T00:00:00",
    time_end: str = "2027-01-30T00:00:00",
    max_slope_limit_deg: float = 15.0,
    mean_slope_limit_deg: float = 10.0,
    roughness_limit_m: float = 25.0,
    slope_weight: float = 0.60,
    r_max_km: float = 12.0,
    decay_rate: float = 0.15,
    science_half_sat: float = 5.0,
    return_to_lander: bool = False,
    filter_infeasible: bool = True,
):
    """Batch-evaluates all sites across landing_sites.csv and user_inspected_sites.csv."""
    t0, t1, resolution = _format_time_window(time_start, time_end)
    df = scores_calculator(
        max_slope_limit_deg=max_slope_limit_deg,
        mean_slope_limit_deg=mean_slope_limit_deg,
        roughness_limit_m=roughness_limit_m,
        slope_weight=slope_weight,
        r_max_km=r_max_km,
        decay_rate=decay_rate,
        science_half_sat=science_half_sat,
        return_to_lander=return_to_lander,
        time_range=(t0, t1),
        time_resolution=(1, resolution),
        filter_infeasible=filter_infeasible,
    )
    return {
        "count": len(df),
        "resolution": resolution,
        "sites": df.to_dict(orient="records"),
    }