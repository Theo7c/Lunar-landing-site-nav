from datetime import datetime
import os
import sys
import time
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from analysis.horizon import GetHorizonsEphemeris  #[cite: 4]
from analysis.inspect_point import (  #[cite: 3, 4]
    inspect_site_terrain,
    inspect_site_horizon,
    CSV_COLUMNS,
)
from analysis.score_calculator import (  #[cite: 5]
    calculate_safety_score,
    calculate_ssi_score,
    calculate_astronomical_scores,
)
from analysis.scores_calculator import scores_calculator  #[cite: 6]
from analysis.render_route_map import render_rover_route_image

app = FastAPI()  #[cite: 4]

origins = [
    "http://localhost:5173",  #[cite: 4]
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  #[cite: 4]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Server-Timing", "X-Timings-Ms"],
)


def _format_time_window(time_start: str, time_end: str) -> tuple[str, str, str]:
    dt0 = datetime.fromisoformat(time_start)  #[cite: 4]
    t0 = dt0.strftime("%Y-%m-%d %H:%M")  #[cite: 4]

    dt1 = datetime.fromisoformat(time_end)  #[cite: 4]
    t1 = dt1.strftime("%Y-%m-%d %H:%M")  #[cite: 4]

    delta_ms = abs((dt1 - dt0).total_seconds() * 1_000)  #[cite: 4]

    MINUTE_MS = 60 * 1000  #[cite: 4]
    HOUR_MS = 60 * MINUTE_MS  #[cite: 4]
    DAY_MS = 24 * HOUR_MS  #[cite: 4]
    MONTH_MS = 30 * DAY_MS  #[cite: 4]
    YEAR_MS = 12 * MONTH_MS  #[cite: 4]

    if delta_ms < 2 * HOUR_MS:  #[cite: 4]
        resolution = "minute"
    elif delta_ms < 7 * DAY_MS:  #[cite: 4]
        resolution = "hour"
    elif delta_ms < 3 * MONTH_MS:  #[cite: 4]
        resolution = "day"
    elif delta_ms < 3 * YEAR_MS:  #[cite: 4]
        resolution = "month"
    else:
        resolution = "year"  #[cite: 4]

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
    science_half_sat: float = 2.5,
    return_to_lander: bool = False,
):
    t_start = time.perf_counter()
    t0, t1, resolution = _format_time_window(time_start, time_end)  #[cite: 4]

    # Section 1: Terrain lookup
    t_sec = time.perf_counter()
    site_row = inspect_site_terrain(lat, lng)  #[cite: 3, 4]
    if not site_row:
        raise HTTPException(status_code=404, detail="No valid DEM elevation data at this coordinate.")
    terrain_ms = (time.perf_counter() - t_sec) * 1000.0

    alt = float(site_row[8]) / 1000.0  #[cite: 3, 4]

    # Section 2: 360-degree horizon profile
    t_sec = time.perf_counter()
    horizons_list = inspect_site_horizon(lat, lng, site_id=site_row[0])  #[cite: 3, 4]
    horizons = np.asarray(horizons_list, dtype=float)  #[cite: 4]
    horizon_ms = (time.perf_counter() - t_sec) * 1000.0

    # Section 3: JPL Horizons Earth & Sun ephemeris HTTP requests
    t_sec = time.perf_counter()
    df = GetHorizonsEphemeris("Earth", (lng, lat, alt), (t0, t1), (1, resolution))  #[cite: 4]
    earth_eph_ms = (time.perf_counter() - t_sec) * 1000.0

    t_sec = time.perf_counter()
    sun_df = GetHorizonsEphemeris("Sun", (lng, lat, alt), (t0, t1), (1, resolution))  #[cite: 4]
    sun_eph_ms = (time.perf_counter() - t_sec) * 1000.0

    # Section 4: Horizon interpolation onto ephemeris azimuths
    t_sec = time.perf_counter()
    df["sunAz"] = sun_df["azimuth"]  #[cite: 4]
    df["sunEl"] = sun_df["altitude"]  #[cite: 4]
    df["earthAz"] = df["azimuth"]  #[cite: 4]
    df["earthEl"] = df["altitude"]  #[cite: 4]

    az_grid = np.arange(361, dtype=float)  #[cite: 4]
    horizon_wrapped = np.append(horizons, horizons[0])  #[cite: 4]
    df["sunHorizonEl"] = np.round(np.interp(np.mod(df["sunAz"], 360.0), az_grid, horizon_wrapped), 2)  #[cite: 4]
    df["earthHorizonEl"] = np.round(np.interp(np.mod(df["earthAz"], 360.0), az_grid, horizon_wrapped), 2)  #[cite: 4]
    df = df.drop(columns=["azimuth", "altitude"])  #[cite: 4]
    interp_ms = (time.perf_counter() - t_sec) * 1000.0

    # Section 5: Score calculations (safety, science routing, sun/earth visibility)
    t_sec = time.perf_counter()
    s_safety = calculate_safety_score(  #[cite: 5]
        site_row=site_row,
        max_slope_limit_deg=max_slope_limit_deg,
        mean_slope_limit_deg=mean_slope_limit_deg,
        roughness_limit_m=roughness_limit_m,
        slope_weight=slope_weight,
    )
    safety_ms = (time.perf_counter() - t_sec) * 1000.0

    t_sec = time.perf_counter()
    s_science = calculate_ssi_score(  #[cite: 5]
        site_row=site_row,
        r_max_km=r_max_km,
        decay_rate=decay_rate,
        science_half_sat=science_half_sat,
        return_to_lander=return_to_lander,
    )
    science_ms = (time.perf_counter() - t_sec) * 1000.0

    t_sec = time.perf_counter()
    s_sun, s_earth = calculate_astronomical_scores(  #[cite: 5]
        site_row=site_row,
        time_range=(t0, t1),
        time_resolution=(1, resolution),
        horizon_profile=horizons,
    )
    astro_score_ms = (time.perf_counter() - t_sec) * 1000.0

    data = df.to_dict(orient="records")  #[cite: 4]
    total_ms = (time.perf_counter() - t_start) * 1000.0

    timings_ms = {
        "terrain_ms": round(terrain_ms, 2),
        "horizon_ms": round(horizon_ms, 2),
        "earth_ephemeris_ms": round(earth_eph_ms, 2),
        "sun_ephemeris_ms": round(sun_eph_ms, 2),
        "horizon_interp_ms": round(interp_ms, 2),
        "safety_score_ms": round(safety_ms, 2),
        "science_route_ms": round(science_ms, 2),
        "astro_score_ms": round(astro_score_ms, 2),
        "total_ms": round(total_ms, 2),
    }
    print(f"[GET /] ({lat}, {lng}) -> {timings_ms}")

    return {
        "lat": lat,  #[cite: 4]
        "lng": lng,  #[cite: 4]
        "alt": alt,  #[cite: 4]
        "data": data,  #[cite: 4]
        "resolution": resolution,  #[cite: 4]
        "site": dict(zip(CSV_COLUMNS, site_row)),  #[cite: 3]
        "horizon_profile": horizons_list,  #[cite: 3]
        "scores": {  #[cite: 5]
            "s_safety": s_safety,
            "s_science": s_science,
            "s_sun": s_sun,
            "s_earth": s_earth,
        },
        "timings_ms": timings_ms,
    }


# ==============================================================================
# 2. FAST SINGLE-SITE TERRAIN + SAFETY & SCIENCE ENDPOINT
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
    science_half_sat: float = 2.5,
    return_to_lander: bool = False,
):
    t_start = time.perf_counter()

    t_sec = time.perf_counter()
    site_row = inspect_site_terrain(lat, lng)  #[cite: 3]
    if not site_row:
        raise HTTPException(status_code=404, detail="No valid DEM elevation data at this coordinate.")
    terrain_ms = (time.perf_counter() - t_sec) * 1000.0

    t_sec = time.perf_counter()
    s_safety = calculate_safety_score(  #[cite: 5]
        site_row=site_row,
        max_slope_limit_deg=max_slope_limit_deg,
        mean_slope_limit_deg=mean_slope_limit_deg,
        roughness_limit_m=roughness_limit_m,
        slope_weight=slope_weight,
    )
    safety_ms = (time.perf_counter() - t_sec) * 1000.0

    t_sec = time.perf_counter()
    s_science = calculate_ssi_score(  #[cite: 5]
        site_row=site_row,
        r_max_km=r_max_km,
        decay_rate=decay_rate,
        science_half_sat=science_half_sat,
        return_to_lander=return_to_lander,
    )
    science_ms = (time.perf_counter() - t_sec) * 1000.0
    total_ms = (time.perf_counter() - t_start) * 1000.0

    timings_ms = {
        "terrain_ms": round(terrain_ms, 2),
        "safety_score_ms": round(safety_ms, 2),
        "science_route_ms": round(science_ms, 2),
        "total_ms": round(total_ms, 2),
    }
    print(f"[GET /api/site/inspect] ({lat}, {lng}) -> {timings_ms}")

    return {
        "lat": lat,
        "lng": lng,
        "alt": float(site_row[8]) / 1000.0,  #[cite: 3]
        "site": dict(zip(CSV_COLUMNS, site_row)),  #[cite: 3]
        "scores": {
            "s_safety": s_safety,
            "s_science": s_science,
        },
        "timings_ms": timings_ms,
    }


# ==============================================================================
# 3. IN-MEMORY 2D ROVER ROUTE & LANDING ZONE IMAGE ENDPOINT
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
    t_start = time.perf_counter()

    t_sec = time.perf_counter()
    site_row = inspect_site_terrain(lat=lat, lon=lng)  #[cite: 3]
    if not site_row:
        raise HTTPException(status_code=404, detail="No valid DEM elevation data at this coordinate.")
    terrain_ms = (time.perf_counter() - t_sec) * 1000.0

    result = render_rover_route_image(
        site_row=site_row,
        r_max_km=r_max_km,
        decay_rate=decay_rate,
        return_to_lander=return_to_lander,
        output_path=None,
        dpi=dpi,
    )
    total_ms = (time.perf_counter() - t_start) * 1000.0

    render_timings = result.get("timings_ms", {})
    server_timing = ", ".join(
        [f"terrain;dur={terrain_ms:.1f}"]
        + [f"{k};dur={v:.1f}" for k, v in render_timings.items()]
        + [f"endpoint_total;dur={total_ms:.1f}"]
    )
    print(f"[GET /api/site/route-map] ({lat}, {lng}) -> terrain={terrain_ms:.1f}ms, {render_timings}")

    return Response(
        content=result["image_bytes"],
        media_type="image/png",
        headers={
            "Server-Timing": server_timing,
            "X-Timings-Ms": str({**render_timings, "endpoint_total_ms": round(total_ms, 2)}),
        },
    )


# ==============================================================================
# 4. BATCH ALL-SITES SCORING ENDPOINT
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
    science_half_sat: float = 2.5,
    return_to_lander: bool = False,
    filter_infeasible: bool = True,
):
    t_start = time.perf_counter()
    t0, t1, resolution = _format_time_window(time_start, time_end)  #[cite: 4]
    df = scores_calculator(  #[cite: 6]
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
    total_ms = (time.perf_counter() - t_start) * 1000.0
    print(f"[GET /api/sites/scores] Scored {len(df)} sites in {total_ms:.1f} ms")

    return {
        "count": len(df),
        "resolution": resolution,
        "timings_ms": {"total_ms": round(total_ms, 2)},
        "sites": df.to_dict(orient="records"),
    }