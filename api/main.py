from datetime import datetime
import os
import sys
import time
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from analysis.GeminiAPICaller import GeneratePrefix, GenerateDefaultQuery, GetAIResponse, FormatResponse

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
)
from analysis.scores_calculator import scores_calculator, DEFAULT_MAX_STOPS
from analysis.render_route_map import render_rover_route_image

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Server-Timing", "X-Timings-Ms"],
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


def _build_full_site_payload(
    site_row: list,
    t0: str,
    t1: str,
    resolution: str,
    max_slope_limit_deg: float = 15.0,
    mean_slope_limit_deg: float = 10.0,
    roughness_limit_m: float = 25.0,
    slope_weight: float = 0.60,
    r_max_km: float = 12.0,
    decay_rate: float = 0.15,
    science_half_sat: float = 5.0,
    return_to_lander: bool = False,
    max_stops: int = DEFAULT_MAX_STOPS,
    precomputed_safety_science: dict | None = None,
) -> tuple[dict, dict]:
    """
    Builds the complete single-site response dictionary (site metadata, 360-deg horizon
    profile, Sun/Earth ephemeris + horizon elevation series, and exact scores).
    """
    lat = float(site_row[2])
    lng = float(site_row[3])
    alt = float(site_row[8]) / 1000.0

    # 360-degree horizon profile
    t_sec = time.perf_counter()
    horizons_list = inspect_site_horizon(lat, lng, site_id=site_row[0])
    horizons = np.asarray(horizons_list, dtype=float)
    horizon_ms = (time.perf_counter() - t_sec) * 1000.0

    # JPL Horizons Earth & Sun ephemeris HTTP requests
    t_sec = time.perf_counter()
    df = GetHorizonsEphemeris("Earth", (lng, lat, alt), (t0, t1), (1, resolution))
    earth_eph_ms = (time.perf_counter() - t_sec) * 1000.0

    t_sec = time.perf_counter()
    sun_df = GetHorizonsEphemeris("Sun", (lng, lat, alt), (t0, t1), (1, resolution))
    sun_eph_ms = (time.perf_counter() - t_sec) * 1000.0

    # Horizon interpolation onto ephemeris azimuths
    t_sec = time.perf_counter()
    df["sunAz"] = sun_df["azimuth"]
    df["sunEl"] = sun_df["altitude"]
    df["earthAz"] = df["azimuth"]
    df["earthEl"] = df["altitude"]

    az_grid = np.arange(361, dtype=float)
    horizon_wrapped = np.append(horizons, horizons[0])
    df["sunHorizonEl"] = np.round(np.interp(np.mod(df["sunAz"], 360.0), az_grid, horizon_wrapped), 2)
    df["earthHorizonEl"] = np.round(np.interp(np.mod(df["earthAz"], 360.0), az_grid, horizon_wrapped), 2)
    df = df.drop(columns=["azimuth", "altitude"])
    interp_ms = (time.perf_counter() - t_sec) * 1000.0

    # Exact Sun & Earth visibility fractions from the interpolated ephemeris + horizon profile
    t_sec = time.perf_counter()
    s_sun = round(float(np.mean(df["sunEl"].to_numpy(dtype=float) > df["sunHorizonEl"].to_numpy(dtype=float))), 4)
    s_earth = round(float(np.mean(df["earthEl"].to_numpy(dtype=float) > df["earthHorizonEl"].to_numpy(dtype=float))), 4)
    astro_score_ms = (time.perf_counter() - t_sec) * 1000.0

    if precomputed_safety_science is not None:
        s_safety = float(precomputed_safety_science["s_safety"])
        s_science = float(precomputed_safety_science["s_science"])
        safety_ms = 0.0
        science_ms = 0.0
    else:
        t_sec = time.perf_counter()
        s_safety = calculate_safety_score(
            site_row=site_row,
            max_slope_limit_deg=max_slope_limit_deg,
            mean_slope_limit_deg=mean_slope_limit_deg,
            roughness_limit_m=roughness_limit_m,
            slope_weight=slope_weight,
        )
        safety_ms = (time.perf_counter() - t_sec) * 1000.0

        t_sec = time.perf_counter()
        s_science = calculate_ssi_score(
            site_row=site_row,
            r_max_km=r_max_km,
            decay_rate=decay_rate,
            science_half_sat=science_half_sat,
            return_to_lander=return_to_lander,
            max_stops=max_stops,
        )
        science_ms = (time.perf_counter() - t_sec) * 1000.0

    scores = {
        "s_safety": s_safety,
        "s_science": s_science,
        "s_sun": s_sun,
        "s_earth": s_earth,
    }

    payload = {
        "lat": lat,
        "lng": lng,
        "alt": alt,
        "data": df.to_dict(orient="records"),
        "resolution": resolution,
        "site": dict(zip(CSV_COLUMNS, site_row)),
        "horizon_profile": horizons_list,
        "scores": scores,
    }
    step_timings = {
        "horizon_ms": round(horizon_ms, 2),
        "earth_ephemeris_ms": round(earth_eph_ms, 2),
        "sun_ephemeris_ms": round(sun_eph_ms, 2),
        "horizon_interp_ms": round(interp_ms, 2),
        "safety_score_ms": round(safety_ms, 2),
        "science_route_ms": round(science_ms, 2),
        "astro_score_ms": round(astro_score_ms, 2),
    }
    return payload, step_timings


# ==============================================================================
# 1. MAIN GRAPH & FULL SITE DATA ENDPOINT
# ==============================================================================
@app.get("/")
async def root(
    lat: float,
    lng: float,
    time_start: str,
    time_end: str,
    area_radius_m: float | None = None,
    max_slope_limit_deg: float = 15.0,
    mean_slope_limit_deg: float = 10.0,
    roughness_limit_m: float = 25.0,
    slope_weight: float = 0.60,
    r_max_km: float = 12.0,
    decay_rate: float = 0.15,
    science_half_sat: float = 5.0,
    return_to_lander: bool = False,
    max_stops: int = DEFAULT_MAX_STOPS,
):
    t_start = time.perf_counter()
    t0, t1, resolution = _format_time_window(time_start, time_end)

    t_sec = time.perf_counter()
    site_row = inspect_site_terrain(lat, lng, area_radius_m=area_radius_m)
    if not site_row:
        raise HTTPException(status_code=404, detail="No valid DEM elevation data at this coordinate.")
    terrain_ms = (time.perf_counter() - t_sec) * 1000.0

    payload, step_timings = _build_full_site_payload(
        site_row=site_row,
        t0=t0,
        t1=t1,
        resolution=resolution,
        max_slope_limit_deg=max_slope_limit_deg,
        mean_slope_limit_deg=mean_slope_limit_deg,
        roughness_limit_m=roughness_limit_m,
        slope_weight=slope_weight,
        r_max_km=r_max_km,
        decay_rate=decay_rate,
        science_half_sat=science_half_sat,
        return_to_lander=return_to_lander,
        max_stops=max_stops,
    )
    total_ms = (time.perf_counter() - t_start) * 1000.0

    timings_ms = {
        "terrain_ms": round(terrain_ms, 2),
        **step_timings,
        "total_ms": round(total_ms, 2),
    }
    print(f"[GET /] ({lat}, {lng}, lz={site_row[7]}m, max_stops={max_stops}) -> {timings_ms}")

    return {
        **payload,
        "timings_ms": timings_ms,
    }


# ==============================================================================
# 2. FAST SINGLE-SITE TERRAIN + SAFETY & SCIENCE ENDPOINT
# ==============================================================================
@app.get("/api/site/inspect")
async def inspect_site_fast(
    lat: float,
    lng: float,
    area_radius_m: float | None = None,
    max_slope_limit_deg: float = 15.0,
    mean_slope_limit_deg: float = 10.0,
    roughness_limit_m: float = 25.0,
    slope_weight: float = 0.60,
    r_max_km: float = 12.0,
    decay_rate: float = 0.15,
    science_half_sat: float = 5.0,
    return_to_lander: bool = False,
    max_stops: int = DEFAULT_MAX_STOPS,
):
    t_start = time.perf_counter()

    t_sec = time.perf_counter()
    site_row = inspect_site_terrain(lat, lng, area_radius_m=area_radius_m)
    if not site_row:
        raise HTTPException(status_code=404, detail="No valid DEM elevation data at this coordinate.")
    terrain_ms = (time.perf_counter() - t_sec) * 1000.0

    t_sec = time.perf_counter()
    s_safety = calculate_safety_score(
        site_row=site_row,
        max_slope_limit_deg=max_slope_limit_deg,
        mean_slope_limit_deg=mean_slope_limit_deg,
        roughness_limit_m=roughness_limit_m,
        slope_weight=slope_weight,
    )
    safety_ms = (time.perf_counter() - t_sec) * 1000.0

    t_sec = time.perf_counter()
    s_science = calculate_ssi_score(
        site_row=site_row,
        r_max_km=r_max_km,
        decay_rate=decay_rate,
        science_half_sat=science_half_sat,
        return_to_lander=return_to_lander,
        max_stops=max_stops,
    )
    science_ms = (time.perf_counter() - t_sec) * 1000.0
    total_ms = (time.perf_counter() - t_start) * 1000.0

    timings_ms = {
        "terrain_ms": round(terrain_ms, 2),
        "safety_score_ms": round(safety_ms, 2),
        "science_route_ms": round(science_ms, 2),
        "total_ms": round(total_ms, 2),
    }
    print(f"[GET /api/site/inspect] ({lat}, {lng}, lz={site_row[7]}m, max_stops={max_stops}) -> {timings_ms}")

    return {
        "lat": lat,
        "lng": lng,
        "alt": float(site_row[8]) / 1000.0,
        "site": dict(zip(CSV_COLUMNS, site_row)),
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
    area_radius_m: float | None = None,
    r_max_km: float = 12.0,
    decay_rate: float = 0.15,
    return_to_lander: bool = False,
    max_stops: int = DEFAULT_MAX_STOPS,
    size_px: int = 150,
    dpi: int = 100,
):
    t_start = time.perf_counter()

    t_sec = time.perf_counter()
    site_row = inspect_site_terrain(lat=lat, lon=lng, area_radius_m=area_radius_m)
    if not site_row:
        raise HTTPException(status_code=404, detail="No valid DEM elevation data at this coordinate.")
    terrain_ms = (time.perf_counter() - t_sec) * 1000.0

    result = render_rover_route_image(
        site_row=site_row,
        area_radius_m=area_radius_m,
        r_max_km=r_max_km,
        decay_rate=decay_rate,
        return_to_lander=return_to_lander,
        max_stops=max_stops,
        output_path=None,
        size_px=size_px,
        dpi=dpi,
    )
    total_ms = (time.perf_counter() - t_start) * 1000.0

    render_timings = result.get("timings_ms", {})
    server_timing = ", ".join(
        [f"terrain;dur={terrain_ms:.1f}"]
        + [f"{k};dur={v:.1f}" for k, v in render_timings.items()]
        + [f"endpoint_total;dur={total_ms:.1f}"]
    )
    print(f"[GET /api/site/route-map] ({lat}, {lng}, {size_px}x{size_px}px) -> terrain={terrain_ms:.1f}ms, {render_timings}")

    return Response(
        content=result["image_bytes"],
        media_type="image/png",
        headers={
            "Server-Timing": server_timing,
            "X-Timings-Ms": str({**render_timings, "endpoint_total_ms": round(total_ms, 2)}),
        },
    )


# ==============================================================================
# 4. FAST BATCH WEIGHTED MCDM SCORING (ESTIMATION ONLY -> TOP-5 LAT/LON)
# ==============================================================================
@app.get("/api/sites/scores")
async def all_sites_scores(
    w_safety: float = 0.25,
    w_science: float = 0.25,
    w_sun: float = 0.25,
    w_earth: float = 0.25,
    top_n: int = 5,
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
    max_stops: int = DEFAULT_MAX_STOPS,
    filter_infeasible: bool = True,
):
    """
    Batch-evaluates all candidate sites using fast local estimation (no external
    JPL Horizons API calls), ranks them by weighted MCDM score, and returns the
    top `top_n` (default 5) candidates' coordinates and estimated scores.
    """
    if any(w < 0.0 for w in (w_safety, w_science, w_sun, w_earth)):
        raise HTTPException(status_code=400, detail="Weights must be non-negative.")

    weight_sum = w_safety + w_science + w_sun + w_earth
    if weight_sum <= 0.0:
        raise HTTPException(status_code=400, detail="At least one weight must be greater than zero.")

    nw_safety = w_safety / weight_sum
    nw_science = w_science / weight_sum
    nw_sun = w_sun / weight_sum
    nw_earth = w_earth / weight_sum

    t_start = time.perf_counter()
    t0, t1, resolution = _format_time_window(time_start, time_end)

    t_batch = time.perf_counter()
    df = scores_calculator(
        max_slope_limit_deg=max_slope_limit_deg,
        mean_slope_limit_deg=mean_slope_limit_deg,
        roughness_limit_m=roughness_limit_m,
        slope_weight=slope_weight,
        r_max_km=r_max_km,
        decay_rate=decay_rate,
        science_half_sat=science_half_sat,
        return_to_lander=return_to_lander,
        max_stops=max_stops,
        time_range=(t0, t1),
        time_resolution=(1, resolution),
        filter_infeasible=filter_infeasible,
    )
    batch_score_ms = (time.perf_counter() - t_batch) * 1000.0

    if df.empty:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "evaluated_count": 0,
            "normalized_weights": {
                "w_safety": round(nw_safety, 4),
                "w_science": round(nw_science, 4),
                "w_sun": round(nw_sun, 4),
                "w_earth": round(nw_earth, 4),
            },
            "timings_ms": {
                "batch_score_ms": round(batch_score_ms, 2),
                "total_ms": round(total_ms, 2),
            },
            "top_sites": [],
        }

    df["total_score"] = np.round(
        nw_safety * df["s_safety"]
        + nw_science * df["s_science"]
        + nw_sun * df["s_sun"]
        + nw_earth * df["s_earth"],
        4,
    )

    ranked_df = (
        df.sort_values(
            by=["total_score", "s_safety", "s_science"],
            ascending=[False, False, False],
        )
        .head(max(1, int(top_n)))
        .reset_index(drop=True)
    )

    top_sites = [
        {
            "rank": idx + 1,
            "site_id": str(row["site_id"]),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "scores": {
                "s_safety": float(row["s_safety"]),
                "s_science": float(row["s_science"]),
                "s_sun": float(row["s_sun"]),
                "s_earth": float(row["s_earth"]),
                "total_score": float(row["total_score"]),
            },
        }
        for idx, row in ranked_df.iterrows()
    ]

    total_ms = (time.perf_counter() - t_start) * 1000.0
    timings_ms = {
        "batch_score_ms": round(batch_score_ms, 2),
        "total_ms": round(total_ms, 2),
    }
    print(f"[GET /api/sites/scores] Evaluated {len(df)} sites -> top {len(top_sites)} in {timings_ms}")

    return {
        "evaluated_count": len(df),
        "normalized_weights": {
            "w_safety": round(nw_safety, 4),
            "w_science": round(nw_science, 4),
            "w_sun": round(nw_sun, 4),
            "w_earth": round(nw_earth, 4),
        },
        "timings_ms": timings_ms,
        "top_sites": top_sites,
    }

@app.get("/api/ai")
def route(query):
    prefix = GeneratePrefix()
    response = GetAIResponse(GEMINI_API_KEY, query, prefix)
    weights, coords, justifications = FormatResponse(response)
    return {
        "weights": weights,
        "coords": coords,
        "justifications": justifications,
    }