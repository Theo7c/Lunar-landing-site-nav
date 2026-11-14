from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from analysis.horizon import GetHorizonsEphemeris
from analysis.inspect_point import inspect_site_terrain, inspect_site_horizon
import numpy as np
# initialise fast api instance
app = FastAPI()

# CORS
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

@app.get("/")
async def root(lat: float, lng: float, time_start: str, time_end: str):
    # format times
    dt0 = datetime.fromisoformat(time_start)
    t0 = dt0.strftime("%Y-%m-%d %H:%M")

    dt1 = datetime.fromisoformat(time_end)
    t1 = dt1.strftime("%Y-%m-%d %H:%M")

    # calculate resolution
    delta_ms = abs((dt1 - dt0).total_seconds() * 1_000)

    MINUTE_MS = 60 * 1000
    HOUR_MS = 60 * MINUTE_MS
    DAY_MS = 24 * HOUR_MS
    MONTH_MS = 30 * DAY_MS
    YEAR_MS = 12 * MONTH_MS

    def get_resolution(delta_ms):
        if delta_ms < 2 * HOUR_MS:
            return "minute"
        if delta_ms < 7 * DAY_MS:
            return "hour"
        if delta_ms < 3 * MONTH_MS:
            return "day"
        if delta_ms < 3 * YEAR_MS:
            return "month"
        return "year"

    resolution = get_resolution(delta_ms)

    alt = inspect_site_terrain(lat, lng)[0][8]/1000

    # earth data
    df = GetHorizonsEphemeris("Earth", (lng, lat, alt), (t0, t1), (1, resolution))

    # sun data
    sun_df = GetHorizonsEphemeris("Sun", (lng, lat, alt), (t0, t1), (1, resolution))

    df["sunAz"] = sun_df["azimuth"]
    df["sunEl"] = sun_df["altitude"]
    df["earthAz"] = df["azimuth"]
    df["earthEl"] = df["altitude"]
    # horizon data
    # 1. Get terrain row (note: inspect_site_terrain returns the 12-element list directly, so index [8] is elevation_m)
    site_row = inspect_site_terrain(lat, lng)
    alt = site_row[8] / 1000.0  #

    # 2. Fetch 360-degree horizon profile (0 to 359 deg) and wrap 360 -> 0 for circular interpolation
    horizons = np.asarray(inspect_site_horizon(lat, lng, site_id=site_row[0]), dtype=float)  #
    az_grid = np.arange(361, dtype=float)
    horizon_wrapped = np.append(horizons, horizons[0])

    # 3. Add horizon elevation angle at each timestamp's Sun and Earth azimuth
    df["sunHorizonEl"] = np.round(np.interp(np.mod(df["sunAz"] % 360.0), az_grid, horizon_wrapped), 2)
    df["earthHorizonEl"] = np.round(np.interp(np.mod(df["earthAz"] % 360.0), az_grid, horizon_wrapped), 2)
    df = df.drop(columns=["azimuth", "altitude"])

    # convert to json
    data = df.to_dict(orient="records")

    return {
        "lat": lat,
        "lng": lng,
        "alt": alt,
        "data": data,
        "resolution": resolution,
    }
