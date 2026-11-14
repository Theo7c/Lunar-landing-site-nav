from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from analysis.horizon import GetHorizonsEphemeris

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

    # TODO: get altitude and set resolution
    alt = -1.9 # km

    # earth data
    df = GetHorizonsEphemeris("Earth", (lng, lat, alt), (t0, t1), (1, resolution))

    # sun data
    sun_df = GetHorizonsEphemeris("Sun", (lng, lat, alt), (t0, t1), (1, resolution))

    df["sunAz"] = sun_df["azimuth"]
    df["sunEl"] = sun_df["altitude"]
    df["earthAz"] = df["azimuth"]
    df["earthEl"] = df["altitude"]

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
