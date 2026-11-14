from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    seconds = abs((dt1 - dt0).total_seconds())

    if seconds < 3600:  # under 1 hour -> minutes
        resolution = seconds / 60, "minutes"
    elif seconds < 86400:  # under 1 day -> hours
        resolution =  seconds / 3600, "hours"
    else:  # 1 day or more -> days
        resolution =  seconds / 86400, "days"

    print(lat, lng, t0, t1, resolution)

    # TODO: make request to api

    # earth data

    # sun data


    # TODO: return data


    return {"message": "Hello World"}