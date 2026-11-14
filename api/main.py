from fastapi import FastAPI

# initialise fast api instance
app = FastAPI()

@app.get("/")
async def root(lat: float, lon: float, time_start: str, time_end: str):

    # make request longitude then latitude
    return {"message": "Hello World"}