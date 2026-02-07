from fastapi import FastAPI
from pydantic import BaseModel
from datetime import date, timedelta

app = FastAPI(title="Cap Planner Models")


class ForecastRequest(BaseModel):
    start_date: date
    periods: int = 13
    value: float = 0.0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forecast/run")
def forecast(req: ForecastRequest):
    series = []
    cur = req.start_date
    for _ in range(max(req.periods, 0)):
        series.append({"date": cur.isoformat(), "value": req.value})
        cur = cur + timedelta(days=7)
    return {"status": "ok", "series": series}
