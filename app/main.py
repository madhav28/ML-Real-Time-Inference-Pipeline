# app/main.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
from redis.asyncio import Redis
from uuid import uuid4
import json
from .models import PredictRequest

app = FastAPI()
r = Redis(host="redis", port=6379, decode_responses=True)

QUEUE = "jobs"

@app.post("/predict")
async def predict(req: PredictRequest):
    job_id = str(uuid4())
    await r.lpush(QUEUE, json.dumps({"id": job_id, **req.dict()}))
    return {"job_id": job_id, "status": "queued"}

@app.get("/result/{job_id}")
async def result(job_id: str):
    res = await r.get(f"result:{job_id}")
    if not res:
        raise HTTPException(status_code=202, detail="still processing")
    return json.loads(res)

