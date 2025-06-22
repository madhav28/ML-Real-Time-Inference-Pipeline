# ML-Real-Time-Inference-Pipeline
_FastAPI + Redis + TorchScript_

Serve recommendations in **single-digit milliseconds** with an ultra-light stack:

* **FastAPI** – async HTTP gateway  
* **Redis** – in-memory job queue + result cache  
* **TorchScript** – model (`recsys.pt`) loaded once per worker  
* **Docker Compose** – 1-command local deployment  

## Project Layout
```text
real-time-pipeline/
├── app/
│   ├── main.py            # FastAPI application
│   ├── models.py          # Pydantic schemas
│   ├── worker.py          # Background inference loop
│   ├── model_loader.py    # Lazy-loads the ML model
│   ├── train_recsys.py    # Script to (re)train and export recsys.pt
│   └── recsys.pt          # TorchScript recommendation model
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Spin everything up:
```bash
docker compose up --build      # API at http://localhost:8000
```

## Endpoints
| Method | Route                | Purpose                       |
|--------|----------------------|-------------------------------|
| POST   | `/predict`           | Queue a prediction job        |
| GET    | `/result/<job_id>`   | Poll until scores are ready   |

## Example
```bash
# Submit job
JOB=$(curl -s -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d '{"user_id": 42, "item_ids": [4,7,19]}' | jq -r .job_id)

# Poll for result
curl http://localhost:8000/result/$JOB
# → {"scores":[0.83,0.41,0.57]}
```