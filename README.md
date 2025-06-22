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
