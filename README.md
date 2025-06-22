# ML-Real-Time-Inference-Pipeline

## Project Layout
real-time-pipeline/
├── app/
│   ├── main.py            # FastAPI application
│   ├── models.py          # Pydantic schemas
│   ├── worker.py          # Background inference loop
│   └── model_loader.py    # Lazy-loads the ML model
├── Dockerfile
├── docker-compose.yml
└── requirements.txt

