# app/worker.py
import asyncio, json
from redis.asyncio import Redis
from .model_loader import get_model

async def main():
    r = Redis(host="redis", port=6379, decode_responses=True)
    model = get_model()

    while True:
        job = await r.brpop("jobs", timeout=0)  # blocking pop
        _, payload = job
        data = json.loads(payload)
        pred = run_inference(model, data)
        await r.set(f"result:{data['id']}", json.dumps(pred), ex=300)

def run_inference(model, data):
    # Your real scoring code here
    return {"score": float(model(torch.tensor(data["item_ids"])).item())}

if __name__ == "__main__":
    asyncio.run(main())

