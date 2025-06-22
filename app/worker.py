# app/worker.py
import asyncio, json
from redis.asyncio import Redis
from .model_loader import get_model
import torch

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
    user_id_tensor = torch.tensor([data["user_id"]] * len(data["item_ids"]))
    item_ids_tensor = torch.tensor(data["item_ids"])
    scores = model(user_id_tensor, item_ids_tensor)
    return {"scores": scores.tolist()}

if __name__ == "__main__":
    asyncio.run(main())

