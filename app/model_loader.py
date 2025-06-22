# app/model_loader.py
import torch
from functools import lru_cache

@lru_cache(maxsize=1)
def get_model():
    model = torch.jit.load("recsys.pt", map_location="cpu")
    model.eval()
    return model

