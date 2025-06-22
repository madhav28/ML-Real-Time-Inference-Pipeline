from functools import lru_cache
from pathlib import Path
import torch

MODEL_PATH = Path(__file__).parent / "recsys.pt"   

@lru_cache(maxsize=1)
def get_model() -> "torch.jit.ScriptModule":
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Copy your recsys.pt here."
        )
    return torch.jit.load(str(MODEL_PATH), map_location="cpu").eval()
