import io
import zipfile
from pathlib import Path

import pandas as pd
import requests
import torch
from torch import nn

ML_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = Path("ml-100k")
MODEL_OUT = Path("recsys.pt")


def download_movielens() -> Path:
    if DATA_DIR.exists():
        return DATA_DIR / "u.data"
    print("Downloading MovieLens-100K ...")
    resp = requests.get(ML_URL, timeout=30)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    zf.extractall()
    return DATA_DIR / "u.data"


class MF(nn.Module):
    """Sample matrix factorization model."""

    def __init__(self, n_users: int, n_items: int, k: int = 32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, k)
        self.item_emb = nn.Embedding(n_items, k)

    def forward(self, user_id: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_id)           
        i = self.item_emb(item_ids)          
        scores = (u * i).sum(dim=-1)          
        return scores.sigmoid()               


def train():
    data_path = download_movielens()
    df = pd.read_csv(
        data_path,
        sep="\t",
        names=["u", "i", "r", "_"],
        engine="python"
    )
    df["u"] -= 1 
    df["i"] -= 1
    n_users = int(df.u.max() + 1)
    n_items = int(df.i.max() + 1)

    model = MF(n_users, n_items)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    df_shuf = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    for epoch in range(5):
        total = 0.0
        for _, row in df_shuf.iterrows():
            u = torch.tensor([int(row.u)])
            i = torch.tensor([int(row.i)])
            r = torch.tensor([float(row.r) / 5.0])
            opt.zero_grad()
            pred = model(u, i)
            loss = loss_fn(pred, r)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {epoch + 1}  avg_loss={total / len(df):.4f}")

    model.eval()
    scripted = torch.jit.trace(
        model,
        (
            torch.tensor([0]),
            torch.tensor([0, 1, 2])
        )
    )
    torch.jit.save(scripted, MODEL_OUT)
    size_kb = MODEL_OUT.stat().st_size // 1024
    print(f"Saved {MODEL_OUT}  ({size_kb} KB)")


if __name__ == "__main__":
    train()