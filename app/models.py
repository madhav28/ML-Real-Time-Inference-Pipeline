# app/models.py
from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    user_id: int = Field(..., ge=1)
    item_ids: List[int] = Field(..., min_items=1, max_items=100)
