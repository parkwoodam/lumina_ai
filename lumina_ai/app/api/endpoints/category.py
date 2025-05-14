from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from app.services.category_service import recommend_category_logic

router = APIRouter()

class FullInput(BaseModel):
    post: List[str]
    comment: List[str]

@router.post("/recommend")
def recommend_category(data: FullInput):
    return recommend_category_logic(data)
