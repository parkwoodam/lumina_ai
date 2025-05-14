from fastapi import APIRouter
from pydantic import BaseModel
from app.services.prediction_service import generate_prediction

router = APIRouter()

class PostRequest(BaseModel):
    post: str

@router.post("/predict")
async def predict(data: PostRequest):
    return generate_prediction(data.post)
