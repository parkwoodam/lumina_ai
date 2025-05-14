from fastapi import APIRouter
from .endpoints import category, post_category, prediction

api_router = APIRouter()
api_router.include_router(category.router)
api_router.include_router(post_category.router)
api_router.include_router(prediction.router)