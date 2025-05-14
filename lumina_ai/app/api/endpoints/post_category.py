from fastapi import APIRouter
from app.services.post_category_service import handle_post_categorization
from app.models.request import PredictInput

router = APIRouter()

@router.post("/post-categorize")
def categorize_post(req: PredictInput):
    return handle_post_categorization(
        post_text=req.postContent,
        image_url=req.postImage,
        hashtags=req.hashtags
    )
