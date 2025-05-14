from pydantic import BaseModel
from typing import List, Optional

class RecommendInput(BaseModel):
    post: List[str]
    comment: List[str]

class PostInput(BaseModel):
    post: List[str]

class PredictInput(BaseModel):
    postContent: str
    postImage: Optional[str] = None
    hashtags: Optional[List[str]] = []