'''
app/schemas_db.py
------------------
DB관련 스키마
'''
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.schemas import ReviewResponse

class ReviewRecord(ReviewResponse):
    # from_attributes=True --> SQLAlchemy 객체를 바로 응답으로 변환
    model_config = ConfigDict(from_attributes=True)

    id: int
    review_text: str
    llm_model: str
    created_at: datetime

class StatsResponse(BaseModel):
    total: int
    by_sentiment: dict[str, int]
    by_category: dict[str, int]