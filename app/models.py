'''
app/models.py
-------------
리뷰를 담을 reviews 테이블 생성
'''
from datetime import datetime

from sqlalchemy import DateTime, Float, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

class Review(Base):
    __tablename__ = 'reviews'

    id: Mapped[int] = mapped_column(primary_key=True)
    review_text: Mapped[str] = mapped_column(Text)

    # index=True --> 빠르게 찾이 위한 색인 (감성/카테고리)
    sentiment: Mapped[str] = mapped_column(String(50), index=True)
    category: Mapped[str] = mapped_column(String(50), index=True)
    summary: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)

    # 어떤 모델로 분석했는지, 모델이 바뀌면 결과 품질도 달라지므로 기록
    llm_model: Mapped[str] = mapped_column(String(100))

    # server_default=func.now() --> 저장 시각을 DB가 채운다.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())