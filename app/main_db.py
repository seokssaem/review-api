'''
app/main_db.py
--------------
리뷰 DB관련 API
'''
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# [중요] app.database를 import하기 "전에" .env를 읽어야 DATABASE_URL이 반영된다.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)

from fastapi import Depends, FastAPI, HTTPException, Query  # noqa: E402
from sqlalchemy import func, select, text  # noqa: E402
from sqlalchemy.exc import SQLAlchemyError  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.database import Base, engine, get_db  # noqa: E402
# from app.gemini_client import ReviewAnalyzer  # noqa: E402
from app.openai_client import ReviewAnalyzer # noqa: E402
from app.models import Review  # noqa: E402
from app.schemas import ReviewRequest, ReviewResponse  # noqa: E402
from app.schemas_db import ReviewRecord, StatsResponse  # noqa: E402

# noqa: E402 --> 린터(코드 검사 도구)의 E402경고를 무시해라!

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # create_all() --> 없는 테이블만 만든다. 이미 있는 테이블의 컬럼 변경은 반영하지 않는다.
    Base.metadata.create_all(bind=engine)
    logger.info('DB 테이블 준비 완료')

    app.state.analyzer = ReviewAnalyzer() 
    yield
    logger.info('서비스 종료 중...')

app = FastAPI(
    title='고객 리뷰 분석 API (PostgreSQL 저장 버전)',
    description='OpenAI API로 분석한 리뷰를 PostgreSQL에 저장하고 조회한다.',
    version='2.0.0',
    lifespan=lifespan,
)

@app.get('/health')
def health_check(db: Session=Depends(get_db)):
    try:
        db.execute(text('SELECT 1'))
        return {'status': 'healthy', 'db': 'ok'}
    except SQLAlchemyError:
        return {'status': 'degraded', 'db': 'error'}

@app.post('/analyze', response_model=ReviewRecord)    
def analyze_and_save(request: ReviewRequest, db: Session=Depends(get_db)):
    # 1) Gemini 분석 + 응답 검증
    try:
        result = app.state.analyzer.analyze(request.review_text)
        analysis = ReviewResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 2) DB 저장. 저장 실패는 분석 실패와 구분해서 알려준다.
    row = Review(
        review_text=request.review_text,
        llm_model=app.state.analyzer.model,
        **analysis.model_dump(),
    )
    try:
        db.add(row)
        db.commit()
        db.refresh(row)
    except SQLAlchemyError as e:
        db.rollback()
        logger.error('DB 저장 실패: %s', e)
        raise HTTPException(status_code=503, detail='분석은 성공했지만 DB 저장에 실패했습니다.')
    return row

@app.get('/reviews', response_model=list[ReviewRecord])
def list_reviews(
    sentiment: str | None = None,
    category: str | None = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, get=0),
    db: Session = Depends(get_db),
):
    stmt = select(Review).order_by(Review.id.desc()).limit(limit).offset(offset)
    if sentiment:
        stmt = stmt.where(Review.sentiment == sentiment)
    if category:
        stmt = stmt.where(Review.category == category)
    return db.scalars(stmt).all()

@app.get('/reviews/stats', response_model=StatsResponse)
def review_stats(db: Session=Depends(get_db)):
    total = db.scalar(select(func.count()).select_from(Review)) or 0
    by_sentiment = dict(db.execute(select(Review.sentiment, func.count()).group_by(Review.sentiment)).all())
    by_category = dict(db.execute(select(Review.category, func.count()).group_by(Review.category)).all())
    return StatsResponse(total=total, by_sentiment=by_sentiment, by_category=by_category)

@app.get('/reviews/{review_id}', response_model=ReviewRecord)
def get_review(review_id: int, db: Session=Depends(get_db)):
    row = db.get(Review, review_id)
    if row is None:
        raise HTTPException(status_code=404, detail='해당 리뷰가 없습니다.')
    return row