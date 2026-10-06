'''
app/main.py
------------
Gemini 기반 고객 리뷰 분석 기능을 HTTP API로 제공하는 FastAPI 애플리케이션

스키마 - 분석기 - API  3단계 중 3번째 작업(조립 담당)
    분리하면 좋은 점 : API 계층(main.py)은 HTTP 요청 검증/ 응답 형식만 담당,
        LLM 호출은 ReviewAnalyzer에 위임한다.
        Gemini를 다른 LLM으로 바꿔도 gemini_client.py만 고치면 된다.
        테스트를 할 때도 분석기를 가짜코드로 바꿔 끼우기 쉽다.
'''
from contextlib import asynccontextmanager # lifespan 함수를 시작/종료 한 쌍으로 만들어주는 도구
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pathlib import Path

from app.gemini_client import ReviewAnalyzer
from app.schemas import ReviewRequest, ReviewResponse

# =========================================
# 1.  .env 로딩
# =========================================
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(env_path, override=False) # override=False --> 실제 환경 변수가 .env보다 우선시한다.

logger = logging.getLogger(__name__) # 이 모듈 전용 로거

# ============================================
# 2. Lifespan : 서버의 시작/종료 생명주기 관리
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버 시작 시 Gemini 클라이언트를 준비하고 종료 시 로그를 남긴다.

    yield 이전 : 서버가 요청을 받기 전에 딱 한 번 실행 (준비)
    yield 이후 : 서버가 종료될 때 딱 한 번 실행 (정리)
    
    """
    try:
        # ReviewAnalyzer() 안에서 API key 존재 여부를 검사한다. (없으면 ValueError)
        # 요청이 올 때 마다 만들지 않고, app.state에 하나만 저장해 모든 요청을 공유한다.
        app.state.analyzer = ReviewAnalyzer()
    except ValueError:
        logger.error('분석기 초기화 실패')
        raise

    # 제어권을 FastAPI에 넘긴다. 이 줄에서 서버가 "요청 받는 상태"로 들어간다.
    yield

    # 서버 종료 시 실행, 닫아야 할 리소스가 아직 없어 로그만 남긴다.
    logger.info('서비스 종료 중...')

# ============================================
# 3. FastAPI 앱 생성
# ============================================
app = FastAPI(
    title='고객 리뷰 분석 API',
    description='Gemini LLM 기반 고객 리뷰 감성 분석 API',
    version='1.0.0',
    lifespan=lifespan
)

# ============================================
# 4. 앤드포인트
# ============================================
@app.get('/health')
def health_check():
    """
    프로세스가 요청에 응답할 수 있는지 확인하는 기본 헬스 체크
    """
    return {'status' : 'healthy'}

@app.post('/analyze', response_model=ReviewResponse)
def analyze_review(request: ReviewRequest):
    """검증된 리뷰를 LLM으로 분석하고 구조화된 결과를 반환한다."""
    try:
        # 검증을 통과한 텍스트만 분석 계층에 넘긴다.
        result = app.state.analyzer.analyze(request.review_text)
        return ReviewResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))