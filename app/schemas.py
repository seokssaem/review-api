'''
app/schemas.py
---------------
리뷰 분석 API의 Contract를 정의하는 파일
    1. ReviewRequest : 클라이언트가 보내는 요청 본문 (입력)
    2. ReviewResponse : 서버가 돌려주는 응답 본문 (출력)

스키마 - 분석기 - API  3단계 중 1단계 작업
    mlops-loan의 schemas.py와 역할이 같다.
'''
from pydantic import BaseModel, Field

# ==================================================
# 1. 요청 스키마 : 사용자가 서버에 보내는 데이터
# ==================================================
class ReviewRequest(BaseModel):
    """LLM에게 전달할 고객 리뷰 요청 본문"""
    review_text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description='분석할 고객 리뷰 텍스트',
        examples=['이 제품 정말 좋아요! 배송도 빠르고 품질이 우수합니다.']
    )

# ==================================================
# 2. 응답 스키마 : 서버가 사용자에게 돌려주는 데이터
# ==================================================
class ReviewResponse(BaseModel):
    """
    Gemini 분석 결과를 외부 클라이언트에게 제공하는 표준 응답 형식

    LLM은 확률적으로 글을 만드는 모델이라 100% 지킨다는 보장이 없다.
    그래서 신뢰하되 검증한다(trust but verify)가 필요하다.
    """
    sentiment: str = Field(
        ...,
        description='감정 분석 결과',
        examples=['긍정', '부정', '중립']
    )

    category: str = Field(
        ...,
        description='리뷰 카테고리',
        examples=['품질', '배송', '가격', '서비스', '기타']
    )

    # summary: 원문이 길어도 후속 처리를 위해 빠르게 읽을 수 있는 요약
    summary: str = Field(
        ...,
        description='리뷰 요약 (1~2문장)'
    )

    # confidence: 모델이 스스로 매긴 신뢰도
    # ge --> greater or equal (이상)
    # le --> less or equal (이하)
    confidence: float = Field(
        ...,
        ge = 0.0,
        le = 1.0,
        description='분석 신뢰도 (0.0~1.0)'
    )