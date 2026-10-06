'''
app/gemini_client.py
----------------------
제미나이 호출과 프롬프트를 한 곳에 캡슐화한 클라이언트 모듈
    1. 환경 변수에서 API키를 읽어 Gemini 클라이언트를 만든다.
    2. 리뷰 한 건을 프롬프트로 만들어 Gemini에 보낸다.
    3. 응답(JSON 문자열)을 파이썬 dict로 바꿔 main.py 에 돌려준다.

스키마 - 분석기 - API  3단계 중 2단계 작업 (외부 호출 담당)
분리하면 좋은 점 : OpenAI / Claude로 바꾸어도 이 파일만 수정하면 된다.
'''
import json
import logging # print 대신 로그 레벨(INFO/ERROR)이 있는 기록 도구
import os
import time

from google import genai
from google.genai import types
from google.genai import errors

logger = logging.getLogger(__name__)

# 오류 코드 --> 잠시 뒤에 다시 하면 될 수도 있는 오류만 재시도!
# 429 : 요청이 너무 많음(분당 한도), 하루 쿼터를 다 사용한 경우에는 재시도 해도 되지 않는다.
# 500 : 구글 서버 내부 오류
# 503 : 과부하 (UNAVALIABLE). 신규 모델에 수요가 몰릴 때.
# 504 : 응답 시간 초과 
# 400(요청이 잘못됨), 403(권한이 없음), 404(모델 없음) 등 바로 실패.
RETRYABLE_CODES = {429, 500, 503, 504}

# =====================================================================
# 1. 응답 스키마 : 이 JSON 구조로 답해줘.라고 제미나이에게 요구하는 틀
# =====================================================================
REVIEW_SCHEMA = {
    "type" : "object",
    "properties" : {
        "sentiment" : {"type" : "string"},
        "category" : {"type" : "string"},
        "summary" : {"type" : "string"},
        "confidence" : {"type" : "number"}                       
    },
    "required" : ["sentiment", "category", "summary", "confidence"]
}

# =====================================================================
# 2. 분석기 클래스
# =====================================================================
class ReviewAnalyzer:
    """환경 변수로 Gemini 클라이언트를 초기화하고 리뷰 한 건을 분석한다."""

    def __init__(self):
        api_key = os.environ.get('GEMINI_API_KEY', '')

        if not api_key:
            raise ValueError('API key가 설정되지 않았습니다.')

        # Gemini 클라이언트 생성 - Gemini 서버와의 연결 도구
        self.client = genai.Client(api_key=api_key)

        self.model = os.environ.get('GEMINI_MODEL', 'gemini-3.8-flash')

        # 일시적 오류 대비 설정
        fallback = os.environ.get('GEMINI_FALLBACK_MODEL', '').strip()
        self.fallback_model = fallback if fallback and fallback != self.model else None
        self.max_retries = max(1, int(os.environ.get('GEMINI_MAX_RETRIES', '3')))
        self.retry_delay = float(os.environ.get('GEMINI_RETRY_DELAY', '1.0'))

        logger.info('분석기 초기화 완료!')

    def analyze(self, review_text: str) -> dict:
        """
        리뷰 텍스트를 감성, 카테고리, 요약, 신뢰도로 구조화한다.

        Args
        ----
            review_text : ReviewRequest에서 길이 검증을 마친 고객 리뷰

        Returns
        -------
            REVIEW_SCHEMA에 맞춰 생성된 Gemini JSON 응답을 파싱한 dict(딕셔너리)
        """
        # 1. 프롬프트 만들기
        prompt = f'''주어진 리뷰 텍스트를 분석해주세요.

                리뷰 : {review_text}

                다음 기준으로 분석하세요:
                - sentiment: "긍정", "부정", "중립" 중 하나
                - category: "배송", "품질", "가격", "서비스", "기타" 중 하나
                - summary: 리뷰 핵심을 1~2문장으로 요약
                - confidence: 0.0~1.0 사이의 신뢰도        
                '''

        # 2. Gemini 호출 (여기서부터 실제 네트워크 통신 + 과금이 발생한다.)
        response = self._generate(prompt)

        # 3. 응답 파싱 (JSON -> dict)
        result = json.loads(response.text)
        return result

    def _call_model(self, model: str, prompt: str):
        """지정한 모델로 Gemini를 딱 한 번 호출 한다. (재시도 없다.)"""
        return self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=REVIEW_SCHEMA
            )
        )

    def _generate_with_retry(self, model: str, prompt: str, attempts: int):
        """
        한 모델에 대해 최대 attempts번 시도한다.
        일시적 오류(RETRYABLE_CODES)만 기다렸다 재시도한다.
        """
        for attempt in range(1, attempts+1):
            try:
                return self._call_model(model, prompt)
            except errors.APIError as e:
                if e.code not in RETRYABLE_CODES or attempt == attempts:
                    raise
                wait = self.retry_delay * (2 ** (attempt - 1))  # 1초, 2초, 4초, ...
                logger.warning('Gemini 일시적 오류 %s (%s) - %d/%d번째 시도 실패, %.1f초 뒤 재시도',
                               e.code, model, attempt, attempts, wait)
                time.sleep(wait)

    def _generate(self, prompt: str):
        """기본 모델 -> 계속 일시적 오류이면 대체 모델 순서로 시도한다."""
        try:
            return self._generate_with_retry(self.model, prompt, self.max_retries)
        except errors.APIError as e:
            if self.fallback_model and e.code in RETRYABLE_CODES:
                logger.warning('기본 모델(%s) 실패 %s - 대체 모델(%s)로 1회 시도합니다.',
                               self.model, e.code, self.fallback_model)
                return self._generate_with_retry(self.fallback_model, prompt, 1)
            raise