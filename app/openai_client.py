'''
app/openai_client.py
----------------------
OpenAI 호출과 프롬프트를 한 곳에 캡슐화한 클라이언트 모듈 (gemini_client.py의 OpenAI 버전)
    1. 환경 변수에서 API키를 읽어 OpenAI 클라이언트를 만든다.
    2. 리뷰 한 건을 프롬프트로 만들어 OpenAI에 보낸다.
    3. 응답(JSON 문자열)을 파이썬 dict로 바꿔 main.py / main_db.py 에 돌려준다.

[바꾸는 방법] 이 파일은 gemini_client.py 를 수정하지 않고 "새로 추가"한 것이다.
클래스 이름(ReviewAnalyzer), analyze() 의 입출력, self.model 이 똑같아서
main_db.py 의 import 한 줄만 바꾸면 나머지 코드는 그대로 동작한다.
    from app.gemini_client import ReviewAnalyzer   -->   from app.openai_client import ReviewAnalyzer
'''
import json
import logging
import os
import time

from openai import APIConnectionError, APIStatusError, OpenAI

logger = logging.getLogger(__name__)

# 재시도 대상 --> 잠시 뒤에 다시 하면 될 수도 있는 오류만!
# 429 : 요청이 너무 많음(분당 한도). 단, "결제 잔액 부족/사용 한도 초과"도 429로 오므로 아래 코드로 따로 걸러낸다.
# 500 : OpenAI 서버 내부 오류, 503 : 모델 과부하, 504 : 응답 시간 초과
# 400(요청이 잘못됨), 401(키가 잘못됨) 등은 다시 해도 소용없으므로 바로 실패.
RETRYABLE_CODES = {429, 500, 502, 503, 504}

# 429 이지만 기다려도 해결되지 않는 경우 (충전하거나 한도를 올려야 한다)
NON_RETRYABLE_429 = {
    "insufficient_quota",
    "credit_balance_exhausted",
    "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
}

# =====================================================================
# 1. 응답 스키마 : 이 JSON 구조로 답해줘.라고 OpenAI에게 요구하는 틀
# =====================================================================
# [Gemini 버전과 다른 점] strict 모드는 ① 모든 object에 additionalProperties: false
# ② 모든 필드를 required 에 넣기, 이 두 가지를 반드시 지켜야 한다.
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string"},
        "category": {"type": "string"},
        "summary": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["sentiment", "category", "summary", "confidence"],
    "additionalProperties": False,
}


# =====================================================================
# 2. 분석기 클래스
# =====================================================================
class ReviewAnalyzer:
    """환경 변수로 OpenAI 클라이언트를 초기화하고 리뷰 한 건을 분석한다."""

    def __init__(self):
        api_key = os.environ.get('OPENAI_API_KEY', '')

        if not api_key:
            raise ValueError('API key가 설정되지 않았습니다.')

        # 재시도는 아래에서 우리가 직접 하므로 SDK의 자동 재시도는 끈다. (둘 다 켜면 시도 횟수가 곱해진다)
        # timeout : 응답이 이 시간(초) 안에 안 오면 포기한다. 무한정 기다리지 않게 하는 안전장치.
        self.client = OpenAI(
            api_key=api_key,
            max_retries=0,
            timeout=float(os.environ.get('OPENAI_TIMEOUT', '30')),
        )

        # [주의] 모델 이름은 계정/시기마다 다르므로 코드에 박지 않고 .env 에서 정한다.
        self.model = os.environ.get('OPENAI_MODEL', 'gpt-6-luna')

        # (선택) 추론(reasoning) 강도. 값을 안 쓰면 요청에 포함하지 않는다.
        # 추론 모델은 "생각하는 시간" 때문에 느릴 수 있다. 낮은 값(예: low)이나 none 을 지원하는
        # 모델이면 응답이 빨라진다. 모델이 지원하지 않는 값을 보내면 400 오류가 난다.
        self.reasoning_effort = os.environ.get('OPENAI_REASONING_EFFORT', '').strip() or None

        # 일시적 오류 대비 설정
        fallback = os.environ.get('OPENAI_FALLBACK_MODEL', '').strip()
        self.fallback_model = fallback if fallback and fallback != self.model else None
        self.max_retries = max(1, int(os.environ.get('OPENAI_MAX_RETRIES', '3')))
        self.retry_delay = float(os.environ.get('OPENAI_RETRY_DELAY', '1.0'))

        logger.info('분석기 초기화 완료! (OpenAI, model=%s)', self.model)

    def analyze(self, review_text: str) -> dict:
        """
        리뷰 텍스트를 감성, 카테고리, 요약, 신뢰도로 구조화한다.

        Args
        ----
            review_text : ReviewRequest에서 길이 검증을 마친 고객 리뷰

        Returns
        -------
            REVIEW_SCHEMA에 맞춰 생성된 OpenAI JSON 응답을 파싱한 dict(딕셔너리)
        """
        # 1. 프롬프트 만들기 (Gemini 버전과 같다)
        prompt = f'''주어진 리뷰 텍스트를 분석해주세요.

                리뷰 : {review_text}

                다음 기준으로 분석하세요:
                - sentiment: "긍정", "부정", "중립" 중 하나
                - category: "배송", "품질", "가격", "서비스", "기타" 중 하나
                - summary: 리뷰 핵심을 1~2문장으로 요약
                - confidence: 0.0~1.0 사이의 신뢰도
                '''

        # 2. OpenAI 호출 (여기서부터 실제 네트워크 통신 + 과금이 발생한다.)
        response = self._generate(prompt)

        # 3. 응답 파싱 (JSON -> dict)
        # output_text 가 비어 있으면 모델이 답변을 거부했거나 끝까지 못 쓴 경우이다.
        text = getattr(response, 'output_text', None)
        if not text:
            status = getattr(response, 'status', None)
            raise ValueError(f'OpenAI 응답이 비어 있습니다. (status={status}) 모델이 거부했거나 중간에 끊겼을 수 있습니다.')
        return json.loads(text)

    def _call_model(self, model: str, prompt: str):
        """지정한 모델로 OpenAI를 딱 한 번 호출 한다. (재시도 없다.)"""
        kwargs = {
            'model': model,
            'input': prompt,
            # Gemini의 response_mime_type + response_schema 에 해당하는 부분
            'text': {
                'format': {
                    'type': 'json_schema',
                    'name': 'review_analysis',
                    'schema': REVIEW_SCHEMA,
                    'strict': True,
                }
            },
        }
        if self.reasoning_effort:
            kwargs['reasoning'] = {'effort': self.reasoning_effort}
        return self.client.responses.create(**kwargs)

    @staticmethod
    def _is_retryable(e: Exception) -> bool:
        """이 오류는 기다렸다 다시 하면 될 가능성이 있는가?"""
        if isinstance(e, APIConnectionError):      # 네트워크 끊김, 시간 초과(APITimeoutError 포함)
            return True
        if isinstance(e, APIStatusError):
            if e.status_code == 429 and getattr(e, 'code', None) in NON_RETRYABLE_429:
                return False                       # 잔액/한도 문제는 기다려도 안 풀린다
            return e.status_code in RETRYABLE_CODES
        return False

    def _generate_with_retry(self, model: str, prompt: str, attempts: int):
        """
        한 모델에 대해 최대 attempts번 시도한다.
        일시적 오류만 기다렸다 재시도한다.
        """
        for attempt in range(1, attempts + 1):
            try:
                return self._call_model(model, prompt)
            except (APIStatusError, APIConnectionError) as e:
                if not self._is_retryable(e) or attempt == attempts:
                    raise
                wait = self.retry_delay * (2 ** (attempt - 1))  # 1초, 2초, 4초, ...
                logger.warning('OpenAI 일시적 오류 %s (%s) - %d/%d번째 시도 실패, %.1f초 뒤 재시도',
                               getattr(e, 'status_code', type(e).__name__), model, attempt, attempts, wait)
                time.sleep(wait)

    def _generate(self, prompt: str):
        """기본 모델 -> 계속 일시적 오류이면 대체 모델 순서로 시도한다."""
        try:
            return self._generate_with_retry(self.model, prompt, self.max_retries)
        except (APIStatusError, APIConnectionError) as e:
            if self.fallback_model and self._is_retryable(e):
                logger.warning('기본 모델(%s) 실패 - 대체 모델(%s)로 1회 시도합니다.',
                               self.model, self.fallback_model)
                return self._generate_with_retry(self.fallback_model, prompt, 1)
            raise