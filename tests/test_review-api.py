'''
tests/test_review-api.py
-------------------------

1. 스키마 테스트 : ReviewRequest / ReviewResponse 검증 규칙
2. 분석기 테스트 : ReviewAnalyzer (Gemini SDK를 가짜로 바꿔 끼워서 진행)
3. API 테스트 : /health, /analyze (분석기를 가짜로 바꿔 끼워서 진행)
4. 실제 Gemini 테스트 : RUN_LIVE_GEMINI=1 일 때만 실행

가짜(Fake) 객체로 바꿔 끼울 수 있는 이유
    main.py가 Gemini를 직접 부르지 않고, app.state.analyzer에 위임하는 구조기 때문
    분리해서 만들면 테스트가 쉬워진다라는 사례를 보여준다.
'''
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from app import main as main_module
from app.schemas import ReviewRequest

# tests/ 의 상위 폴더(review-api)를 import 경로에 추가해서 'from app....' 동작 가능하게 한다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 테스트 1: 스키마 - 빈 문자열을 거부해야 한다 (단위 테스트)
def test_빈_리뷰는_거부한다():
    # pytest.raises: 이 안에서 ValidationError가 나야 통과! 
    #                   에러가 나지 않으면 테스트가 실패!
    with pytest.raises(ValidationError):
        ReviewRequest(review_text='')

# 테스트 2: 경계값 - 5000자는 통과, 5001자는 거부 
# parametrize --> 같은 테스트를 입력값만 바꿔서 여러번 실행해라!
#   'length, ok' --> 테스트함수가 받을 변수 이름들(아래 함수의 매개변수와 같아야 한다.)
#   [(5000, True), (5001, False)] --> 실행해야 하는 케이스 목록. 한 쌍이 한 번의 실행
#       1번째 실행: length=5000, ok=True
#       2번째 실행: length=5001, ok=False 
@pytest.mark.parametrize('length, ok', [(5000, True), (5001, False)])
def test_길이_경계값(length, ok):
    text = '가' * length
    if ok:
        ReviewRequest(review_text=text) # 에러가 나면 테스트 실패
    else:
        with pytest.raises(ValidationError):
            ReviewRequest(review_text=text)

# 테스트 3: API - 가짜 분석기를 끼워 /analyze가 200을 주는지 (통합 테스트)            
class FakeAnalyzer:
    """Gemini를 부르지 않고 미리 정한 결과를 돌려주는 가짜 분석기"""
    def analyze(self, review_text):
        return {
            'sentiment': '긍정',
            'category': '배송',
            'summary': '배송이 빠르다.',
            'confidence': 0.9,
        }

def test_analyze_성공(monkeypatch):
    """
    monkeypatch --> 함수 매개변수 이름만 적으면 pytest가 알아서 넣어주는 도구
    
    1. 바꿔치기: main.py안의 ReviewAnalyzer 라는 이름을 가짜로 교체한다.
        main.py의 lifespan은 서버가 시작될 때 app.state.analyzer = ReviewAnalyzer()를 실행
        진짜 ReviewAnalyzer()는 API key가 없으면 ValueError로 서버 시작이 실패하고 있으면
        Gemini를 호출해서 과금한다. 그래서 테스트에서는 이 자리를 가짜가 대신하게 한다.

        monkeypatch.setattr(대상, "이름", 새 값)
            대상 : main_module  --> app/main.py 모듈
            "이름" : "ReviewAnalyzer()" --> 그 모듈 안에서 바꿀 이름
            새 값 : lambda: FakeAnalyzer() --> 호출되면 FakeAnalyzer() 객체를 돌려주는 함수

    2. 서버를 띄운 것처럼 만들기: TestClient --> 진짜 서버 없이 요청을 보내 볼 수 있는 가짜 브라우저

    3. 결과 확인: 둘 중 하나라도 틀리면 테스트 실패
    """
    monkeypatch.setattr(main_module, "ReviewAnalyzer", lambda: FakeAnalyzer())  # 1

    with TestClient(main_module.app) as client:
        r = client.post('/analyze', json={"review_text":"배송 빨라요!"})  # 2

    # 3
    assert r.status_code == 200  # 정상 응답인가?? 
    assert r.json()["sentiment"] == "긍정" # 가짜 분석기가 준 값이 그대로 응답에 담겼는가??

    # 테스트가 끝나면 monkeypatch가 ReviewAnalyzer를 원래 본 클래스로 자동 복구한다.