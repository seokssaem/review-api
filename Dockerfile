# 리뷰 분석 API (review-api)를 위한 Dockerfile

FROM python:3.10-slim

# 컨테이너 안에서 애플리케이션 파일을 배치하고 실행할 기준 위치
WORKDIR /app

# 의존성 목록 파일을 먼저 복사 - 라이브러리를 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# .dockerignore 에서 제외하지 않은 프로젝트 소스코드 전체를 이미지에 복사
COPY . .

# 컨테이너 내부 수신 포트를 문서화한다. - 포트 겹치게 하지 않기 위해.
EXPOSE 8005

# app/main.py의 FastAPI 객체를 Uvicorn으로 실행한다.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005", "--reload"]