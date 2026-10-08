'''
app/database.py
---------------
PostgreSQL 연결
'''
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql+psycopg2://postgres:1234@localhost:5432/review_db',
)

# pool_pre_ping=True --> 연결이 끊겼으면 사용하기 전에 확인해서 다시 연결한다.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

class Base(DeclarativeBase):
    pass

def get_db():
    """요청마다 DB 세션을 하나 열고, 끝나면 반드시 닫는다.(FastAPI의 Depends로 주입)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()