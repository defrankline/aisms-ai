from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import config

engine = create_engine(
    config.DATABASE_URL,
    echo=config.SQL_ECHO,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
