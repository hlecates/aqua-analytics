import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[4]  # .../aqua-analytics
_DEFAULT_DB = _REPO / "nescac" / "app" / "backend" / "instance" / "nescac.sqlite3"

DATABASE_URL = os.getenv("NESCAC_DB_URL", f"sqlite:///{_DEFAULT_DB}")

# Ensure instance dir exists if using default SQLite location
if DATABASE_URL.startswith("sqlite:///"):
    _DEFAULT_DB.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    future=True,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)
Base = declarative_base()

def init_db() -> None:
    from . import models  # ensure metadata registered
    Base.metadata.create_all(bind=engine)

@contextmanager
def get_session() -> Iterator[SessionLocal]:
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
