from collections.abc import AsyncGenerator

from fastapi import Request
from sqlalchemy.ext.asyncio import (
    AsyncSession, async_sessionmaker,
    create_async_engine, AsyncEngine,
)
from backend.app.core.config import get_settings
from backend.app.models.db_models import Base

settings = get_settings()

def setup_db() -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """Initialize database connection pool and create tables."""
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        pool_size=20,
        max_overflow=10,
        pool_pre_ping=True
    )

    db_sessionmaker = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    return engine, db_sessionmaker

async def init_models(engine: AsyncEngine) -> None:
    """Create tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for database session.
    Provides a session for each FastAPI request.
    """
    db_sessionmaker = getattr(request.app.state, "db_sessionmaker", None)
    if not db_sessionmaker:
        raise RuntimeError("Database sessionmaker not found in app state. Did lifespan run?")
    
    async with db_sessionmaker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise

# async def close_db() -> None:
#     """Close database connection pool."""
#     global engine
#     if engine is not None:
#         await engine.dispose()
#         engine = None
