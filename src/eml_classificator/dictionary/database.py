"""
Database connection and session management.

Implements connection pooling and session factory as specified
in brainstorming v2 section 5.1.
"""

from contextlib import contextmanager
from typing import Generator

import structlog
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, scoped_session, sessionmaker

logger = structlog.get_logger(__name__)

# Global singletons
_engine: Engine | None = None
_SessionFactory: scoped_session | None = None


def get_engine() -> Engine:
    """
    Get or create SQLAlchemy engine (singleton).

    Returns:
        SQLAlchemy Engine instance

    Examples:
        >>> engine = get_engine()
        >>> engine.url.database
        'eml_classificator'
    """
    global _engine

    if _engine is None:
        from eml_classificator.config import settings

        _engine = create_engine(
            settings.dictionary_db_url,
            pool_size=settings.dictionary_db_pool_size,
            echo=settings.dictionary_db_echo_sql,
            pool_pre_ping=True,  # Verify connections before using
        )

        logger.info(
            "dictionary_db_engine_created",
            database=_engine.url.database,
            pool_size=settings.dictionary_db_pool_size,
        )

    return _engine


def get_session_factory() -> scoped_session:
    """
    Get or create scoped session factory (singleton).

    Returns:
        Scoped session factory

    Examples:
        >>> factory = get_session_factory()
        >>> session = factory()
    """
    global _SessionFactory

    if _SessionFactory is None:
        engine = get_engine()
        _SessionFactory = scoped_session(sessionmaker(bind=engine))
        logger.info("dictionary_db_session_factory_created")

    return _SessionFactory


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions with automatic commit/rollback.

    Usage:
        >>> with get_db_session() as session:
        ...     entry = LexiconEntry(...)
        ...     session.add(entry)
        ...     # Automatically commits on success, rolls back on exception

    Yields:
        SQLAlchemy Session instance

    Raises:
        Exception: Any database exception (after rollback)
    """
    SessionFactory = get_session_factory()
    session = SessionFactory()

    try:
        yield session
        session.commit()
        logger.debug("dictionary_db_session_committed")
    except Exception as e:
        session.rollback()
        logger.error("dictionary_db_session_rollback", error=str(e), error_type=type(e).__name__)
        raise
    finally:
        session.close()
        logger.debug("dictionary_db_session_closed")


def create_all_tables() -> None:
    """
    Create all database tables.

    WARNING: Only use this for testing or initial setup.
    For production, use Alembic migrations.

    Examples:
        >>> create_all_tables()
        # Tables created: label_registry, lexicon_entries, keyword_observations
    """
    from .models import Base

    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("dictionary_db_tables_created")


def drop_all_tables() -> None:
    """
    Drop all database tables.

    WARNING: Destructive operation. Only use for testing.

    Examples:
        >>> drop_all_tables()
        # All dictionary tables dropped
    """
    from .models import Base

    engine = get_engine()
    Base.metadata.drop_all(engine)
    logger.warning("dictionary_db_tables_dropped")
