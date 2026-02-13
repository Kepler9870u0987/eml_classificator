"""
FastAPI application for email ingestion service.

This is the main application that orchestrates all endpoints and middleware.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from ..version import API_VERSION
from ..config import settings
from ..logging_config import setup_logging
from .routes import health, version, ingest, candidates, classification, pipeline
from .middleware import (
    setup_logging_middleware,
    setup_error_handling_middleware,
    setup_metrics_middleware,
)

# Setup logging on module import
setup_logging()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    logger.info(
        "Starting EML Classificator Ingestion API",
        version=API_VERSION,
        log_level=settings.log_level,
        pii_enabled=settings.default_pii_level != "none",
    )
    yield
    logger.info("Shutting down EML Classificator Ingestion API")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI app instance
    """
    app = FastAPI(
        title="Email Classification Pipeline - Full Stack",
        description="Deterministic email parsing, canonicalization, PII redaction, keyword extraction, and LLM classification",
        version=API_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware (order matters - first added = outermost)
    setup_error_handling_middleware(app)
    setup_logging_middleware(app)
    if settings.enable_metrics:
        setup_metrics_middleware(app)

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(version.router, prefix="/api/v1", tags=["Version"])
    app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["Ingestion"])
    app.include_router(candidates.router, tags=["Candidates"])  # Phase 2
    app.include_router(classification.router, tags=["Classification"])  # Phase 3
    app.include_router(pipeline.router, tags=["Pipeline"])  # Full pipeline

    return app


# Create app instance
app = create_app()


def main() -> None:
    """
    Entry point for running the API server directly.

    For development use. In production, use uvicorn directly.
    """
    import uvicorn

    uvicorn.run(
        "eml_classificator.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
