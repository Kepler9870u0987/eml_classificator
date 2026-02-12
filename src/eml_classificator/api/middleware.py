"""
FastAPI middleware for logging, metrics, and error handling.
"""

import time
from typing import Callable
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import structlog

logger = structlog.get_logger(__name__)


def setup_logging_middleware(app: FastAPI) -> None:
    """
    Setup request/response logging middleware.

    Logs all requests with:
    - Request method, path, query params
    - Response status code
    - Processing time
    """

    @app.middleware("http")
    async def log_requests(request: Request, call_next: Callable) -> Response:
        # Start timer
        start_time = time.time()

        # Log incoming request
        logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
            query_params=str(request.query_params),
            client=request.client.host if request.client else None,
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Log response
            logger.info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                process_time_ms=round(process_time * 1000, 2),
            )

            # Add custom header
            response.headers["X-Process-Time"] = str(process_time)

            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                "Request failed",
                method=request.method,
                path=request.url.path,
                process_time_ms=round(process_time * 1000, 2),
                error=str(e),
                exc_info=True,
            )
            raise


def setup_error_handling_middleware(app: FastAPI) -> None:
    """
    Setup global error handling middleware.

    Catches unhandled exceptions and returns consistent error responses.
    """

    @app.middleware("http")
    async def handle_errors(request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            logger.error(
                "Unhandled exception",
                method=request.method,
                path=request.url.path,
                error=str(e),
                exc_info=True,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Internal server error",
                    "detail": str(e) if app.debug else "An unexpected error occurred",
                },
            )


def setup_metrics_middleware(app: FastAPI) -> None:
    """
    Setup basic metrics middleware.

    Tracks:
    - Request count per endpoint
    - Response time per endpoint
    - Status code distribution

    Note: For production, use prometheus-fastapi-instrumentator
    """
    # Placeholder for metrics collection
    # In production, this would integrate with Prometheus

    @app.middleware("http")
    async def collect_metrics(request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        # Log metrics
        logger.debug(
            "Metrics",
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            duration_seconds=process_time,
        )

        return response
