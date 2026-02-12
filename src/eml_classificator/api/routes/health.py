"""
Health check endpoint for monitoring.
"""

import time
from fastapi import APIRouter

from ...models.api_models import HealthResponse
from ...version import API_VERSION

router = APIRouter()

# Track start time for uptime calculation
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint for monitoring.

    Returns:
        Health status and uptime
    """
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        uptime_seconds=time.time() - _start_time,
    )
