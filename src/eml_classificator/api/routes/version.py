"""
Version information endpoint.
"""

from fastapi import APIRouter, Query

from ...models.api_models import VersionResponse
from ...version import API_VERSION, get_current_pipeline_version

router = APIRouter()


@ router.get("/version", response_model=VersionResponse)
async def get_version(
    pii_level: str = Query(default="standard", pattern="^(none|minimal|standard|aggressive)$")
) -> VersionResponse:
    """
    Get current API and pipeline version information.

    Args:
        pii_level: PII redaction level to include in pipeline version

    Returns:
        Version information for audit and debugging
    """
    return VersionResponse(
        api_version=API_VERSION,
        pipeline_version=get_current_pipeline_version(pii_redaction_level=pii_level),
    )
