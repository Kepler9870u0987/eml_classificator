# Data models for the email classification pipeline

from .pipeline_version import PipelineVersion
from .email_document import AttachmentMetadata, EmailDocument, EmailHeaders
from .api_models import (
    HealthResponse,
    IngestEmlRequest,
    IngestEmlResponse,
    VersionResponse,
)

__all__ = [
    "PipelineVersion",
    "EmailDocument",
    "EmailHeaders",
    " AttachmentMetadata",
    "IngestEmlRequest",
    "IngestEmlResponse",
    "HealthResponse",
    "VersionResponse",
]
