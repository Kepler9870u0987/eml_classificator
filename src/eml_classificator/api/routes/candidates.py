"""
Candidate generation API endpoint (Phase 2).

Provides endpoint to extract keyword candidates from already-ingested documents.
"""

from typing import Optional

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...models.email_document import EmailDocument
from ...models.candidates import CandidateExtractionResult
from ...candidates import extract_candidates
from ...config import settings


logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/candidates", tags=["candidates"])


class ExtractCandidatesRequest(BaseModel):
    """Request model for candidate extraction."""

    document: EmailDocument = Field(description="EmailDocument from Phase 1 ingestion")
    enable_embeddings: bool = Field(
        default=True, description="Whether to use KeyBERT for semantic scoring"
    )
    top_n_keybert: int = Field(default=50, description="Number of keywords to extract from KeyBERT")


@router.post("/extract", response_model=CandidateExtractionResult)
async def extract_candidates_endpoint(request: ExtractCandidatesRequest):
    """
    Extract keyword candidates from an EmailDocument.

    This expects an already-processed EmailDocument (from /ingest/eml).
    Applies tokenization, n-grams, filtering, and optional KeyBERT scoring.

    Args:
        request: ExtractCandidatesRequest with EmailDocument and options

    Returns:
        CandidateExtractionResult with raw, filtered, and enriched candidates

    Raises:
        HTTPException: If candidate extraction fails

    Examples:
        POST /api/v1/candidates/extract
        {
            "document": {...},  // EmailDocument
            "enable_embeddings": true,
            "top_n_keybert": 50
        }
    """
    logger.info(
        "Received candidate extraction request",
        document_id=request.document.document_id,
        enable_embeddings=request.enable_embeddings,
    )

    try:
        # Extract candidates using Phase 2 pipeline
        result = extract_candidates(
            doc=request.document,
            enable_embeddings=request.enable_embeddings,
            top_n_keybert=request.top_n_keybert,
            score_weights={
                "count": settings.score_weight_count,
                "embedding": settings.score_weight_embedding,
                "subject_bonus": settings.score_weight_subject_bonus,
            },
        )

        logger.info(
            "Candidate extraction successful",
            document_id=request.document.document_id,
            enriched_candidates_count=len(result.enriched_candidates),
            processing_time_ms=result.processing_time_ms,
        )

        return result

    except Exception as e:
        logger.error(
            "Candidate extraction failed",
            document_id=request.document.document_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Candidate extraction failed: {str(e)}"
        ) from e


@router.get("/health")
async def health_check():
    """
    Health check endpoint for candidate generation service.

    Returns:
        Dict with status and configuration info
    """
    return {
        "status": "healthy",
        "service": "candidate_generation",
        "embeddings_enabled": settings.enable_embeddings,
        "spacy_model": settings.spacy_model_name,
    }
