"""
Keyword candidate generation module (Phase 2).

Public API for extracting, filtering, and scoring keyword candidates from
EmailDocument objects produced by Phase 1 ingestion.
"""

import time
from typing import Dict, List, Optional

import structlog

from ..models.email_document import EmailDocument
from ..models.candidates import CandidateExtractionResult, KeywordCandidate
from .tokenizer import TOKENIZER_VERSION, NER_MODEL_VERSION, tokenize_it
from .candidate_builder import build_candidates
from .embeddings import (
    EMBEDDING_MODEL_VERSION,
    enrich_candidates_with_embeddings,
    calculate_composite_scores,
)
from .filters import filter_candidates, STOPLIST_VERSION
from .stopwords import STOPWORDS_IT


logger = structlog.get_logger()

__all__ = [
    "extract_candidates",
    "CandidateExtractionResult",
    "KeywordCandidate",
    "TOKENIZER_VERSION",
    "STOPLIST_VERSION",
    "EMBEDDING_MODEL_VERSION",
    "NER_MODEL_VERSION",
]


def extract_candidates(
    doc: EmailDocument,
    enable_embeddings: bool = True,
    top_n_keybert: int = 50,
    score_weights: Optional[Dict[str, float]] = None,
) -> CandidateExtractionResult:
    """
    Complete candidate extraction pipeline from EmailDocument.

    Pipeline stages:
    1. Build raw candidates (tokenize, n-gram, count, lemmatize)
    2. Filter candidates (stopwords, blacklist)
    3. Enrich with embeddings (KeyBERT) [optional]
    4. Calculate composite scores
    5. Sort by composite_score descending

    Args:
        doc: EmailDocument from Phase 1
        enable_embeddings: Whether to use KeyBERT (default True)
        top_n_keybert: Number of keywords to extract from KeyBERT
        score_weights: Custom weights for composite scoring

    Returns:
        CandidateExtractionResult with all pipeline stages and metadata

    Examples:
        >>> from eml_classificator.parsing.eml_parser import parse_eml_bytes
        >>> eml_bytes = open("sample.eml", "rb").read()
        >>> parsed = parse_eml_bytes(eml_bytes)
        >>> # ... build EmailDocument ...
        >>> result = extract_candidates(doc)
        >>> print(f"Top candidate: {result.enriched_candidates[0].term}")
        'problema'
    """
    start_time = time.time()

    logger.info(
        "Starting candidate extraction",
        document_id=doc.document_id,
        subject_length=len(doc.headers.subject or ""),
        body_length=len(doc.canonical_text or ""),
        enable_embeddings=enable_embeddings,
    )

    # Stage 1: Build raw candidates
    raw_candidates = build_candidates(doc)

    logger.debug(
        "Built raw candidates",
        document_id=doc.document_id,
        raw_count=len(raw_candidates),
    )

    # Stage 2: Filter candidates
    filtered_candidates = filter_candidates(raw_candidates)

    logger.debug(
        "Filtered candidates",
        document_id=doc.document_id,
        filtered_count=len(filtered_candidates),
        removed_count=len(raw_candidates) - len(filtered_candidates),
    )

    # Stage 3: Enrich with embeddings (if enabled)
    if enable_embeddings:
        enriched_candidates = enrich_candidates_with_embeddings(
            doc, filtered_candidates, top_n=top_n_keybert
        )
        embedding_model_version = EMBEDDING_MODEL_VERSION
    else:
        enriched_candidates = filtered_candidates
        embedding_model_version = "none"

    # Stage 4: Calculate composite scores
    enriched_candidates = calculate_composite_scores(enriched_candidates, weights=score_weights)

    # Sort by composite score descending
    enriched_candidates.sort(key=lambda x: x.composite_score, reverse=True)

    # Calculate statistics
    subj_tokens = tokenize_it(doc.headers.subject or "")
    body_tokens = tokenize_it(doc.canonical_text or "")

    processing_time_ms = (time.time() - start_time) * 1000

    logger.info(
        "Candidate extraction complete",
        document_id=doc.document_id,
        raw_candidates=len(raw_candidates),
        filtered_candidates=len(filtered_candidates),
        enriched_candidates=len(enriched_candidates),
        processing_time_ms=round(processing_time_ms, 2),
    )

    # Build result
    result = CandidateExtractionResult(
        document_id=doc.document_id,
        raw_candidates=raw_candidates,
        filtered_candidates=filtered_candidates,
        enriched_candidates=enriched_candidates,
        tokenizer_version=TOKENIZER_VERSION,
        stoplist_version=STOPLIST_VERSION,
        embedding_model_version=embedding_model_version,
        ner_model_version=NER_MODEL_VERSION,
        total_tokens_subject=len(subj_tokens),
        total_tokens_body=len(body_tokens),
        candidates_before_filter=len(raw_candidates),
        candidates_after_filter=len(enriched_candidates),
        processing_time_ms=processing_time_ms,
    )

    return result
