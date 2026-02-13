"""
Data models for keyword candidate generation (Phase 2).

This module defines structures for deterministic candidate extraction,
tracking, and scoring from email documents.
"""

from typing import List, Literal
from pydantic import BaseModel, Field


# Source type for candidate origin tracking
CandidateSource = Literal["subject", "body"]


class KeywordCandidate(BaseModel):
    """
    A single keyword candidate extracted from an email.

    Deterministic representation with stable ID and complete provenance.
    """

    candidate_id: str = Field(description="Stable SHA1-based ID: stable_id(source, term)")
    source: CandidateSource = Field(description="Origin: 'subject' or 'body'")
    term: str = Field(description="The n-gram term (unigram, bigram, or trigram)")
    count: int = Field(description="Frequency in the source text", ge=1)
    lemma: str = Field(description="Lemmatized form using spaCy")
    ngram_size: int = Field(description="1=unigram, 2=bigram, 3=trigram", ge=1, le=3)

    # Semantic enrichment (from KeyBERT)
    embedding_score: float = Field(
        default=0.0, description="Cosine similarity score from KeyBERT (0.0-1.0)", ge=0.0, le=1.0
    )
    composite_score: float = Field(
        default=0.0,
        description="Combined score: count + embedding + subject_bonus",
        ge=0.0,
    )


class CandidateExtractionResult(BaseModel):
    """
    Complete result of candidate extraction from an email document.

    Includes raw candidates, filtered candidates, and processing metadata.
    """

    document_id: str = Field(description="Reference to source EmailDocument")

    # Candidates at different stages
    raw_candidates: List[KeywordCandidate] = Field(
        default_factory=list, description="All extracted candidates before filtering"
    )
    filtered_candidates: List[KeywordCandidate] = Field(
        default_factory=list, description="Candidates after stopword/blacklist filtering"
    )
    enriched_candidates: List[KeywordCandidate] = Field(
        default_factory=list, description="Final candidates with embedding scores"
    )

    # Processing metadata (versioning for determinism)
    tokenizer_version: str = Field(description="Tokenizer version for reproducibility")
    stoplist_version: str = Field(description="Stopwords list version (e.g., 'stopwords-it-2025.1')")
    embedding_model_version: str = Field(description="KeyBERT model version or 'none'")
    ner_model_version: str = Field(description="spaCy NER model version for lemmatization")

    # Statistics
    total_tokens_subject: int = Field(description="Token count in subject", ge=0)
    total_tokens_body: int = Field(description="Token count in body", ge=0)
    candidates_before_filter: int = Field(description="Count before filtering", ge=0)
    candidates_after_filter: int = Field(description="Count after filtering", ge=0)
    processing_time_ms: float = Field(description="Extraction time in milliseconds", ge=0.0)
