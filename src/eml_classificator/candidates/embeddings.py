"""
KeyBERT integration for semantic candidate scoring (Phase 2).

Uses sentence-transformers for multilingual embeddings to add semantic
relevance scores to candidates extracted from email text.
"""

from typing import Dict, List, Optional

import numpy as np
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

from ..models.email_document import EmailDocument
from ..models.candidates import KeywordCandidate


# Model version for reproducibility
EMBEDDING_MODEL_VERSION = "paraphrase-multilingual-mpnet-base-v2"

# Singleton model instance (lazy-loaded)
_keybert_model: Optional[KeyBERT] = None


def get_keybert_model() -> KeyBERT:
    """
    Get or initialize KeyBERT model (singleton pattern).

    Uses paraphrase-multilingual-mpnet-base-v2 for multilingual support.
    Model is loaded once and cached for performance (~420MB download on first use).

    Returns:
        Initialized KeyBERT instance with multilingual model
    """
    global _keybert_model
    if _keybert_model is None:
        transformer = SentenceTransformer(EMBEDDING_MODEL_VERSION)
        _keybert_model = KeyBERT(model=transformer)
    return _keybert_model


def enrich_candidates_with_embeddings(
    doc: EmailDocument,
    candidates: List[KeywordCandidate],
    top_n: int = 50,
    diversity: float = 0.7,
) -> List[KeywordCandidate]:
    """
    Enrich candidates with KeyBERT semantic scores.

    Process:
    1. Combine subject + canonical_text
    2. Extract keywords with KeyBERT (using MMR for diversity)
    3. Map KeyBERT scores to candidate terms
    4. Update embedding_score field

    Args:
        doc: Source EmailDocument
        candidates: List of candidates to enrich
        top_n: Number of top keywords to extract from KeyBERT
        diversity: MMR diversity parameter (0.0-1.0, higher = more diverse)

    Returns:
        Updated candidates with embedding_score populated

    Examples:
        >>> doc = EmailDocument(...)  # with text about "fattura"
        >>> candidates = [KeywordCandidate(term="fattura", ...)]
        >>> enriched = enrich_candidates_with_embeddings(doc, candidates)
       >>> enriched[0].embedding_score  # > 0.0 if relevant
        0.75
    """
    # Combine text
    full_text = f"{doc.headers.subject or ''}\n{doc.canonical_text or ''}"

    if not full_text.strip():
        return candidates

    # Extract keywords with KeyBERT
    kw_model = get_keybert_model()

    try:
        keybert_keywords = kw_model.extract_keywords(
            full_text,
            keyphrase_ngram_range=(1, 3),
            stop_words="italian",
            top_n=top_n,
            use_mmr=True,  # Maximal Marginal Relevance for diversity
            diversity=diversity,
        )
    except Exception as e:
        # Log error but continue (don't fail pipeline)
        print(f"KeyBERT extraction failed: {e}")
        keybert_keywords = []

    # Build score mapping: term -> score
    keybert_scores: Dict[str, float] = {kw: score for kw, score in keybert_keywords}

    # Enrich candidates
    for cand in candidates:
        # Try exact match on term or lemma
        score = keybert_scores.get(cand.term, keybert_scores.get(cand.lemma, 0.0))
        cand.embedding_score = float(score)

    return candidates


def calculate_composite_scores(
    candidates: List[KeywordCandidate], weights: Optional[Dict[str, float]] = None
) -> List[KeywordCandidate]:
    """
    Calculate composite scores combining count, embedding, and source bonus.

    Formula:
        composite_score =
            weight_count * log1p(count) / 5.0 +
            weight_embedding * embedding_score +
            weight_subject * (1.0 if source=='subject' else 0.0)

    Args:
        candidates: List of candidates to score
        weights: Weight dict (default: count=0.3, embedding=0.5, subject_bonus=0.2)

    Returns:
        Updated candidates with composite_score populated

    Examples:
        >>> candidates = [
        ...     KeywordCandidate(source="subject", count=2, embedding_score=0.8, ...),
        ...     KeywordCandidate(source="body", count=2, embedding_score=0.8, ...),
        ... ]
        >>> scored = calculate_composite_scores(candidates)
        >>> scored[0].composite_score > scored[1].composite_score  # subject bonus
        True
    """
    if weights is None:
        weights = {"count": 0.3, "embedding": 0.5, "subject_bonus": 0.2}

    for cand in candidates:
        # Normalize count with log1p
        count_norm = np.log1p(cand.count) / 5.0

        # Embedding score (already 0.0-1.0)
        embedding_score = cand.embedding_score

        # Subject bonus (binary)
        subject_bonus = 1.0 if cand.source == "subject" else 0.0

        # Calculate composite
        cand.composite_score = (
            weights["count"] * count_norm
            + weights["embedding"] * embedding_score
            + weights["subject_bonus"] * subject_bonus
        )

    return candidates
