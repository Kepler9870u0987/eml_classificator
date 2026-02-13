"""
Candidate builder for extracting keyword candidates from email documents (Phase 2).

Extracts unigrams, bigrams, and trigrams from subject and body with:
- Deterministic counting (Counter-based)
- Source tracking (subject vs body)
- Stable sorting
- spaCy lemmatization
- Integration with EmailDocument model from Phase 1
"""

from collections import Counter
from typing import List, Tuple

from ..models.email_document import EmailDocument
from ..models.candidates import KeywordCandidate, CandidateSource
from .tokenizer import tokenize_it, lemmatize_term, ngrams, stable_id, TOKENIZER_VERSION


# Minimum term length to consider
MIN_TERM_LENGTH = 3


def build_candidates(
    doc: EmailDocument, max_ngram: int = 3, min_term_length: int = MIN_TERM_LENGTH
) -> List[KeywordCandidate]:
    """
    Extract keyword candidates from EmailDocument.

    Process:
    1. Tokenize subject and canonical_text (body)
    2. Generate n-grams (1..max_ngram)
    3. Count occurrences by (source, term)
    4. Lemmatize terms using spaCy
    5. Create KeywordCandidate objects with stable IDs
    6. Sort deterministically: (source, -count, term)

    Args:
        doc: EmailDocument from Phase 1 ingestion
        max_ngram: Maximum n-gram size (default 3)
        min_term_length: Minimum character length for terms

    Returns:
        List of KeywordCandidate objects, sorted deterministically

    Examples:
        >>> doc = EmailDocument(...)  # with subject="Problema ordine 12345"
        >>> candidates = build_candidates(doc)
        >>> [c.term for c in candidates[:3]]
        ['problema', 'ordine', '12345']  # subject candidates first
    """
    # Extract text from document
    subj = doc.headers.subject or ""
    body = doc.canonical_text or ""

    # Tokenize
    subj_tokens = tokenize_it(subj)
    body_tokens = tokenize_it(body)

    # Collect all (source, term, ngram_size) tuples
    items: List[Tuple[CandidateSource, str, int]] = []

    for source_name, tokens in [("subject", subj_tokens), ("body", body_tokens)]:
        source: CandidateSource = source_name  # type: ignore
        # Generate n-grams for all sizes
        for n in range(1, max_ngram + 1):
            n_grams = ngrams(tokens, n)
            for term in n_grams:
                # Filter by length
                if len(term) < min_term_length:
                    continue
                items.append((source, term, n))

    # Count occurrences by (source, term)
    counts: Counter = Counter((source, term) for source, term, _ in items)

    # Determine ngram_size for each (source, term) pair
    ngram_sizes = {}
    for source, term, ngram_size in items:
        if (source, term) not in ngram_sizes:
            ngram_sizes[(source, term)] = ngram_size

    # Build candidate objects
    candidates = []
    for (source, term), count in counts.items():
        # Lemmatize the term
        lemma = lemmatize_term(term)

        # Get n-gram size
        ngram_size = ngram_sizes.get((source, term), len(term.split()))

        # Generate stable ID
        cid = stable_id(source, term)

        candidates.append(
            KeywordCandidate(
                candidate_id=cid,
                source=source,
                term=term,
                count=count,
                lemma=lemma,
                ngram_size=ngram_size,
                embedding_score=0.0,  # Will be enriched later
                composite_score=0.0,  # Will be calculated later
            )
        )

    # Stable sort: source (subject first), then by count desc, then term alphabetically
    candidates.sort(key=lambda x: (0 if x.source == "subject" else 1, -x.count, x.term))

    return candidates
