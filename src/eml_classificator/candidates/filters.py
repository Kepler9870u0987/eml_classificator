"""
Hard filters for candidate cleaning (Phase 2).

Applies deterministic filters to remove:
- Italian stopwords
- Blacklist patterns (re:, fwd:, isolated numbers)
- Terms below quality thresholds
"""

import re
from typing import List, Optional, Set

from ..models.candidates import KeywordCandidate
from .stopwords import STOPWORDS_IT, STOPLIST_VERSION


# Blacklist patterns (regex)
BLACKLIST_PATTERNS = [
    r"^re:",  # Email reply prefixes
    r"^fwd:",  # Forward prefixes
    r"^r:",  # Short reply
    r"^fw:",  # Short forward
    r"^\d+$",  # Isolated numbers only
    r"^[0-9\-/\.]+$",  # Date-like patterns without context
]


def filter_candidates(
    candidates: List[KeywordCandidate],
    stopwords: Optional[Set[str]] = None,
    blacklist_patterns: Optional[List[str]] = None,
) -> List[KeywordCandidate]:
    """
    Apply hard deterministic filters to candidates.

    Filters applied:
    1. Stopword removal (Italian) - checks both term and lemma
    2. Blacklist pattern matching (regex)
    3. Minimum length check (redundant but safe)

    Args:
        candidates: List of candidates to filter
        stopwords: Custom stopword set (default: STOPWORDS_IT)
        blacklist_patterns: Custom blacklist patterns (default: BLACKLIST_PATTERNS)

    Returns:
        Filtered list of KeywordCandidate objects

    Examples:
        >>> candidates = [
        ...     KeywordCandidate(term="grazie", lemma="grazie", ...),  # stopword
        ...     KeywordCandidate(term="problema", lemma="problema", ...),  # valid
        ... ]
        >>> filtered = filter_candidates(candidates)
        >>> [c.term for c in filtered]
        ['problema']
    """
    if stopwords is None:
        stopwords = STOPWORDS_IT
    if blacklist_patterns is None:
        blacklist_patterns = BLACKLIST_PATTERNS

    filtered = []

    for cand in candidates:
        term_lower = cand.term.lower()
        lemma_lower = cand.lemma.lower()

        # Filter 1: Stopwords (check both term and lemma)
        if lemma_lower in stopwords or term_lower in stopwords:
            continue

        # Filter 2: Blacklist patterns
        is_blacklisted = any(
            re.match(pattern, term_lower, re.IGNORECASE) for pattern in blacklist_patterns
        )
        if is_blacklisted:
            continue

        # Filter 3: Length check (redundant but defensive)
        if len(cand.term) < 3:
            continue

        filtered.append(cand)

    return filtered
