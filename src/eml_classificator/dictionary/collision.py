"""
Collision detection for keywords appearing in multiple labels.

Implements collision detection as specified in brainstorming v2 section 5.2.
"""

from collections import defaultdict
from typing import Dict, List, Set

import structlog

logger = structlog.get_logger(__name__)


def detect_collisions(
    lexicon_entries: List[dict], threshold: int = 2
) -> Dict[str, List[str]]:
    """
    Detect keywords that appear in multiple labels (collisions).

    As specified in brainstorming v2 section 5.2 (lines 1398-1578).

    A keyword collides when it appears in more than `threshold` labels,
    indicating ambiguity that should be resolved before promotion.

    Args:
        lexicon_entries: List of dictionary entries with 'lemma' and 'label_id'
        threshold: Max number of labels allowed (>threshold = collision)

    Returns:
        Dictionary mapping colliding lemmas to list of label_ids

    Examples:
        >>> entries = [
        ...     {"lemma": "fattura", "label_id": "FATTURAZIONE"},
        ...     {"lemma": "fattura", "label_id": "CONTABILITA"},
        ...     {"lemma": "fattura", "label_id": "PAGAMENTO"},
        ...     {"lemma": "problema", "label_id": "ASSISTENZA"},
        ... ]
        >>> collisions = detect_collisions(entries, threshold=2)
        >>> collisions
        {'fattura': ['FATTURAZIONE', 'CONTABILITA', 'PAGAMENTO']}
    """
    lemma_to_labels: Dict[str, Set[str]] = defaultdict(set)

    # Build index of lemma -> {label_ids}
    for entry in lexicon_entries:
        lemma = entry.get("lemma")
        label_id = entry.get("label_id")

        if not lemma or not label_id:
            logger.warning("collision_detection_missing_fields", entry=entry)
            continue

        lemma_to_labels[lemma].add(label_id)

    # Filter collisions (lemmas appearing in > threshold labels)
    collisions = {
        lemma: sorted(list(labels))
        for lemma, labels in lemma_to_labels.items()
        if len(labels) > threshold
    }

    logger.info(
        "collision_detection_complete",
        total_lemmas=len(lemma_to_labels),
        collisions_found=len(collisions),
        threshold=threshold,
    )

    return collisions


def get_collision_index(stats: Dict[str, Dict[str, dict]]) -> Dict[str, Set[str]]:
    """
    Build collision index from statistics.

    For each lemma, find all label_ids where it appears.

    Args:
        stats: Statistics dict from KeywordPromoter.compute_stats()
               Format: {label_id: {lemma: {doc_freq, total_count, ...}}}

    Returns:
        Dictionary mapping lemma to set of label_ids

    Examples:
        >>> stats = {
        ...     "FATTURAZIONE": {"fattura": {"doc_freq": 5}},
        ...     "CONTABILITA": {"fattura": {"doc_freq": 3}},
        ... }
        >>> index = get_collision_index(stats)
        >>> index["fattura"]
        {'FATTURAZIONE', 'CONTABILITA'}
    """
    collision_index: Dict[str, Set[str]] = defaultdict(set)

    for label_id, lemmas in stats.items():
        for lemma in lemmas.keys():
            collision_index[lemma].add(label_id)

    logger.debug(
        "collision_index_built",
        total_lemmas=len(collision_index),
        labels_processed=len(stats),
    )

    return dict(collision_index)


def resolve_collision_by_score(
    lemma: str, collision_labels: List[str], stats: Dict[str, Dict[str, dict]]
) -> str:
    """
    Resolve collision by selecting label with highest score.

    Args:
        lemma: Colliding lemma
        collision_labels: List of label_ids with this lemma
        stats: Statistics dict from KeywordPromoter.compute_stats()

    Returns:
        Label_id with highest embedding score

    Examples:
        >>> stats = {
        ...     "FATTURAZIONE": {"fattura": {"avg_embedding_score": 0.8}},
        ...     "CONTABILITA": {"fattura": {"avg_embedding_score": 0.5}},
        ... }
        >>> winner = resolve_collision_by_score("fattura", ["FATTURAZIONE", "CONTABILITA"], stats)
        >>> winner
        'FATTURAZIONE'
    """
    best_label = collision_labels[0]
    best_score = 0.0

    for label_id in collision_labels:
        if label_id in stats and lemma in stats[label_id]:
            score = stats[label_id][lemma].get("avg_embedding_score", 0.0)
            if score > best_score:
                best_score = score
                best_label = label_id

    logger.debug(
        "collision_resolved",
        lemma=lemma,
        winner=best_label,
        score=best_score,
        candidates=len(collision_labels),
    )

    return best_label
