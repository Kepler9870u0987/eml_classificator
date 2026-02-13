"""
Confidence adjustment for topic classifications.

Implements confidence calibration as specified in brainstorming v2 section 4.3.

Combines multiple signals to produce adjusted confidence:
- LLM confidence (30%) - model's own assessment
- Keyword quality (40%) - composite scores of selected keywords
- Evidence coverage (20%) - number and quality of evidence quotes
- Collision penalty (10%) - keyword ambiguity across labels

This provides more reliable confidence than raw LLM output.
"""

from typing import Dict, List, Set, Optional
import numpy as np

import structlog

from eml_classificator.classification.schemas import TopicAssignment
from eml_classificator.config import settings


logger = structlog.get_logger(__name__)


# ============================================================================
# COLLISION INDEX
# ============================================================================

def build_collision_index(
    all_topics: List[TopicAssignment],
    additional_lexicon: Optional[Dict[str, Set[str]]] = None
) -> Dict[str, Set[str]]:
    """
    Build collision index: lemma → set of label_ids where it appears.

    A keyword that appears in multiple topics is "ambiguous" and should
    have reduced confidence contribution.

    Args:
        all_topics: List of all topic assignments in this classification
        additional_lexicon: Optional external lexicon {lemma: {label_ids}}

    Returns:
        Dict mapping lemma → set of label_ids
    """
    collision_index: Dict[str, Set[str]] = {}

    # Add from current topics
    for topic in all_topics:
        label_id = topic.label_id.value if hasattr(topic.label_id, 'value') else topic.label_id

        for kw in topic.keywords_in_text:
            lemma = kw.lemma
            if lemma not in collision_index:
                collision_index[lemma] = set()
            collision_index[lemma].add(label_id)

    # Merge with additional lexicon if provided
    if additional_lexicon:
        for lemma, label_ids in additional_lexicon.items():
            if lemma not in collision_index:
                collision_index[lemma] = set()
            collision_index[lemma].update(label_ids)

    return collision_index


# ============================================================================
# CONFIDENCE ADJUSTMENT
# ============================================================================

def compute_topic_confidence_adjusted(
    topic: TopicAssignment,
    candidates: List[Dict],
    collision_index: Dict[str, Set[str]],
    llm_confidence: float,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute adjusted confidence for a topic.

    Combines:
    - LLM confidence (weight: 0.3)
    - Keyword quality score (weight: 0.4)
    - Evidence coverage (weight: 0.2)
    - Collision penalty (weight: 0.1)

    Args:
        topic: TopicAssignment to score
        candidates: List of candidate dicts (with composite_score)
        collision_index: Dict mapping lemma → set of label_ids
        llm_confidence: Original confidence from LLM
        weights: Optional custom weights dict

    Returns:
        Adjusted confidence in [0.0, 1.0]
    """
    if weights is None:
        weights = {
            "llm": settings.confidence_weight_llm,
            "keywords": settings.confidence_weight_keywords,
            "evidence": settings.confidence_weight_evidence,
            "collision": settings.confidence_weight_collision,
        }

    # Validate weights sum to 1.0
    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0):
        logger.warning(
            "confidence_weights_do_not_sum_to_1",
            weights=weights,
            sum=weight_sum
        )
        # Normalize
        weights = {k: v / weight_sum for k, v in weights.items()}

    # Component 1: LLM confidence
    llm_component = llm_confidence

    # Component 2: Keyword quality
    keyword_component = _compute_keyword_quality_score(topic, candidates)

    # Component 3: Evidence coverage
    evidence_component = _compute_evidence_coverage_score(topic)

    # Component 4: Collision penalty (inverted - lower is better)
    collision_component = _compute_collision_penalty_score(topic, collision_index)

    # Weighted combination
    adjusted_confidence = (
        weights["llm"] * llm_component +
        weights["keywords"] * keyword_component +
        weights["evidence"] * evidence_component +
        weights["collision"] * collision_component
    )

    # Clip to [0.0, 1.0]
    adjusted_confidence = np.clip(adjusted_confidence, 0.0, 1.0)

    logger.debug(
        "topic_confidence_adjusted",
        label_id=topic.label_id,
        llm_confidence=llm_confidence,
        adjusted_confidence=adjusted_confidence,
        components={
            "llm": llm_component,
            "keywords": keyword_component,
            "evidence": evidence_component,
            "collision": collision_component
        }
    )

    return float(adjusted_confidence)


def _compute_keyword_quality_score(
    topic: TopicAssignment,
    candidates: List[Dict]
) -> float:
    """
    Compute keyword quality component.

    Uses composite_score from candidates to assess keyword quality.

    Args:
        topic: TopicAssignment
        candidates: List of candidate dicts

    Returns:
        Score in [0.0, 1.0]
    """
    if not topic.keywords_in_text:
        return 0.1  # Very low if no keywords

    # Build candidate map
    cand_map = {c["candidate_id"]: c for c in candidates}

    # Get scores for selected keywords
    keyword_scores = []
    for kw in topic.keywords_in_text:
        cand = cand_map.get(kw.candidate_id)
        if cand:
            # Get composite_score (should be in [0, 1] range)
            score = cand.get("composite_score", 0.0)
            keyword_scores.append(score)
        else:
            # Keyword not in candidate map (shouldn't happen after validation)
            keyword_scores.append(0.0)

    if not keyword_scores:
        return 0.1

    # Average keyword score
    avg_score = np.mean(keyword_scores)

    # Normalize to [0, 1] if needed (composite_score should already be normalized)
    return float(np.clip(avg_score, 0.0, 1.0))


def _compute_evidence_coverage_score(topic: TopicAssignment) -> float:
    """
    Compute evidence coverage component.

    More evidence = higher confidence.
    Normalized by expected evidence count (2).

    Args:
        topic: TopicAssignment

    Returns:
        Score in [0.0, 1.0]
    """
    evidence_count = len(topic.evidence)

    if evidence_count == 0:
        return 0.2  # Low score if no evidence

    # Normalize: 2+ evidence = full score
    normalized_score = min(evidence_count / 2.0, 1.0)

    return normalized_score


def _compute_collision_penalty_score(
    topic: TopicAssignment,
    collision_index: Dict[str, Set[str]]
) -> float:
    """
    Compute collision penalty component.

    Keywords that appear in multiple topics are ambiguous and reduce confidence.

    Args:
        topic: TopicAssignment
        collision_index: Dict mapping lemma → set of label_ids

    Returns:
        Score in [0.0, 1.0] where 1.0 = no collision, 0.0 = maximum collision
    """
    if not topic.keywords_in_text:
        return 0.5  # Neutral if no keywords

    label_id = topic.label_id.value if hasattr(topic.label_id, 'value') else topic.label_id

    penalties = []
    for kw in topic.keywords_in_text:
        lemma = kw.lemma
        labels_with_lemma = collision_index.get(lemma, {label_id})

        if len(labels_with_lemma) > 1:
            # Keyword is ambiguous (appears in multiple labels)
            # Penalty proportional to number of collisions
            penalty = 1.0 / len(labels_with_lemma)
            penalties.append(penalty)
        else:
            # No collision - full score
            penalties.append(1.0)

    # Average penalty across all keywords
    avg_penalty = np.mean(penalties) if penalties else 0.5

    return float(avg_penalty)


# ============================================================================
# BULK ADJUSTMENT
# ============================================================================

def adjust_all_topic_confidences(
    topics: List[TopicAssignment],
    candidates: List[Dict],
    external_lexicon: Optional[Dict[str, Set[str]]] = None,
    weights: Optional[Dict[str, float]] = None
) -> List[TopicAssignment]:
    """
    Adjust confidence for all topics in a classification result.

    Args:
        topics: List of TopicAssignments
        candidates: List of candidate dicts
        external_lexicon: Optional external lexicon for collision detection
        weights: Optional custom confidence weights

    Returns:
        List of TopicAssignments with adjusted confidence
    """
    # Build collision index
    collision_index = build_collision_index(topics, external_lexicon)

    # Adjust each topic
    for topic in topics:
        llm_conf = topic.confidence

        adjusted_conf = compute_topic_confidence_adjusted(
            topic=topic,
            candidates=candidates,
            collision_index=collision_index,
            llm_confidence=llm_conf,
            weights=weights
        )

        # Store both original and adjusted
        topic.confidence_adjusted = adjusted_conf
        # Optionally override main confidence field
        # topic.confidence = adjusted_conf

    logger.info(
        "all_topic_confidences_adjusted",
        topics_count=len(topics),
        avg_adjustment=np.mean([
            abs(t.confidence - (t.confidence_adjusted or t.confidence))
            for t in topics
        ])
    )

    return topics


# ============================================================================
# CONFIDENCE THRESHOLDING
# ============================================================================

def filter_topics_by_confidence(
    topics: List[TopicAssignment],
    threshold: float = 0.3,
    use_adjusted: bool = True
) -> List[TopicAssignment]:
    """
    Filter topics below confidence threshold.

    Args:
        topics: List of TopicAssignments
        threshold: Minimum confidence (default: from settings)
        use_adjusted: Use adjusted confidence if available

    Returns:
        Filtered list of topics
    """
    if threshold is None:
        threshold = settings.classification_confidence_threshold

    filtered = []
    for topic in topics:
        # Use adjusted confidence if available and requested
        conf = topic.confidence_adjusted if (use_adjusted and topic.confidence_adjusted) else topic.confidence

        if conf >= threshold:
            filtered.append(topic)
        else:
            logger.debug(
                "topic_filtered_by_confidence",
                label_id=topic.label_id,
                confidence=conf,
                threshold=threshold
            )

    # Ensure at least one topic (UNKNOWN_TOPIC fallback)
    if not filtered:
        logger.warning(
            "all_topics_filtered_by_confidence",
            original_count=len(topics),
            threshold=threshold
        )
        # Could return UNKNOWN_TOPIC here as fallback

    return filtered


# ============================================================================
# CONFIDENCE CALIBRATION (ADVANCED)
# ============================================================================

def calibrate_confidence_from_validation_set(
    predictions: List[Dict],
    ground_truth: List[Dict]
):
    """
    Calibrate confidence using isotonic regression or similar.

    This is a placeholder for future implementation to improve
    confidence calibration using labeled validation data.

    Args:
        predictions: List of {topic, confidence_predicted} dicts
        ground_truth: List of {topic, is_correct} dicts
    """
    # TODO: Implement confidence calibration
    # Could use sklearn's CalibratedClassifierCV or isotonic regression
    logger.warning("calibrate_confidence_from_validation_set not yet implemented")
    pass
