"""
Keyword observation collection from classification results.

Implements observation collection as specified in brainstorming v2 section 5.2.
"""

import uuid
from datetime import datetime
from typing import List

import structlog

from eml_classificator.classification.schemas import ClassificationResult

from .database import get_db_session
from .models import KeywordObservation

logger = structlog.get_logger(__name__)


def collect_observations_from_classification(
    classification_result: ClassificationResult, message_id: str, dict_version: int
) -> List[str]:
    """
    Extract keyword observations from classification result and save to database.

    As specified in brainstorming v2 section 5.2 (lines 1398-1578).

    After each classification, collect keywords from each topic to build
    statistics for promotion decisions.

    Args:
        classification_result: Classification output with topics and keywords
        message_id: Unique email/message identifier
        dict_version: Dictionary version used for this classification

    Returns:
        List of created observation IDs

    Examples:
        >>> result = ClassificationResult(...)  # With topics and keywords
        >>> obs_ids = collect_observations_from_classification(result, "msg123", 1)
        >>> len(obs_ids)
        15  # Number of keywords collected
    """
    obs_ids = []

    try:
        with get_db_session() as session:
            for topic in classification_result.topics:
                # Handle both enum and string label_id
                label_id = (
                    topic.label_id.value
                    if hasattr(topic.label_id, "value")
                    else topic.label_id
                )

                # Collect keywords from this topic
                for kw in topic.keywords_in_text:
                    obs_id = str(uuid.uuid4())

                    obs = KeywordObservation(
                        obs_id=obs_id,
                        message_id=message_id,
                        label_id=label_id,
                        lemma=kw.lemma,
                        term=kw.term,
                        count=kw.count,
                        embedding_score=getattr(kw, "embedding_score", None),
                        dict_version=dict_version,
                        promoted_to_active=False,
                        observed_at=datetime.utcnow(),
                    )

                    session.add(obs)
                    obs_ids.append(obs_id)

            session.commit()

            logger.info(
                "observations_collected",
                message_id=message_id,
                dict_version=dict_version,
                topics_count=len(classification_result.topics),
                observations_count=len(obs_ids),
            )

    except Exception as e:
        logger.error(
            "observations_collection_failed",
            message_id=message_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise

    return obs_ids


def get_recent_observations(
    label_id: str | None = None, limit: int = 1000, since_days: int = 30
) -> List[dict]:
    """
    Retrieve recent observations for analysis.

    Args:
        label_id: Optional filter by label
        limit: Maximum number of observations to return
        since_days: Only return observations from last N days

    Returns:
        List of observation dictionaries

    Examples:
        >>> observations = get_recent_observations(label_id="FATTURAZIONE", limit=100)
        >>> len(observations)
        100
        >>> observations[0]["lemma"]
        'fattura'
    """
    from datetime import timedelta

    cutoff_date = datetime.utcnow() - timedelta(days=since_days)

    with get_db_session() as session:
        query = session.query(KeywordObservation).filter(
            KeywordObservation.observed_at >= cutoff_date
        )

        if label_id:
            query = query.filter(KeywordObservation.label_id == label_id)

        observations = query.order_by(KeywordObservation.observed_at.desc()).limit(limit).all()

        results = [
            {
                "obs_id": obs.obs_id,
                "message_id": obs.message_id,
                "label_id": obs.label_id,
                "lemma": obs.lemma,
                "term": obs.term,
                "count": obs.count,
                "embedding_score": obs.embedding_score,
                "dict_version": obs.dict_version,
                "observed_at": obs.observed_at.isoformat(),
            }
            for obs in observations
        ]

        logger.debug(
            "observations_retrieved",
            label_id=label_id,
            count=len(results),
            since_days=since_days,
        )

        return results


def mark_observations_promoted(obs_ids: List[str]) -> int:
    """
    Mark observations as promoted to active.

    Updates the promoted_to_active flag for given observation IDs.

    Args:
        obs_ids: List of observation IDs to mark

    Returns:
        Number of observations updated

    Examples:
        >>> updated = mark_observations_promoted(["obs1", "obs2", "obs3"])
        >>> updated
        3
    """
    with get_db_session() as session:
        updated_count = (
            session.query(KeywordObservation)
            .filter(KeywordObservation.obs_id.in_(obs_ids))
            .update({"promoted_to_active": True}, synchronize_session=False)
        )

        session.commit()

        logger.info("observations_marked_promoted", count=updated_count)

        return updated_count
