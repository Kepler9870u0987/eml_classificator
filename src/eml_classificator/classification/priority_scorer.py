"""
Rule-based priority scoring for email triage.

Implements parametric priority scoring as specified in brainstorming v2 section 4.2.

Priority determined by weighted features:
- Urgent term frequency ("urgente", "bloccante", etc.)
- High priority terms ("problema", "errore")
- Sentiment (negative → priority boost)
- Customer status (new → priority boost)
- Deadline signals (date extraction)
- VIP status (from CRM)

Produces raw score that is bucketed into: low | medium | high | urgent
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import structlog

from eml_classificator.classification.schemas import Priority, PriorityValue, SentimentValue, CustomerStatusValue
from eml_classificator.config import settings


logger = structlog.get_logger(__name__)


# ============================================================================
# TERM LISTS
# ============================================================================

# Italian urgent terms
URGENT_TERMS = [
    "urgente",
    "urgenza",
    "immediato",
    "immediatamente",
    "bloccante",
    "bloccato",
    "fermo",
    "disdetta",
    "disdire",
    "rimborso",
    "reclamo",
    "critico",
    "grave",
    "sla",
    "scaduto",
    "avvocato",
    "legale",
    "diffida",
    "danni",
]

# Italian high priority terms
HIGH_TERMS = [
    "problema",
    "errore",
    "guasto",
    "non funziona",
    "funzionante",
    "assistenza",
    "supporto",
    "aiuto",
    "necessario",
    "necessaria",
    "importante",
    "urgenza",
]


# ============================================================================
# DATE/DEADLINE EXTRACTION
# ============================================================================

def extract_deadline_signals(text: str) -> tuple[int, List[str]]:
    """
    Extract deadline signals from text.

    Looks for:
    - "entro il DD/MM" or "entro DD giorni"
    - "scadenza DD/MM" or "scadenza: YYYY-MM-DD"
    - Relative dates ("entro domani", "entro oggi")

    Args:
        text: Email body canonical

    Returns:
        (urgency_boost, matched_patterns) tuple
        - urgency_boost: 0-3 based on deadline imminence
        - matched_patterns: List of matched deadline phrases
    """
    text_lower = text.lower()
    matched_patterns = []
    urgency_boost = 0

    # Immediate deadlines (today, tomorrow)
    immediate_patterns = [
        r"(?i)\bentro oggi\b",
        r"(?i)\bentro domani\b",
        r"(?i)\bentro stasera\b",
        r"(?i)\bal più presto\b",
    ]

    for pattern in immediate_patterns:
        if re.search(pattern, text):
            matched_patterns.append(pattern)
            urgency_boost = max(urgency_boost, 3)  # Max boost

    # Near-term deadlines (within days)
    near_term_patterns = [
        r"(?i)\bentro\s+\d{1,2}\s+giorni\b",
        r"(?i)\bentro.*questa settimana\b",
        r"(?i)\bscadenza\s*:\s*\d{4}-\d{2}-\d{2}\b",
        r"(?i)\bentro il\s+\d{1,2}[/\-]\d{1,2}\b",
    ]

    for pattern in near_term_patterns:
        if re.search(pattern, text):
            matched_patterns.append(pattern)
            urgency_boost = max(urgency_boost, 2)  # Medium boost

    # General deadline mention
    generic_patterns = [
        r"(?i)\bscadenza\b",
        r"(?i)\btermine\b.*\b(ultimo|finale)\b",
        r"(?i)\btempo\b.*\b(limitato|scaduto)\b",
    ]

    for pattern in generic_patterns:
        if re.search(pattern, text) and urgency_boost == 0:
            matched_patterns.append(pattern)
            urgency_boost = 1  # Small boost

    return urgency_boost, matched_patterns


# ============================================================================
# PRIORITY SCORER
# ============================================================================

@dataclass
class PriorityWeights:
    """
    Configurable weights for priority scoring.

    Can be learned from historical data or manually tuned.
    """
    urgent_terms: float = 3.0
    high_terms: float = 1.5
    sentiment_negative: float = 2.0
    customer_new: float = 1.0
    deadline: float = 2.0
    vip: float = 2.5

    @classmethod
    def from_config(cls) -> "PriorityWeights":
        """Load weights from settings."""
        return cls(
            urgent_terms=settings.priority_weight_urgent_terms,
            high_terms=settings.priority_weight_high_terms,
            sentiment_negative=settings.priority_weight_sentiment_negative,
            customer_new=settings.priority_weight_customer_new,
            deadline=settings.priority_weight_deadline,
            vip=settings.priority_weight_vip,
        )


@dataclass
class PriorityThresholds:
    """
    Thresholds for bucketing raw score into priority levels.
    """
    urgent: float = 7.0
    high: float = 4.0
    medium: float = 2.0

    @classmethod
    def from_config(cls) -> "PriorityThresholds":
        """Load thresholds from settings."""
        return cls(
            urgent=settings.priority_threshold_urgent,
            high=settings.priority_threshold_high,
            medium=settings.priority_threshold_medium,
        )


class PriorityScorer:
    """
    Rule-based priority scorer with configurable weights.

    Computes raw score from multiple features and buckets into priority levels.
    """

    def __init__(
        self,
        weights: Optional[PriorityWeights] = None,
        thresholds: Optional[PriorityThresholds] = None
    ):
        self.weights = weights or PriorityWeights.from_config()
        self.thresholds = thresholds or PriorityThresholds.from_config()

        self.logger = logger.bind(component="priority_scorer")

    def score(
        self,
        subject: str,
        body_canonical: str,
        sentiment_value: SentimentValue,
        customer_value: CustomerStatusValue,
        vip_status: bool = False
    ) -> Priority:
        """
        Compute priority score and bucket.

        Args:
            subject: Email subject
            body_canonical: Email body canonicalized
            sentiment_value: Sentiment classification
            customer_value: Customer status
            vip_status: Whether customer is VIP (from CRM)

        Returns:
            Priority with value, confidence, signals, raw_score
        """
        text = (subject + "\n" + body_canonical).lower()
        score = 0.0
        signals = []

        # Feature 1: Urgent terms
        urgent_count = sum(1 for term in URGENT_TERMS if term in text)
        if urgent_count > 0:
            score += self.weights.urgent_terms * urgent_count
            signals.append(f"urgent_keywords:{urgent_count}")

        # Feature 2: High priority terms
        high_count = sum(1 for term in HIGH_TERMS if term in text)
        if high_count > 0:
            score += self.weights.high_terms * high_count
            signals.append(f"high_keywords:{high_count}")

        # Feature 3: Sentiment
        if sentiment_value == SentimentValue.NEGATIVE:
            score += self.weights.sentiment_negative
            signals.append("negative_sentiment")

        # Feature 4: Customer status
        if customer_value == CustomerStatusValue.NEW:
            score += self.weights.customer_new
            signals.append("new_customer")

        # Feature 5: Deadline signals
        deadline_boost, deadline_patterns = extract_deadline_signals(text)
        if deadline_boost > 0:
            score += self.weights.deadline * deadline_boost
            signals.append("deadline_mentioned")

        # Feature 6: VIP customer
        if vip_status:
            score += self.weights.vip
            signals.append("vip_customer")

        # Bucket score into priority level
        priority_val, confidence = self._bucket_score(score)

        self.logger.debug(
            "priority_scored",
            raw_score=score,
            priority=priority_val,
            signals_count=len(signals)
        )

        return Priority(
            value=priority_val,
            confidence=confidence,
            signals=signals,
            raw_score=score
        )

    def _bucket_score(self, score: float) -> tuple[PriorityValue, float]:
        """
        Bucket raw score into priority level with confidence.

        Args:
            score: Raw priority score

        Returns:
            (priority_value, confidence) tuple
        """
        if score >= self.thresholds.urgent:
            return PriorityValue.URGENT, 0.95

        if score >= self.thresholds.high:
            return PriorityValue.HIGH, 0.85

        if score >= self.thresholds.medium:
            return PriorityValue.MEDIUM, 0.75

        return PriorityValue.LOW, 0.70

    def calibrate_from_data(self, training_data: List[Dict]):
        """
        Calibrate weights from historical training data.

        This is a placeholder for future implementation using logistic regression
        or similar optimization to learn weights from labeled examples.

        Args:
            training_data: List of {features, priority_true} dicts
        """
        # TODO: Implement weight learning
        # Could use sklearn's LogisticRegression or similar
        self.logger.warning("calibrate_from_data not yet implemented")
        pass


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def compute_priority(
    subject: str,
    body_canonical: str,
    sentiment_value: SentimentValue,
    customer_value: CustomerStatusValue,
    vip_status: bool = False,
    scorer: Optional[PriorityScorer] = None
) -> Priority:
    """
    Compute priority using default or provided scorer.

    Args:
        subject: Email subject
        body_canonical: Email body
        sentiment_value: Sentiment classification
        customer_value: Customer status
        vip_status: VIP flag
        scorer: Optional custom scorer (default: from config)

    Returns:
        Priority result
    """
    if scorer is None:
        scorer = PriorityScorer()

    return scorer.score(
        subject=subject,
        body_canonical=body_canonical,
        sentiment_value=sentiment_value,
        customer_value=customer_value,
        vip_status=vip_status
    )


def override_priority_with_rules(
    llm_priority: Priority,
    subject: str,
    body_canonical: str,
    sentiment_value: SentimentValue,
    customer_value: CustomerStatusValue,
    vip_status: bool = False
) -> Priority:
    """
    Override LLM priority guess with rule-based scoring.

    This ensures priority is deterministic and based on objective signals.

    Args:
        llm_priority: Priority guessed by LLM (used as fallback if rules give low confidence)
        subject, body_canonical, sentiment_value, customer_value, vip_status: Input features

    Returns:
        Rule-based Priority (overrides LLM)
    """
    # Log LLM guess for comparison
    logger.debug(
        "overriding_llm_priority",
        llm_value=llm_priority.value,
        llm_confidence=llm_priority.confidence
    )

    # Compute rule-based priority
    rule_priority = compute_priority(
        subject=subject,
        body_canonical=body_canonical,
        sentiment_value=sentiment_value,
        customer_value=customer_value,
        vip_status=vip_status
    )

    # Log override if different
    if rule_priority.value != llm_priority.value:
        logger.info(
            "priority_overridden",
            llm_value=llm_priority.value,
            rule_value=rule_priority.value,
            raw_score=rule_priority.raw_score,
            signals=rule_priority.signals
        )

    return rule_priority
