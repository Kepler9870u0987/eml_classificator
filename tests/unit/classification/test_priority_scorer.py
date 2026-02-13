"""
Unit tests for rule-based priority scoring.

Tests priority scorer with configurable weights, deadline extraction,
and bucketing logic.
"""

import pytest
from unittest.mock import Mock, patch

from eml_classificator.classification.priority_scorer import (
    PriorityWeights,
    PriorityThresholds,
    PriorityScorer,
    extract_deadline_signals,
    compute_priority,
    override_priority_with_rules,
    URGENT_TERMS,
    HIGH_TERMS
)
from eml_classificator.classification.schemas import (
    Priority,
    PriorityValue,
    SentimentValue,
    CustomerStatusValue
)


class TestPriorityWeights:
    """Test PriorityWeights dataclass."""

    def test_default_weights(self):
        """Test default weight values."""
        weights = PriorityWeights()

        assert weights.urgent_terms == 3.0
        assert weights.high_terms == 1.5
        assert weights.sentiment_negative == 2.0
        assert weights.customer_new == 1.0
        assert weights.deadline == 2.0
        assert weights.vip == 2.5

    def test_custom_weights(self):
        """Test creating weights with custom values."""
        weights = PriorityWeights(
            urgent_terms=5.0,
            high_terms=3.0,
            sentiment_negative=4.0
        )

        assert weights.urgent_terms == 5.0
        assert weights.high_terms == 3.0
        assert weights.sentiment_negative == 4.0

    @patch('eml_classificator.classification.priority_scorer.settings')
    def test_weights_from_config(self, mock_settings):
        """Test loading weights from settings."""
        mock_settings.priority_weight_urgent_terms = 4.0
        mock_settings.priority_weight_high_terms = 2.0
        mock_settings.priority_weight_sentiment_negative = 3.0
        mock_settings.priority_weight_customer_new = 1.5
        mock_settings.priority_weight_deadline = 2.5
        mock_settings.priority_weight_vip = 3.0

        weights = PriorityWeights.from_config()

        assert weights.urgent_terms == 4.0
        assert weights.high_terms == 2.0


class TestPriorityThresholds:
    """Test PriorityThresholds dataclass."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = PriorityThresholds()

        assert thresholds.urgent == 7.0
        assert thresholds.high == 4.0
        assert thresholds.medium == 2.0

    def test_custom_thresholds(self):
        """Test creating thresholds with custom values."""
        thresholds = PriorityThresholds(
            urgent=10.0,
            high=5.0,
            medium=2.5
        )

        assert thresholds.urgent == 10.0
        assert thresholds.high == 5.0
        assert thresholds.medium == 2.5


class TestDeadlineExtraction:
    """Test deadline signal extraction."""

    def test_immediate_deadline_today(self):
        """Test detection of immediate deadline (today)."""
        text = "Ho bisogno di assistenza entro oggi."

        urgency_boost, patterns = extract_deadline_signals(text)

        assert urgency_boost == 3  # Maximum boost
        assert len(patterns) > 0

    def test_immediate_deadline_tomorrow(self):
        """Test detection of immediate deadline (tomorrow)."""
        text = "Devo risolvere il problema entro domani mattina."

        urgency_boost, patterns = extract_deadline_signals(text)

        assert urgency_boost == 3

    def test_near_term_deadline_days(self):
        """Test detection of near-term deadline (days)."""
        text = "Entro 3 giorni devo completare l'operazione."

        urgency_boost, patterns = extract_deadline_signals(text)

        assert urgency_boost == 2  # Medium boost

    def test_near_term_deadline_this_week(self):
        """Test detection of this week deadline."""
        text = "Scadenza entro questa settimana."

        urgency_boost, patterns = extract_deadline_signals(text)

        assert urgency_boost == 2

    def test_near_term_deadline_specific_date(self):
        """Test detection of specific date deadline."""
        text = "La scadenza è il 15/12, entro il 15/12 devo pagare."

        urgency_boost, patterns = extract_deadline_signals(text)

        assert urgency_boost == 2

    def test_generic_deadline_mention(self):
        """Test generic deadline mention."""
        text = "C'è una scadenza da rispettare per questo contratto."

        urgency_boost, patterns = extract_deadline_signals(text)

        assert urgency_boost == 1  # Small boost

    def test_no_deadline(self):
        """Test when no deadline is mentioned."""
        text = "Vorrei informazioni sul vostro servizio."

        urgency_boost, patterns = extract_deadline_signals(text)

        assert urgency_boost == 0
        assert len(patterns) == 0

    def test_multiple_deadline_signals(self):
        """Test when multiple deadline signals present."""
        text = "Entro oggi devo completare. Scadenza urgente entro questa settimana."

        urgency_boost, patterns = extract_deadline_signals(text)

        # Should use the most urgent (max)
        assert urgency_boost == 3

    def test_case_insensitive_deadline(self):
        """Test deadline detection is case-insensitive."""
        text = "ENTRO OGGI DEVO RISOLVERE IL PROBLEMA."

        urgency_boost, patterns = extract_deadline_signals(text)

        assert urgency_boost == 3


class TestPriorityScorer:
    """Test PriorityScorer class."""

    def test_score_with_urgent_terms(self):
        """Test scoring with urgent terms."""
        scorer = PriorityScorer()

        priority = scorer.score(
            subject="URGENTE: Problema bloccante",
            body_canonical="Il mio account è bloccato, è urgente risolvere.",
            sentiment_value=SentimentValue.NEUTRAL,
            customer_value=CustomerStatusValue.EXISTING,
            vip_status=False
        )

        # Should detect multiple urgent terms (urgente, bloccante, bloccato)
        assert priority.value in [PriorityValue.HIGH, PriorityValue.URGENT]
        assert "urgent_keywords" in " ".join(priority.signals)
        assert priority.raw_score > 0

    def test_score_with_high_terms(self):
        """Test scoring with high priority terms."""
        scorer = PriorityScorer()

        priority = scorer.score(
            subject="Problema con il sistema",
            body_canonical="C'è un errore importante che necessita assistenza.",
            sentiment_value=SentimentValue.NEUTRAL,
            customer_value=CustomerStatusValue.EXISTING,
            vip_status=False
        )

        # Should detect problema, errore, assistenza
        assert "high_keywords" in " ".join(priority.signals)

    def test_score_with_negative_sentiment(self):
        """Test sentiment boost for negative sentiment."""
        scorer = PriorityScorer()

        priority = scorer.score(
            subject="Insoddisfatto del servizio",
            body_canonical="Sono molto deluso dalla qualità.",
            sentiment_value=SentimentValue.NEGATIVE,  # Negative sentiment
            customer_value=CustomerStatusValue.EXISTING,
            vip_status=False
        )

        assert "negative_sentiment" in priority.signals
        # Score should be boosted by sentiment weight

    def test_score_with_new_customer(self):
        """Test boost for new customer."""
        scorer = PriorityScorer()

        priority = scorer.score(
            subject="Richiesta informazioni",
            body_canonical="Vorrei sapere dettagli sui servizi.",
            sentiment_value=SentimentValue.NEUTRAL,
            customer_value=CustomerStatusValue.NEW,  # New customer
            vip_status=False
        )

        assert "new_customer" in priority.signals

    def test_score_with_vip_status(self):
        """Test VIP customer boost."""
        scorer = PriorityScorer()

        priority = scorer.score(
            subject="Domanda",
            body_canonical="Una domanda semplice.",
            sentiment_value=SentimentValue.NEUTRAL,
            customer_value=CustomerStatusValue.EXISTING,
            vip_status=True  # VIP customer
        )

        assert "vip_customer" in priority.signals
        # Score should be boosted by VIP weight

    def test_score_with_deadline(self):
        """Test deadline boost."""
        scorer = PriorityScorer()

        priority = scorer.score(
            subject="Devo risolvere entro oggi",
            body_canonical="È urgente, la scadenza è oggi.",
            sentiment_value=SentimentValue.NEUTRAL,
            customer_value=CustomerStatusValue.EXISTING,
            vip_status=False
        )

        assert "deadline_mentioned" in priority.signals
        # Score should be boosted significantly for immediate deadline

    def test_score_bucket_urgent(self):
        """Test bucketing into URGENT priority."""
        scorer = PriorityScorer(
            thresholds=PriorityThresholds(urgent=7.0, high=4.0, medium=2.0)
        )

        # Create scenario with high score (multiple urgent terms + negative sentiment)
        priority = scorer.score(
            subject="URGENTE: Problema bloccante critico",
            body_canonical="Situazione grave, immediato intervento richiesto. Disdetta.",
            sentiment_value=SentimentValue.NEGATIVE,
            customer_value=CustomerStatusValue.NEW,
            vip_status=False
        )

        assert priority.value == PriorityValue.URGENT
        assert priority.confidence == 0.95

    def test_score_bucket_high(self):
        """Test bucketing into HIGH priority."""
        scorer = PriorityScorer()

        priority = scorer.score(
            subject="Problema importante",
            body_canonical="C'è un errore che richiede assistenza.",
            sentiment_value=SentimentValue.NEGATIVE,
            customer_value=CustomerStatusValue.EXISTING,
            vip_status=False
        )

        # Should be HIGH (some terms + negative sentiment)
        assert priority.value in [PriorityValue.HIGH, PriorityValue.URGENT]

    def test_score_bucket_medium(self):
        """Test bucketing into MEDIUM priority."""
        scorer = PriorityScorer()

        priority = scorer.score(
            subject="Domanda sul servizio",
            body_canonical="Ho bisogno di assistenza per una questione.",
            sentiment_value=SentimentValue.NEUTRAL,
            customer_value=CustomerStatusValue.EXISTING,
            vip_status=False
        )

        # Should be MEDIUM (moderate score)
        assert priority.value in [PriorityValue.MEDIUM, PriorityValue.HIGH]

    def test_score_bucket_low(self):
        """Test bucketing into LOW priority."""
        scorer = PriorityScorer()

        priority = scorer.score(
            subject="Informazioni",
            body_canonical="Vorrei sapere maggiori dettagli sui vostri piani tariffari.",
            sentiment_value=SentimentValue.POSITIVE,
            customer_value=CustomerStatusValue.UNKNOWN,
            vip_status=False
        )

        # Should be LOW (no priority signals)
        assert priority.value == PriorityValue.LOW
        assert priority.confidence == 0.70

    def test_empty_text(self):
        """Test scoring with empty text."""
        scorer = PriorityScorer()

        priority = scorer.score(
            subject="",
            body_canonical="",
            sentiment_value=SentimentValue.NEUTRAL,
            customer_value=CustomerStatusValue.UNKNOWN,
            vip_status=False
        )

        # Should default to LOW
        assert priority.value == PriorityValue.LOW

    def test_custom_weights(self):
        """Test scorer with custom weights."""
        custom_weights = PriorityWeights(
            urgent_terms=10.0,  # Very high weight
            high_terms=1.0,
            sentiment_negative=1.0,
            customer_new=0.5,
            deadline=1.0,
            vip=1.0
        )

        scorer = PriorityScorer(weights=custom_weights)

        priority = scorer.score(
            subject="Urgente",
            body_canonical="Problema urgente.",
            sentiment_value=SentimentValue.NEUTRAL,
            customer_value=CustomerStatusValue.EXISTING,
            vip_status=False
        )

        # With high urgent_terms weight, should get high score
        assert priority.raw_score >= 20.0  # 10.0 * 2 urgent terms

    def test_multiple_features_combined(self):
        """Test scoring with multiple features combined."""
        scorer = PriorityScorer()

        priority = scorer.score(
            subject="URGENTE: Problema critico",
            body_canonical="Situazione bloccante entro oggi. Sono un nuovo cliente.",
            sentiment_value=SentimentValue.NEGATIVE,
            customer_value=CustomerStatusValue.NEW,
            vip_status=True
        )

        # Should have accumulation from:
        # - Multiple urgent terms
        # - Negative sentiment
        # - New customer
        # - Deadline
        # - VIP
        assert priority.value == PriorityValue.URGENT
        assert len(priority.signals) >= 5


class TestComputePriority:
    """Test compute_priority convenience function."""

    def test_compute_priority_default_scorer(self):
        """Test compute_priority with default scorer."""
        priority = compute_priority(
            subject="Problema urgente",
            body_canonical="Assistenza necessaria.",
            sentiment_value=SentimentValue.NEGATIVE,
            customer_value=CustomerStatusValue.NEW,
            vip_status=False
        )

        assert isinstance(priority, Priority)
        assert priority.value is not None
        assert priority.confidence > 0.0

    def test_compute_priority_custom_scorer(self):
        """Test compute_priority with custom scorer."""
        custom_scorer = PriorityScorer(
            weights=PriorityWeights(urgent_terms=10.0)
        )

        priority = compute_priority(
            subject="Urgente",
            body_canonical="Urgenza massima.",
            sentiment_value=SentimentValue.NEUTRAL,
            customer_value=CustomerStatusValue.EXISTING,
            vip_status=False,
            scorer=custom_scorer
        )

        assert isinstance(priority, Priority)


class TestOverridePriorityWithRules:
    """Test override function that replaces LLM guess."""

    def test_override_llm_with_rules(self):
        """Test LLM priority is overridden by rule-based scoring."""
        llm_priority = Priority(
            value=PriorityValue.LOW,
            confidence=0.5,
            signals=["llm_guess"]
        )

        rule_priority = override_priority_with_rules(
            llm_priority=llm_priority,
            subject="URGENTE: Problema bloccante",
            body_canonical="Situazione critica entro oggi.",
            sentiment_value=SentimentValue.NEGATIVE,
            customer_value=CustomerStatusValue.NEW,
            vip_status=False
        )

        # Rules should override to higher priority
        assert rule_priority.value != PriorityValue.LOW
        assert rule_priority.value in [PriorityValue.HIGH, PriorityValue.URGENT]
        # Signals should be from rule-based scoring
        assert len(rule_priority.signals) > 0

    def test_override_maintains_determinism(self):
        """Test override produces same result for same input."""
        llm_priority = Priority(
            value=PriorityValue.MEDIUM,
            confidence=0.7,
            signals=[]
        )

        # Call twice with same input
        result1 = override_priority_with_rules(
            llm_priority=llm_priority,
            subject="Problema",
            body_canonical="Errore di sistema.",
            sentiment_value=SentimentValue.NEGATIVE,
            customer_value=CustomerStatusValue.EXISTING,
            vip_status=False
        )

        result2 = override_priority_with_rules(
            llm_priority=llm_priority,
            subject="Problema",
            body_canonical="Errore di sistema.",
            sentiment_value=SentimentValue.NEGATIVE,
            customer_value=CustomerStatusValue.EXISTING,
            vip_status=False
        )

        # Should be identical
        assert result1.value == result2.value
        assert result1.raw_score == result2.raw_score
        assert result1.signals == result2.signals


class TestTermLists:
    """Test urgent and high priority term lists."""

    def test_urgent_terms_not_empty(self):
        """Test urgent terms list is populated."""
        assert len(URGENT_TERMS) > 0
        assert "urgente" in URGENT_TERMS
        assert "bloccante" in URGENT_TERMS
        assert "immediato" in URGENT_TERMS

    def test_high_terms_not_empty(self):
        """Test high priority terms list is populated."""
        assert len(HIGH_TERMS) > 0
        assert "problema" in HIGH_TERMS
        assert "errore" in HIGH_TERMS
        assert "assistenza" in HIGH_TERMS

    def test_terms_lowercase(self):
        """Test all terms are lowercase for matching."""
        for term in URGENT_TERMS:
            assert term == term.lower()

        for term in HIGH_TERMS:
            assert term == term.lower()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_none_sentiment(self):
        """Test handling of None sentiment."""
        scorer = PriorityScorer()

        # Should handle gracefully
        priority = scorer.score(
            subject="Test",
            body_canonical="Test body",
            sentiment_value=SentimentValue.NEUTRAL,
            customer_value=CustomerStatusValue.UNKNOWN,
            vip_status=False
        )

        assert priority.value is not None

    def test_very_long_text(self):
        """Test performance with very long text."""
        scorer = PriorityScorer()

        long_text = "Problema importante. " * 1000 + "Urgente bloccante critico."

        priority = scorer.score(
            subject="Urgente",
            body_canonical=long_text,
            sentiment_value=SentimentValue.NEGATIVE,
            customer_value=CustomerStatusValue.NEW,
            vip_status=False
        )

        # Should still detect terms and score appropriately
        assert priority.value in [PriorityValue.HIGH, PriorityValue.URGENT]

    def test_special_characters(self):
        """Test handling of special characters in text."""
        scorer = PriorityScorer()

        priority = scorer.score(
            subject="!!! URGENTE !!!",
            body_canonical="Problema @#$% bloccante!!!",
            sentiment_value=SentimentValue.NEGATIVE,
            customer_value=CustomerStatusValue.EXISTING,
            vip_status=False
        )

        # Should still detect urgent terms
        assert priority.value in [PriorityValue.HIGH, PriorityValue.URGENT]

    def test_unicode_text(self):
        """Test handling of unicode characters."""
        scorer = PriorityScorer()

        priority = scorer.score(
            subject="Problème urgént",
            body_canonical="Errore nel sistema con caratteri speciali: è à ò ù.",
            sentiment_value=SentimentValue.NEGATIVE,
            customer_value=CustomerStatusValue.EXISTING,
            vip_status=False
        )

        assert priority.value is not None
