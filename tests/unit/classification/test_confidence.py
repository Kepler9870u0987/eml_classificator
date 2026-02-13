"""
Unit tests for confidence adjustment logic.

Tests 4-component weighted confidence formula, collision detection,
and confidence calibration.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from eml_classificator.classification.confidence import (
    build_collision_index,
    compute_topic_confidence_adjusted,
    adjust_all_topic_confidences,
    filter_topics_by_confidence,
    _compute_keyword_quality_score,
    _compute_evidence_coverage_score,
    _compute_collision_penalty_score
)
from eml_classificator.classification.schemas import (
    TopicAssignment,
    TopicLabel,
    KeywordInText,
    Evidence
)


class TestCollisionIndex:
    """Test collision index building."""

    def test_build_collision_index_single_topic(self):
        """Test collision index with single topic."""
        topics = [
            TopicAssignment(
                label_id=TopicLabel.FATTURAZIONE,
                confidence=0.9,
                keywords_in_text=[
                    KeywordInText(candidate_id="abc", lemma="fattura", count=3),
                    KeywordInText(candidate_id="def", lemma="pagamento", count=2)
                ],
                evidence=[Evidence(quote="test")]
            )
        ]

        collision_index = build_collision_index(topics)

        assert "fattura" in collision_index
        assert "pagamento" in collision_index
        assert TopicLabel.FATTURAZIONE.value in collision_index["fattura"]
        assert len(collision_index["fattura"]) == 1  # No collision

    def test_build_collision_index_multiple_topics(self):
        """Test collision index with multiple topics sharing keywords."""
        topics = [
            TopicAssignment(
                label_id=TopicLabel.FATTURAZIONE,
                confidence=0.9,
                keywords_in_text=[
                    KeywordInText(candidate_id="abc", lemma="fattura", count=3),
                    KeywordInText(candidate_id="def", lemma="problema", count=1)  # Shared
                ],
                evidence=[Evidence(quote="test")]
            ),
            TopicAssignment(
                label_id=TopicLabel.RECLAMO,
                confidence=0.8,
                keywords_in_text=[
                    KeywordInText(candidate_id="ghi", lemma="problema", count=2),  # Shared
                    KeywordInText(candidate_id="jkl", lemma="reclamo", count=1)
                ],
                evidence=[Evidence(quote="test")]
            )
        ]

        collision_index = build_collision_index(topics)

        # "problema" appears in both topics
        assert len(collision_index["problema"]) == 2
        assert TopicLabel.FATTURAZIONE.value in collision_index["problema"]
        assert TopicLabel.RECLAMO.value in collision_index["problema"]

        # "fattura" and "reclamo" are unique
        assert len(collision_index["fattura"]) == 1
        assert len(collision_index["reclamo"]) == 1

    def test_build_collision_index_with_external_lexicon(self):
        """Test collision index merges with external lexicon."""
        topics = [
            TopicAssignment(
                label_id=TopicLabel.FATTURAZIONE,
                confidence=0.9,
                keywords_in_text=[
                    KeywordInText(candidate_id="abc", lemma="fattura", count=3)
                ],
                evidence=[Evidence(quote="test")]
            )
        ]

        external_lexicon = {
            "fattura": {TopicLabel.DOCUMENTI.value},  # Also in DOCUMENTI
            "contratto": {TopicLabel.CONTRATTO.value}
        }

        collision_index = build_collision_index(topics, external_lexicon)

        # "fattura" now appears in both FATTURAZIONE and DOCUMENTI
        assert len(collision_index["fattura"]) == 2
        assert TopicLabel.FATTURAZIONE.value in collision_index["fattura"]
        assert TopicLabel.DOCUMENTI.value in collision_index["fattura"]

        # "contratto" only from external lexicon
        assert len(collision_index["contratto"]) == 1

    def test_build_collision_index_empty(self):
        """Test collision index with no topics."""
        topics = []

        collision_index = build_collision_index(topics)

        assert len(collision_index) == 0


class TestKeywordQualityScore:
    """Test keyword quality scoring component."""

    def test_keyword_quality_with_good_candidates(self):
        """Test keyword quality with high composite scores."""
        topic = TopicAssignment(
            label_id=TopicLabel.FATTURAZIONE,
            confidence=0.9,
            keywords_in_text=[
                KeywordInText(candidate_id="abc", lemma="fattura", count=3),
                KeywordInText(candidate_id="def", lemma="pagamento", count=2)
            ],
            evidence=[Evidence(quote="test")]
        )

        candidates = [
            {"candidate_id": "abc", "lemma": "fattura", "composite_score": 0.95},
            {"candidate_id": "def", "lemma": "pagamento", "composite_score": 0.85}
        ]

        score = _compute_keyword_quality_score(topic, candidates)

        # Average of 0.95 and 0.85
        assert score == pytest.approx(0.90, abs=0.01)

    def test_keyword_quality_with_poor_candidates(self):
        """Test keyword quality with low composite scores."""
        topic = TopicAssignment(
            label_id=TopicLabel.FATTURAZIONE,
            confidence=0.9,
            keywords_in_text=[
                KeywordInText(candidate_id="abc", lemma="weak", count=1)
            ],
            evidence=[Evidence(quote="test")]
        )

        candidates = [
            {"candidate_id": "abc", "lemma": "weak", "composite_score": 0.2}
        ]

        score = _compute_keyword_quality_score(topic, candidates)

        assert score == pytest.approx(0.2, abs=0.01)

    def test_keyword_quality_no_keywords(self):
        """Test keyword quality with no keywords."""
        # Use model_construct to bypass validation for edge case testing
        topic = TopicAssignment.model_construct(
            label_id=TopicLabel.UNKNOWN_TOPIC,
            confidence=0.5,
            keywords_in_text=[],  # Empty
            evidence=[Evidence(quote="test")]
        )

        candidates = []

        score = _compute_keyword_quality_score(topic, candidates)

        assert score == 0.1  # Low score for no keywords

    def test_keyword_quality_missing_candidate(self):
        """Test keyword quality when candidate not in candidate list."""
        topic = TopicAssignment(
            label_id=TopicLabel.FATTURAZIONE,
            confidence=0.9,
            keywords_in_text=[
                KeywordInText(candidate_id="missing", lemma="unknown", count=1)
            ],
            evidence=[Evidence(quote="test")]
        )

        candidates = [
            {"candidate_id": "abc", "lemma": "other", "composite_score": 0.9}
        ]

        score = _compute_keyword_quality_score(topic, candidates)

        # Should get 0.0 for missing candidate
        assert score == 0.0


class TestEvidenceCoverageScore:
    """Test evidence coverage scoring component."""

    def test_evidence_coverage_no_evidence(self):
        """Test evidence coverage with no evidence."""
        # Use model_construct to bypass validation for edge case testing
        topic = TopicAssignment.model_construct(
            label_id=TopicLabel.FATTURAZIONE,
            confidence=0.9,
            keywords_in_text=[KeywordInText(candidate_id="abc", lemma="x", count=1)],
            evidence=[]  # Empty
        )

        score = _compute_evidence_coverage_score(topic)

        assert score == 0.2  # Low score for no evidence

    def test_evidence_coverage_one_evidence(self):
        """Test evidence coverage with one evidence."""
        topic = TopicAssignment(
            label_id=TopicLabel.FATTURAZIONE,
            confidence=0.9,
            keywords_in_text=[KeywordInText(candidate_id="abc", lemma="x", count=1)],
            evidence=[Evidence(quote="ho ricevuto la fattura")]
        )

        score = _compute_evidence_coverage_score(topic)

        # 1 / 2.0 = 0.5
        assert score == pytest.approx(0.5, abs=0.01)

    def test_evidence_coverage_two_evidence(self):
        """Test evidence coverage with two evidence (full score)."""
        topic = TopicAssignment(
            label_id=TopicLabel.FATTURAZIONE,
            confidence=0.9,
            keywords_in_text=[KeywordInText(candidate_id="abc", lemma="x", count=1)],
            evidence=[
                Evidence(quote="fattura errata"),
                Evidence(quote="ho pagato ma non risulta")
            ]
        )

        score = _compute_evidence_coverage_score(topic)

        # 2 / 2.0 = 1.0
        assert score == 1.0

    def test_evidence_coverage_more_than_two(self):
        """Test evidence coverage caps at 1.0 for more than 2 evidence."""
        # Use model_construct to bypass validation for edge case testing
        topic = TopicAssignment.model_construct(
            label_id=TopicLabel.FATTURAZIONE,
            confidence=0.9,
            keywords_in_text=[KeywordInText(candidate_id="abc", lemma="x", count=1)],
            evidence=[
                Evidence(quote="evidence 1"),
                Evidence(quote="evidence 2"),
                Evidence(quote="evidence 3")
            ]
        )

        score = _compute_evidence_coverage_score(topic)

        # Should cap at 1.0
        assert score == 1.0


class TestCollisionPenaltyScore:
    """Test collision penalty scoring component."""

    def test_collision_penalty_no_collision(self):
        """Test collision penalty with unique keywords."""
        topic = TopicAssignment(
            label_id=TopicLabel.FATTURAZIONE,
            confidence=0.9,
            keywords_in_text=[
                KeywordInText(candidate_id="abc", lemma="fattura", count=3),
                KeywordInText(candidate_id="def", lemma="pagamento", count=2)
            ],
            evidence=[Evidence(quote="test")]
        )

        collision_index = {
            "fattura": {TopicLabel.FATTURAZIONE.value},
            "pagamento": {TopicLabel.FATTURAZIONE.value}
        }

        score = _compute_collision_penalty_score(topic, collision_index)

        # No collision - full score
        assert score == 1.0

    def test_collision_penalty_with_collision(self):
        """Test collision penalty when keywords are ambiguous."""
        topic = TopicAssignment(
            label_id=TopicLabel.FATTURAZIONE,
            confidence=0.9,
            keywords_in_text=[
                KeywordInText(candidate_id="abc", lemma="problema", count=2)  # Ambiguous
            ],
            evidence=[Evidence(quote="test")]
        )

        collision_index = {
            "problema": {
                TopicLabel.FATTURAZIONE.value,
                TopicLabel.RECLAMO.value,
                TopicLabel.ASSISTENZA_TECNICA.value
            }  # Appears in 3 topics
        }

        score = _compute_collision_penalty_score(topic, collision_index)

        # Penalty: 1/3 = 0.333
        assert score == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_collision_penalty_mixed(self):
        """Test collision penalty with mix of unique and ambiguous keywords."""
        topic = TopicAssignment(
            label_id=TopicLabel.FATTURAZIONE,
            confidence=0.9,
            keywords_in_text=[
                KeywordInText(candidate_id="abc", lemma="fattura", count=3),  # Unique
                KeywordInText(candidate_id="def", lemma="problema", count=1)  # Ambiguous
            ],
            evidence=[Evidence(quote="test")]
        )

        collision_index = {
            "fattura": {TopicLabel.FATTURAZIONE.value},  # No collision
            "problema": {TopicLabel.FATTURAZIONE.value, TopicLabel.RECLAMO.value}  # Collision
        }

        score = _compute_collision_penalty_score(topic, collision_index)

        # Average: (1.0 + 0.5) / 2 = 0.75
        assert score == pytest.approx(0.75, abs=0.01)

    def test_collision_penalty_no_keywords(self):
        """Test collision penalty with no keywords."""
        # Use model_construct to bypass validation for edge case testing
        topic = TopicAssignment.model_construct(
            label_id=TopicLabel.UNKNOWN_TOPIC,
            confidence=0.5,
            keywords_in_text=[],
            evidence=[Evidence(quote="test")]
        )

        collision_index = {}

        score = _compute_collision_penalty_score(topic, collision_index)

        # Neutral score for no keywords
        assert score == 0.5


class TestTopicConfidenceAdjusted:
    """Test complete confidence adjustment calculation."""

    def test_confidence_adjustment_high_quality(self):
        """Test confidence adjustment with high-quality topic."""
        topic = TopicAssignment(
            label_id=TopicLabel.FATTURAZIONE,
            confidence=0.9,  # LLM confidence
            keywords_in_text=[
                KeywordInText(candidate_id="abc", lemma="fattura", count=3),
                KeywordInText(candidate_id="def", lemma="pagamento", count=2)
            ],
            evidence=[
                Evidence(quote="ho ricevuto la fattura"),
                Evidence(quote="il pagamento non risulta")
            ]
        )

        candidates = [
            {"candidate_id": "abc", "composite_score": 0.95},
            {"candidate_id": "def", "composite_score": 0.90}
        ]

        collision_index = {
            "fattura": {TopicLabel.FATTURAZIONE.value},
            "pagamento": {TopicLabel.FATTURAZIONE.value}
        }

        adjusted = compute_topic_confidence_adjusted(
            topic=topic,
            candidates=candidates,
            collision_index=collision_index,
            llm_confidence=0.9
        )

        # Should be high (good keywords, 2 evidence, no collision)
        assert adjusted > 0.85
        assert adjusted <= 1.0

    def test_confidence_adjustment_low_quality(self):
        """Test confidence adjustment with low-quality topic."""
        topic = TopicAssignment(
            label_id=TopicLabel.UNKNOWN_TOPIC,
            confidence=0.4,  # Low LLM confidence
            keywords_in_text=[
                KeywordInText(candidate_id="abc", lemma="weak", count=1)
            ],
            evidence=[Evidence(quote="short")]  # Only 1 evidence
        )

        candidates = [
            {"candidate_id": "abc", "composite_score": 0.3}  # Low quality
        ]

        collision_index = {
            "weak": {
                TopicLabel.UNKNOWN_TOPIC.value,
                TopicLabel.FATTURAZIONE.value,
                TopicLabel.RECLAMO.value
            }  # High collision
        }

        adjusted = compute_topic_confidence_adjusted(
            topic=topic,
            candidates=candidates,
            collision_index=collision_index,
            llm_confidence=0.4
        )

        # Should be lower than original
        assert adjusted < 0.6

    def test_confidence_adjustment_custom_weights(self):
        """Test confidence adjustment with custom weights."""
        topic = TopicAssignment(
            label_id=TopicLabel.FATTURAZIONE,
            confidence=0.8,
            keywords_in_text=[KeywordInText(candidate_id="abc", lemma="x", count=1)],
            evidence=[Evidence(quote="test")]
        )

        candidates = [{"candidate_id": "abc", "composite_score": 0.9}]
        collision_index = {"x": {TopicLabel.FATTURAZIONE.value}}

        # Custom weights (emphasize keywords)
        custom_weights = {
            "llm": 0.1,
            "keywords": 0.7,
            "evidence": 0.1,
            "collision": 0.1
        }

        adjusted = compute_topic_confidence_adjusted(
            topic=topic,
            candidates=candidates,
            collision_index=collision_index,
            llm_confidence=0.8,
            weights=custom_weights
        )

        # Should be influenced heavily by keyword score (0.9)
        assert adjusted > 0.7

    def test_confidence_adjustment_clipping(self):
        """Test confidence is clipped to [0, 1] range."""
        topic = TopicAssignment(
            label_id=TopicLabel.FATTURAZIONE,
            confidence=0.95,
            keywords_in_text=[KeywordInText(candidate_id="abc", lemma="x", count=1)],
            evidence=[Evidence(quote="test"), Evidence(quote="test2")]
        )

        candidates = [{"candidate_id": "abc", "composite_score": 1.0}]
        collision_index = {"x": {TopicLabel.FATTURAZIONE.value}}

        adjusted = compute_topic_confidence_adjusted(
            topic=topic,
            candidates=candidates,
            collision_index=collision_index,
            llm_confidence=0.95
        )

        # Should never exceed 1.0
        assert adjusted <= 1.0
        assert adjusted >= 0.0


class TestAdjustAllTopicConfidences:
    """Test bulk confidence adjustment."""

    def test_adjust_all_topics(self):
        """Test adjusting confidence for multiple topics."""
        topics = [
            TopicAssignment(
                label_id=TopicLabel.FATTURAZIONE,
                confidence=0.9,
                keywords_in_text=[KeywordInText(candidate_id="abc", lemma="fattura", count=3)],
                evidence=[Evidence(quote="test")]
            ),
            TopicAssignment(
                label_id=TopicLabel.RECLAMO,
                confidence=0.7,
                keywords_in_text=[KeywordInText(candidate_id="def", lemma="reclamo", count=2)],
                evidence=[Evidence(quote="test")]
            )
        ]

        candidates = [
            {"candidate_id": "abc", "composite_score": 0.9},
            {"candidate_id": "def", "composite_score": 0.8}
        ]

        adjusted_topics = adjust_all_topic_confidences(topics, candidates)

        assert len(adjusted_topics) == 2
        # Each topic should have confidence_adjusted field
        assert hasattr(adjusted_topics[0], 'confidence_adjusted')
        assert hasattr(adjusted_topics[1], 'confidence_adjusted')
        assert adjusted_topics[0].confidence_adjusted is not None
        assert adjusted_topics[1].confidence_adjusted is not None

    def test_adjust_all_topics_empty(self):
        """Test adjusting confidence with no topics."""
        topics = []
        candidates = []

        adjusted_topics = adjust_all_topic_confidences(topics, candidates)

        assert len(adjusted_topics) == 0


class TestFilterTopicsByConfidence:
    """Test confidence thresholding."""

    def test_filter_above_threshold(self):
        """Test filtering keeps topics above threshold."""
        topics = [
            TopicAssignment(
                label_id=TopicLabel.FATTURAZIONE,
                confidence=0.9,
                keywords_in_text=[KeywordInText(candidate_id="abc", lemma="x", count=1)],
                evidence=[Evidence(quote="test")]
            ),
            TopicAssignment(
                label_id=TopicLabel.RECLAMO,
                confidence=0.5,
                keywords_in_text=[KeywordInText(candidate_id="def", lemma="y", count=1)],
                evidence=[Evidence(quote="test")]
            ),
            TopicAssignment(
                label_id=TopicLabel.UNKNOWN_TOPIC,
                confidence=0.2,  # Below threshold
                keywords_in_text=[KeywordInText(candidate_id="ghi", lemma="z", count=1)],
                evidence=[Evidence(quote="test")]
            )
        ]

        filtered = filter_topics_by_confidence(topics, threshold=0.3, use_adjusted=False)

        assert len(filtered) == 2
        assert filtered[0].label_id == TopicLabel.FATTURAZIONE
        assert filtered[1].label_id == TopicLabel.RECLAMO

    def test_filter_all_below_threshold(self):
        """Test filtering when all topics below threshold."""
        topics = [
            TopicAssignment(
                label_id=TopicLabel.UNKNOWN_TOPIC,
                confidence=0.1,
                keywords_in_text=[KeywordInText(candidate_id="abc", lemma="x", count=1)],
                evidence=[Evidence(quote="test")]
            )
        ]

        filtered = filter_topics_by_confidence(topics, threshold=0.3, use_adjusted=False)

        # Should filter out all
        assert len(filtered) == 0

    def test_filter_use_adjusted_confidence(self):
        """Test filtering uses adjusted confidence when available."""
        topic = TopicAssignment(
            label_id=TopicLabel.FATTURAZIONE,
            confidence=0.4,  # Original below threshold
            keywords_in_text=[KeywordInText(candidate_id="abc", lemma="x", count=1)],
            evidence=[Evidence(quote="test")]
        )
        topic.confidence_adjusted = 0.8  # Adjusted above threshold

        filtered = filter_topics_by_confidence([topic], threshold=0.5, use_adjusted=True)

        assert len(filtered) == 1  # Should keep it

        filtered_orig = filter_topics_by_confidence([topic], threshold=0.5, use_adjusted=False)

        assert len(filtered_orig) == 0  # Would filter with original


class TestWeightValidation:
    """Test weight validation and normalization."""

    @patch('eml_classificator.classification.confidence.settings')
    def test_weights_sum_to_one(self, mock_settings):
        """Test weights are normalized if they don't sum to 1."""
        mock_settings.confidence_weight_llm = 0.3
        mock_settings.confidence_weight_keywords = 0.4
        mock_settings.confidence_weight_evidence = 0.2
        mock_settings.confidence_weight_collision = 0.1

        topic = TopicAssignment(
            label_id=TopicLabel.FATTURAZIONE,
            confidence=0.9,
            keywords_in_text=[KeywordInText(candidate_id="abc", lemma="x", count=1)],
            evidence=[Evidence(quote="test")]
        )

        candidates = [{"candidate_id": "abc", "composite_score": 0.9}]
        collision_index = {"x": {TopicLabel.FATTURAZIONE.value}}

        # Should work without errors
        adjusted = compute_topic_confidence_adjusted(
            topic=topic,
            candidates=candidates,
            collision_index=collision_index,
            llm_confidence=0.9
        )

        assert 0.0 <= adjusted <= 1.0

    def test_invalid_weights_normalized(self):
        """Test invalid weights (not summing to 1) are normalized."""
        topic = TopicAssignment(
            label_id=TopicLabel.FATTURAZIONE,
            confidence=0.9,
            keywords_in_text=[KeywordInText(candidate_id="abc", lemma="x", count=1)],
            evidence=[Evidence(quote="test")]
        )

        candidates = [{"candidate_id": "abc", "composite_score": 0.9}]
        collision_index = {"x": {TopicLabel.FATTURAZIONE.value}}

        # Weights that don't sum to 1
        invalid_weights = {
            "llm": 0.5,
            "keywords": 0.5,
            "evidence": 0.5,
            "collision": 0.5
        }  # Sum = 2.0

        # Should normalize and work
        adjusted = compute_topic_confidence_adjusted(
            topic=topic,
            candidates=candidates,
            collision_index=collision_index,
            llm_confidence=0.9,
            weights=invalid_weights
        )

        assert 0.0 <= adjusted <= 1.0
