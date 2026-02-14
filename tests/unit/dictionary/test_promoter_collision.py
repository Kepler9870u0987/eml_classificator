"""
Unit tests for collision detection and keyword promoter.

Tests collision detection, statistical aggregation, and promotion logic
as specified in brainstorming v2 section 5.2.
"""

import pytest

from eml_classificator.dictionary.collision import (
    detect_collisions,
    get_collision_index,
    resolve_collision_by_score,
)
from eml_classificator.dictionary.promoter import KeywordPromoter


class TestCollisionDetection:
    """Test collision detection for keywords appearing in multiple labels."""

    def test_detect_collisions_basic(self):
        """Test basic collision detection."""
        entries = [
            {"lemma": "fattura", "label_id": "FATTURAZIONE"},
            {"lemma": "fattura", "label_id": "CONTABILITA"},
            {"lemma": "fattura", "label_id": "PAGAMENTO"},
            {"lemma": "problema", "label_id": "ASSISTENZA"},
        ]

        collisions = detect_collisions(entries, threshold=2)

        assert "fattura" in collisions
        assert len(collisions["fattura"]) == 3
        assert set(collisions["fattura"]) == {"FATTURAZIONE", "CONTABILITA", "PAGAMENTO"}
        assert "problema" not in collisions  # Only in 1 label, below threshold

    def test_detect_collisions_threshold_2(self):
        """Test collision detection with threshold=2."""
        entries = [
            {"lemma": "documento", "label_id": "FATTURAZIONE"},
            {"lemma": "documento", "label_id": "CONTABILITA"},
            {"lemma": "richiesta", "label_id": "ASSISTENZA"},
        ]

        collisions = detect_collisions(entries, threshold=2)

        assert len(collisions) == 0  # "documento" in 2 labels, not > threshold

    def test_detect_collisions_threshold_1(self):
        """Test collision detection with threshold=1."""
        entries = [
            {"lemma": "documento", "label_id": "FATTURAZIONE"},
            {"lemma": "documento", "label_id": "CONTABILITA"},
        ]

        collisions = detect_collisions(entries, threshold=1)

        assert "documento" in collisions
        assert len(collisions["documento"]) == 2

    def test_detect_collisions_no_collisions(self):
        """Test collision detection with no collisions."""
        entries = [
            {"lemma": "fattura", "label_id": "FATTURAZIONE"},
            {"lemma": "problema", "label_id": "ASSISTENZA"},
            {"lemma": "pagamento", "label_id": "PAGAMENTO"},
        ]

        collisions = detect_collisions(entries, threshold=2)

        assert len(collisions) == 0

    def test_detect_collisions_missing_fields(self):
        """Test collision detection with missing fields."""
        entries = [
            {"lemma": "fattura", "label_id": "FATTURAZIONE"},
            {"lemma": None, "label_id": "CONTABILITA"},  # Missing lemma
            {"lemma": "problema"},  # Missing label_id
        ]

        collisions = detect_collisions(entries, threshold=2)

        assert len(collisions) == 0  # All entries filtered out

    def test_get_collision_index(self):
        """Test collision index building from stats."""
        stats = {
            "FATTURAZIONE": {"fattura": {"doc_freq": 5}, "pagamento": {"doc_freq": 2}},
            "CONTABILITA": {"fattura": {"doc_freq": 3}, "saldo": {"doc_freq": 4}},
            "PAGAMENTO": {"fattura": {"doc_freq": 1}},
        }

        index = get_collision_index(stats)

        assert "fattura" in index
        assert len(index["fattura"]) == 3
        assert set(index["fattura"]) == {"FATTURAZIONE", "CONTABILITA", "PAGAMENTO"}

        assert "pagamento" in index
        assert len(index["pagamento"]) == 1

        assert "saldo" in index
        assert len(index["saldo"]) == 1

    def test_resolve_collision_by_score(self):
        """Test collision resolution by embedding score."""
        stats = {
            "FATTURAZIONE": {"fattura": {"avg_embedding_score": 0.8}},
            "CONTABILITA": {"fattura": {"avg_embedding_score": 0.5}},
            "PAGAMENTO": {"fattura": {"avg_embedding_score": 0.6}},
        }

        winner = resolve_collision_by_score("fattura", ["FATTURAZIONE", "CONTABILITA", "PAGAMENTO"], stats)

        assert winner == "FATTURAZIONE"  # Highest score

    def test_resolve_collision_by_score_tie(self):
        """Test collision resolution with tied scores."""
        stats = {
            "FATTURAZIONE": {"fattura": {"avg_embedding_score": 0.7}},
            "CONTABILITA": {"fattura": {"avg_embedding_score": 0.7}},
        }

        winner = resolve_collision_by_score("fattura", ["FATTURAZIONE", "CONTABILITA"], stats)

        assert winner in ["FATTURAZIONE", "CONTABILITA"]  # Either is valid


class TestKeywordPromoter:
    """Test keyword promoter with deterministic rules."""

    def test_promoter_initialization_default(self):
        """Test promoter initialization with default config."""
        promoter = KeywordPromoter()

        assert promoter.config["regex_min_doc_freq"] > 0
        assert promoter.config["ner_min_doc_freq"] > 0
        assert promoter.config["min_total_count"] > 0

    def test_promoter_initialization_custom(self):
        """Test promoter initialization with custom config."""
        config = {
            "regex_min_doc_freq": 5,
            "regex_min_embedding_score": 0.5,
            "ner_min_doc_freq": 3,
            "ner_min_embedding_score": 0.3,
            "max_collision_labels": 3,
            "min_total_count": 10,
        }

        promoter = KeywordPromoter(config=config)

        assert promoter.config == config

    def test_compute_stats_basic(self):
        """Test statistics computation."""
        observations = [
            {
                "label_id": "FATTURAZIONE",
                "lemma": "fattura",
                "message_id": "msg1",
                "count": 2,
                "embedding_score": 0.8,
            },
            {
                "label_id": "FATTURAZIONE",
                "lemma": "fattura",
                "message_id": "msg2",
                "count": 1,
                "embedding_score": 0.75,
            },
            {
                "label_id": "FATTURAZIONE",
                "lemma": "fattura",
                "message_id": "msg3",
                "count": 3,
                "embedding_score": 0.82,
            },
        ]

        promoter = KeywordPromoter()
        stats = promoter.compute_stats(observations)

        assert "FATTURAZIONE" in stats
        assert "fattura" in stats["FATTURAZIONE"]

        fattura_stats = stats["FATTURAZIONE"]["fattura"]
        assert fattura_stats["doc_freq"] == 3  # 3 unique messages
        assert fattura_stats["total_count"] == 6  # 2 + 1 + 3
        assert 0.7 < fattura_stats["avg_embedding_score"] < 0.85  # Average of 0.8, 0.75, 0.82

    def test_compute_stats_multiple_labels(self):
        """Test statistics computation with multiple labels."""
        observations = [
            {"label_id": "FATTURAZIONE", "lemma": "fattura", "message_id": "msg1", "count": 2, "embedding_score": 0.8},
            {"label_id": "FATTURAZIONE", "lemma": "pagamento", "message_id": "msg1", "count": 1, "embedding_score": 0.7},
            {"label_id": "CONTABILITA", "lemma": "saldo", "message_id": "msg2", "count": 3, "embedding_score": 0.6},
        ]

        promoter = KeywordPromoter()
        stats = promoter.compute_stats(observations)

        assert len(stats) == 2  # 2 labels
        assert "FATTURAZIONE" in stats
        assert "CONTABILITA" in stats
        assert len(stats["FATTURAZIONE"]) == 2  # 2 lemmas in FATTURAZIONE

    def test_promote_keywords_regex_active(self):
        """Test promotion to regex_active."""
        observations = [
            # High doc_freq, high score, no collisions -> regex_active
            {"label_id": "FATTURAZIONE", "lemma": "fattura", "message_id": "msg1", "term": "fattura", "count": 2, "embedding_score": 0.9},
            {"label_id": "FATTURAZIONE", "lemma": "fattura", "message_id": "msg2", "term": "Fattura", "count": 1, "embedding_score": 0.85},
            {"label_id": "FATTURAZIONE", "lemma": "fattura", "message_id": "msg3", "term": "FATTURA", "count": 3, "embedding_score": 0.88},
        ]

        config = {
            "regex_min_doc_freq": 3,
            "regex_min_embedding_score": 0.35,
            "ner_min_doc_freq": 2,
            "ner_min_embedding_score": 0.25,
            "max_collision_labels": 2,
            "min_total_count": 5,
        }

        promoter = KeywordPromoter(config=config)
        updates = promoter.promote_keywords(observations)

        assert len(updates["regex_active"]) == 1
        assert len(updates["ner_active"]) == 1  # Also promoted to NER
        assert updates["regex_active"][0]["lemma"] == "fattura"
        assert updates["regex_active"][0]["dict_type"] == "regex"
        assert "regex_pattern" in updates["regex_active"][0]

    def test_promote_keywords_ner_only(self):
        """Test promotion to ner_active only (not regex)."""
        observations = [
            # Lower doc_freq, meets NER threshold but not regex
            {"label_id": "ASSISTENZA", "lemma": "problema", "message_id": "msg1", "term": "problema", "count": 3, "embedding_score": 0.7},
            {"label_id": "ASSISTENZA", "lemma": "problema", "message_id": "msg2", "term": "problemi", "count": 2, "embedding_score": 0.65},
        ]

        config = {
            "regex_min_doc_freq": 3,
            "regex_min_embedding_score": 0.35,
            "ner_min_doc_freq": 2,
            "ner_min_embedding_score": 0.25,
            "max_collision_labels": 2,
            "min_total_count": 5,
        }

        promoter = KeywordPromoter(config=config)
        updates = promoter.promote_keywords(observations)

        assert len(updates["regex_active"]) == 0  # doc_freq = 2, needs 3
        assert len(updates["ner_active"]) == 1
        assert updates["ner_active"][0]["lemma"] == "problema"

    def test_promote_keywords_rejected_low_count(self):
        """Test rejection due to low total count."""
        observations = [
            {"label_id": "FATTURAZIONE", "lemma": "fattura", "message_id": "msg1", "term": "fattura", "count": 1, "embedding_score": 0.9},
            {"label_id": "FATTURAZIONE", "lemma": "fattura", "message_id": "msg2", "term": "Fattura", "count": 1, "embedding_score": 0.85},
        ]

        config = {
            "regex_min_doc_freq": 2,
            "regex_min_embedding_score": 0.35,
            "ner_min_doc_freq": 2,
            "ner_min_embedding_score": 0.25,
            "max_collision_labels": 2,
            "min_total_count": 5,  # Total count = 2, needs 5
        }

        promoter = KeywordPromoter(config=config)
        updates = promoter.promote_keywords(observations)

        assert len(updates["rejected"]) == 1
        assert updates["rejected"][0]["reason"] == "low_count"

    def test_promote_keywords_quarantined_high_collision(self):
        """Test quarantine due to high collision."""
        observations = [
            # Same lemma in 3 labels -> quarantined
            {"label_id": "FATTURAZIONE", "lemma": "documento", "message_id": "msg1", "term": "documento", "count": 6, "embedding_score": 0.8},
            {"label_id": "CONTABILITA", "lemma": "documento", "message_id": "msg2", "term": "documento", "count": 7, "embedding_score": 0.75},
            {"label_id": "ASSISTENZA", "lemma": "documento", "message_id": "msg3", "term": "documento", "count": 5, "embedding_score": 0.7},
        ]

        config = {
            "regex_min_doc_freq": 1,
            "regex_min_embedding_score": 0.35,
            "ner_min_doc_freq": 1,
            "ner_min_embedding_score": 0.25,
            "max_collision_labels": 2,  # 3 labels > threshold
            "min_total_count": 5,
        }

        promoter = KeywordPromoter(config=config)
        updates = promoter.promote_keywords(observations)

        assert len(updates["quarantined"]) == 3  # One per label
        assert all(update["reason"] == "high_collision" for update in updates["quarantined"])

    def test_promote_keywords_quarantined_insufficient_evidence(self):
        """Test quarantine due to insufficient evidence."""
        observations = [
            # Low doc_freq and low score
            {"label_id": "FATTURAZIONE", "lemma": "test", "message_id": "msg1", "term": "test", "count": 10, "embedding_score": 0.1},
        ]

        config = {
            "regex_min_doc_freq": 3,
            "regex_min_embedding_score": 0.35,
            "ner_min_doc_freq": 2,
            "ner_min_embedding_score": 0.25,
            "max_collision_labels": 2,
            "min_total_count": 5,
        }

        promoter = KeywordPromoter(config=config)
        updates = promoter.promote_keywords(observations)

        assert len(updates["quarantined"]) == 1
        assert updates["quarantined"][0]["reason"] == "insufficient_evidence"

    def test_promote_keywords_surface_forms(self):
        """Test surface forms collection."""
        observations = [
            {"label_id": "FATTURAZIONE", "lemma": "fattura", "message_id": "msg1", "term": "fattura", "count": 2, "embedding_score": 0.9},
            {"label_id": "FATTURAZIONE", "lemma": "fattura", "message_id": "msg2", "term": "Fattura", "count": 1, "embedding_score": 0.85},
            {"label_id": "FATTURAZIONE", "lemma": "fattura", "message_id": "msg3", "term": "FATTURA", "count": 3, "embedding_score": 0.88},
        ]

        config = {
            "regex_min_doc_freq": 3,
            "regex_min_embedding_score": 0.35,
            "ner_min_doc_freq": 2,
            "ner_min_embedding_score": 0.25,
            "max_collision_labels": 2,
            "min_total_count": 5,
        }

        promoter = KeywordPromoter(config=config)
        updates = promoter.promote_keywords(observations)

        assert len(updates["regex_active"]) == 1
        surface_forms = updates["regex_active"][0]["surface_forms"]
        assert len(surface_forms) == 3
        assert set(surface_forms) == {"fattura", "Fattura", "FATTURA"}
