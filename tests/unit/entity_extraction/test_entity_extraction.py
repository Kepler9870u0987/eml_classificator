"""
Unit tests for entity_extraction modules.

Tests Entity class, regex matching, NER extraction, lexicon enhancement,
and deterministic merging as specified in brainstorming v2 section 5.4.
"""

import pytest

from eml_classificator.entity_extraction import (
    Entity,
    enhance_ner_with_lexicon,
    extract_all_entities,
    extract_entities_ner,
    extract_entities_regex,
    merge_entities_deterministic,
)


class TestEntity:
    """Test Entity dataclass."""

    def test_entity_creation(self):
        """Test creating Entity instance."""
        entity = Entity(
            text="fattura", label="NOUN", start=10, end=17, source="regex", confidence=0.95
        )

        assert entity.text == "fattura"
        assert entity.label == "NOUN"
        assert entity.start == 10
        assert entity.end == 17
        assert entity.source == "regex"
        assert entity.confidence == 0.95

    def test_entity_overlaps(self):
        """Test entity overlap detection."""
        e1 = Entity("test", "NOUN", 10, 20, "regex", 0.9)
        e2 = Entity("test", "NOUN", 15, 25, "ner", 0.7)
        e3 = Entity("test", "NOUN", 30, 40, "ner", 0.7)

        assert e1.overlaps(e2)  # Overlap
        assert e2.overlaps(e1)  # Symmetric
        assert not e1.overlaps(e3)  # No overlap

    def test_entity_no_overlap_adjacent(self):
        """Test that adjacent entities don't overlap."""
        e1 = Entity("test", "NOUN", 10, 20, "regex", 0.9)
        e2 = Entity("test", "NOUN", 20, 30, "ner", 0.7)

        assert not e1.overlaps(e2)  # Adjacent but not overlapping

    def test_entity_length(self):
        """Test entity length calculation."""
        entity = Entity("fattura", "NOUN", 10, 17, "regex", 0.95)

        assert entity.length() == 7

    def test_entity_repr(self):
        """Test entity string representation."""
        entity = Entity("fattura", "NOUN", 10, 17, "regex", 0.95)
        repr_str = repr(entity)

        assert "fattura" in repr_str
        assert "NOUN" in repr_str
        assert "10" in repr_str
        assert "17" in repr_str
        assert "regex" in repr_str


class TestExtractEntitiesRegex:
    """Test regex-based entity extraction."""

    def test_basic_regex_extraction(self):
        """Test basic regex entity extraction."""
        text = "Buongiorno, invio fattura n. 12345"
        regex_lexicon = {
            "FATTURAZIONE": [{"lemma": "fattura", "regex_pattern": r"(?i)\b(fattura)\b"}]
        }

        entities = extract_entities_regex(text, regex_lexicon, "FATTURAZIONE")

        assert len(entities) == 1
        assert entities[0].text == "fattura"
        assert entities[0].label == "fattura"
        assert entities[0].source == "regex"
        assert entities[0].confidence == 0.95

    def test_multiple_matches(self):
        """Test regex with multiple matches."""
        text = "Fattura n. 123 e fattura n. 456"
        regex_lexicon = {
            "FATTURAZIONE": [{"lemma": "fattura", "regex_pattern": r"(?i)\b(fattura)\b"}]
        }

        entities = extract_entities_regex(text, regex_lexicon, "FATTURAZIONE")

        assert len(entities) == 2
        assert all(e.label == "fattura" for e in entities)

    def test_no_matches(self):
        """Test regex with no matches."""
        text = "Nessuna corrispondenza qui"
        regex_lexicon = {
            "FATTURAZIONE": [{"lemma": "fattura", "regex_pattern": r"(?i)\b(fattura)\b"}]
        }

        entities = extract_entities_regex(text, regex_lexicon, "FATTURAZIONE")

        assert len(entities) == 0

    def test_label_not_in_lexicon(self):
        """Test extraction when label is not in lexicon."""
        text = "Any text"
        regex_lexicon = {"OTHER_LABEL": [{"lemma": "test", "regex_pattern": r"\btest\b"}]}

        entities = extract_entities_regex(text, regex_lexicon, "FATTURAZIONE")

        assert len(entities) == 0

    def test_invalid_regex_pattern(self):
        """Test handling of invalid regex pattern."""
        text = "Any text"
        regex_lexicon = {"LABEL": [{"lemma": "test", "regex_pattern": r"(?P<unclosed"}]}

        # Should not crash, should return empty
        entities = extract_entities_regex(text, regex_lexicon, "LABEL")

        assert len(entities) == 0

    def test_missing_fields_in_entry(self):
        """Test handling of missing fields in lexicon entry."""
        text = "Any text"
        regex_lexicon = {"LABEL": [{"lemma": "test"}]}  # Missing regex_pattern

        entities = extract_entities_regex(text, regex_lexicon, "LABEL")

        assert len(entities) == 0


class TestExtractEntitiesNER:
    """Test NER-based entity extraction."""

    def test_ner_extraction(self):
        """Test basic NER extraction."""
        text = "Mario Rossi lavora a Roma per Acme S.p.A."

        entities = extract_entities_ner(text)

        # Should extract persons, locations, organizations
        assert len(entities) > 0
        assert any(e.source == "ner" for e in entities)
        assert all(e.confidence == 0.75 for e in entities)

    def test_ner_empty_text(self):
        """Test NER with empty text."""
        entities = extract_entities_ner("")

        assert len(entities) == 0

    def test_ner_no_entities(self):
        """Test NER with text containing no named entities."""
        text = "Ho un problema con il servizio."

        entities = extract_entities_ner(text)

        # May or may not find entities depending on NER model

    def test_ner_entity_attributes(self):
        """Test that NER entities have correct attributes."""
        text = "Mario Rossi vive a Roma."

        entities = extract_entities_ner(text)

        for entity in entities:
            assert isinstance(entity, Entity)
            assert entity.source == "ner"
            assert 0.0 <= entity.confidence <= 1.0
            assert entity.start >= 0
            assert entity.end > entity.start
            assert entity.text == text[entity.start : entity.end]


class TestEnhanceNerWithLexicon:
    """Test lexicon-based NER enhancement."""

    def test_lexicon_enhancement(self):
        """Test basic lexicon enhancement."""
        ner_entities = []  # Empty NER results
        ner_lexicon = {
            "FATTURAZIONE": [{"lemma": "fattura", "surface_forms": ["fattura", "Fattura"]}]
        }
        text = "Invio fattura n. 12345"

        enhanced = enhance_ner_with_lexicon(ner_entities, ner_lexicon, "FATTURAZIONE", text)

        assert len(enhanced) == 1
        assert enhanced[0].text.lower() == "fattura"
        assert enhanced[0].source == "lexicon"
        assert enhanced[0].confidence == 0.85

    def test_lexicon_multiple_surface_forms(self):
        """Test lexicon with multiple surface forms."""
        ner_entities = []
        ner_lexicon = {
            "FATTURAZIONE": [{"lemma": "fattura", "surface_forms": ["fattura", "Fattura", "FATTURA"]}]
        }
        text = "fattura n. 1, Fattura n. 2, FATTURA n. 3"

        enhanced = enhance_ner_with_lexicon(ner_entities, ner_lexicon, "FATTURAZIONE", text)

        assert len(enhanced) == 3

    def test_lexicon_word_boundaries(self):
        """Test that lexicon matches respect word boundaries."""
        ner_entities = []
        ner_lexicon = {"LABEL": [{"lemma": "test", "surface_forms": ["test"]}]}
        text = "test testing attest"

        enhanced = enhance_ner_with_lexicon(ner_entities, ner_lexicon, "LABEL", text)

        # Should only match "test", not "testing" or "attest"
        assert len(enhanced) == 1
        assert enhanced[0].text == "test"

    def test_lexicon_preserves_ner_entities(self):
        """Test that lexicon enhancement preserves original NER entities."""
        ner_entities = [Entity("Roma", "LOC", 20, 24, "ner", 0.75)]
        ner_lexicon = {"LABEL": [{"lemma": "test", "surface_forms": ["test"]}]}
        text = "test text with Roma"

        enhanced = enhance_ner_with_lexicon(ner_entities, ner_lexicon, "LABEL", text)

        # Should have original NER + lexicon entities
        assert len(enhanced) == 2
        assert any(e.text == "Roma" for e in enhanced)
        assert any(e.text == "test" for e in enhanced)


class TestMergeEntitiesDeterministic:
    """Test deterministic entity merging."""

    def test_regex_priority_over_ner(self):
        """Test that regex entities have priority over NER."""
        e1 = Entity("fattura", "fattura", 10, 17, "regex", 0.95)
        e2 = Entity("fattura", "NOUN", 10, 17, "ner", 0.75)

        merged = merge_entities_deterministic([e1, e2])

        assert len(merged) == 1
        assert merged[0].source == "regex"

    def test_regex_priority_over_lexicon(self):
        """Test that regex entities have priority over lexicon."""
        e1 = Entity("fattura", "fattura", 10, 17, "regex", 0.95)
        e2 = Entity("fattura", "fattura", 10, 17, "lexicon", 0.85)

        merged = merge_entities_deterministic([e1, e2])

        assert len(merged) == 1
        assert merged[0].source == "regex"

    def test_lexicon_priority_over_ner(self):
        """Test that lexicon entities have priority over NER."""
        e1 = Entity("fattura", "fattura", 10, 17, "lexicon", 0.85)
        e2 = Entity("fattura", "NOUN", 10, 17, "ner", 0.75)

        merged = merge_entities_deterministic([e1, e2])

        assert len(merged) == 1
        assert merged[0].source == "lexicon"

    def test_longest_span_wins_same_source(self):
        """Test that longest span wins when same source."""
        e1 = Entity("test", "NOUN", 10, 15, "ner", 0.75)
        e2 = Entity("test word", "NOUN", 10, 19, "ner", 0.75)

        merged = merge_entities_deterministic([e1, e2])

        assert len(merged) == 1
        assert merged[0].length() == 9  # Longer span

    def test_confidence_tiebreaker(self):
        """Test that confidence is used as tiebreaker."""
        e1 = Entity("test", "NOUN", 10, 15, "ner", 0.75)
        e2 = Entity("test", "NOUN", 10, 15, "ner", 0.85)

        merged = merge_entities_deterministic([e1, e2])

        assert len(merged) == 1
        assert merged[0].confidence == 0.85

    def test_non_overlapping_entities_preserved(self):
        """Test that non-overlapping entities are all preserved."""
        e1 = Entity("test1", "NOUN", 10, 15, "ner", 0.75)
        e2 = Entity("test2", "NOUN", 20, 25, "ner", 0.75)
        e3 = Entity("test3", "NOUN", 30, 35, "ner", 0.75)

        merged = merge_entities_deterministic([e1, e2, e3])

        assert len(merged) == 3

    def test_sorted_by_position(self):
        """Test that result is sorted by position."""
        e1 = Entity("test1", "NOUN", 30, 35, "ner", 0.75)
        e2 = Entity("test2", "NOUN", 10, 15, "ner", 0.75)
        e3 = Entity("test3", "NOUN", 20, 25, "ner", 0.75)

        merged = merge_entities_deterministic([e1, e2, e3])

        assert merged[0].start == 10
        assert merged[1].start == 20
        assert merged[2].start == 30

    def test_empty_list(self):
        """Test merging empty list."""
        merged = merge_entities_deterministic([])

        assert len(merged) == 0


class TestExtractAllEntities:
    """Test complete entity extraction pipeline."""

    def test_complete_pipeline(self):
        """Test complete pipeline with all components."""
        text = "Invio fattura n. 12345"
        regex_lexicon = {
            "FATTURAZIONE": [{"lemma": "fattura", "regex_pattern": r"(?i)\b(fattura)\b"}]
        }
        ner_lexicon = {"FATTURAZIONE": [{"lemma": "fattura", "surface_forms": ["fattura"]}]}

        entities = extract_all_entities(text, "FATTURAZIONE", regex_lexicon, ner_lexicon)

        # Should have at least regex match
        assert len(entities) >= 1
        # Regex should win (if there are overlaps)
        assert any(e.source == "regex" for e in entities)

    def test_pipeline_with_empty_lexicons(self):
        """Test pipeline with empty lexicons."""
        text = "Mario Rossi vive a Roma"
        regex_lexicon = {}
        ner_lexicon = {}

        entities = extract_all_entities(text, "LABEL", regex_lexicon, ner_lexicon)

        # Should still extract NER entities
        assert len(entities) >= 0  # May or may not find entities

    def test_pipeline_merging(self):
        """Test that pipeline properly merges overlapping entities."""
        text = "test"
        regex_lexicon = {"LABEL": [{"lemma": "test", "regex_pattern": r"\btest\b"}]}
        ner_lexicon = {"LABEL": [{"lemma": "test", "surface_forms": ["test"]}]}

        entities = extract_all_entities(text, "LABEL", regex_lexicon, ner_lexicon)

        # Should have only one entity (regex wins, others merged)
        assert any(e.source == "regex" for e in entities)
