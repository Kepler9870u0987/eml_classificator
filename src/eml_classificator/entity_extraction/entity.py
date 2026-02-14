"""
Entity dataclass for entity extraction.

Represents extracted entities with span information as specified
in brainstorming v2 section 5.4.
"""

from dataclasses import dataclass


@dataclass
class Entity:
    """
    Represents an extracted entity with span information.

    As specified in brainstorming v2 section 5.4, lines 1677-1691.

    Attributes:
        text: Matched text from source document
        label: Entity label (lemma for regex/lexicon, NER type for spaCy)
        start: Character start position in source text
        end: Character end position in source text
        source: Extraction method ("regex" | "ner" | "lexicon")
        confidence: Confidence score (0.0-1.0)
    """

    text: str
    label: str
    start: int
    end: int
    source: str  # "regex" | "ner" | "lexicon"
    confidence: float = 1.0

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Entity('{self.text}', {self.label}, [{self.start},{self.end}], {self.source})"

    def overlaps(self, other: "Entity") -> bool:
        """
        Check if two entities overlap.

        Args:
            other: Another entity to check overlap with

        Returns:
            True if entities overlap, False otherwise
        """
        return not (self.end <= other.start or other.end <= self.start)

    def length(self) -> int:
        """
        Get entity span length.

        Returns:
            Number of characters in entity span
        """
        return self.end - self.start
