"""
Canonicalization models for tracking text transformations.

This module defines data structures for audit trails during email text normalization.
"""

from dataclasses import dataclass
from typing import Literal

# Type alias for section types
SectionType = Literal["quote", "signature", "disclaimer", "reply_header"]


@dataclass
class RemovedSection:
    """
    Traccia cosa è stato rimosso per audit.

    Tracks removed sections during text canonicalization for GDPR compliance
    and debugging. Records the type of content removed, its position in the
    original text, and the actual content.

    Attributes:
        type: Type of removed content (quote, signature, disclaimer, reply_header)
        span_start: Start position in original text (character index)
        span_end: End position in original text (character index)
        content: The actual removed content
    """

    type: str  # "quote" | "signature" | "disclaimer" | "reply_header"
    span_start: int
    span_end: int
    content: str

    def __post_init__(self):
        """Validate section type."""
        valid_types = {"quote", "signature", "disclaimer", "reply_header"}
        if self.type not in valid_types:
            raise ValueError(
                f"Invalid type '{self.type}'. Must be one of {valid_types}"
            )
        if self.span_start < 0 or self.span_end < 0:
            raise ValueError("Span positions must be non-negative")
        if self.span_start > self.span_end:
            raise ValueError("span_start must be <= span_end")
