"""
SQLAlchemy models for dictionary database.

Implements database schema as specified in brainstorming v2 section 5.1 (lines 1334-1380).
"""

from datetime import datetime
from typing import List

from sqlalchemy import JSON, Boolean, Column, Float, Index, Integer, String, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class LabelRegistry(Base):
    """
    Registry of classification labels/topics.

    Tracks label lifecycle: active | proposed | deprecated | merged

    As specified in brainstorming v2 section 5.1.
    """

    __tablename__ = "label_registry"

    label_id = Column(String, primary_key=True)  # e.g., "FATTURAZIONE"
    name = Column(String, nullable=False)  # Human-readable name
    status = Column(String, nullable=False)  # active | proposed | deprecated | merged
    created_at = Column(TIMESTAMP, server_default=func.now())
    merged_into = Column(String, nullable=True)  # If merged, points to new label_id

    __table_args__ = (Index("idx_label_status", "status"),)

    def __repr__(self):
        return f"<LabelRegistry(label_id={self.label_id}, status={self.status})>"


class LexiconEntry(Base):
    """
    Dictionary entries for regex or NER-based entity extraction.

    Each entry represents a keyword/term associated with a label,
    with statistics for promotion decisions.

    As specified in brainstorming v2 section 5.1.
    """

    __tablename__ = "lexicon_entries"

    entry_id = Column(String, primary_key=True)  # UUID
    label_id = Column(String, nullable=False)  # Foreign key to label_registry
    dict_type = Column(String, nullable=False)  # 'regex' | 'ner'
    lemma = Column(String, nullable=False)  # Canonical form
    surface_forms = Column(JSON, nullable=False)  # ["fattura", "Fattura", "FATTURA"]
    regex_pattern = Column(String, nullable=True)  # Only for dict_type='regex'
    status = Column(
        String, nullable=False
    )  # active | candidate | quarantined | rejected

    # Statistics (from promoter)
    doc_freq = Column(Integer, default=0)  # Number of unique documents
    total_count = Column(Integer, default=0)  # Total occurrences
    embedding_score = Column(Float, nullable=True)  # Average KeyBERT score

    # Timestamps
    first_seen_at = Column(TIMESTAMP, server_default=func.now())
    last_seen_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    dict_version_added = Column(Integer, nullable=False)  # Version when promoted

    __table_args__ = (
        Index("idx_label_status", "label_id", "status"),
        Index("idx_lemma", "lemma"),
        Index("idx_dict_type", "dict_type"),
    )

    def __repr__(self):
        return f"<LexiconEntry(entry_id={self.entry_id}, lemma={self.lemma}, status={self.status})>"


class KeywordObservation(Base):
    """
    Raw observations from classification outputs.

    Collected after each classification to track keyword frequency
    for promotion decisions.

    As specified in brainstorming v2 section 5.1.
    """

    __tablename__ = "keyword_observations"

    obs_id = Column(String, primary_key=True)  # UUID
    message_id = Column(String, nullable=False)  # Email/message ID
    label_id = Column(String, nullable=False)  # Classification label
    lemma = Column(String, nullable=False)  # Canonical form
    term = Column(String, nullable=False)  # Surface form as seen in text
    count = Column(Integer, default=1)  # Occurrences in this message
    embedding_score = Column(Float, nullable=True)  # KeyBERT score
    dict_version = Column(Integer, nullable=False)  # Dictionary version used
    promoted_to_active = Column(Boolean, default=False)  # Whether promoted
    observed_at = Column(TIMESTAMP, server_default=func.now())

    __table_args__ = (
        Index("idx_label_lemma", "label_id", "lemma"),
        Index("idx_observed_at", "observed_at"),
        Index("idx_message", "message_id"),
    )

    def __repr__(self):
        return (
            f"<KeywordObservation(obs_id={self.obs_id}, "
            f"lemma={self.lemma}, label_id={self.label_id})>"
        )
