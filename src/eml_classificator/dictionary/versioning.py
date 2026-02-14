"""
Dictionary version management.

Implements version control for dictionaries to ensure reproducibility
as specified in brainstorming v2 section 5.1 (lines 1384-1397).
"""

from datetime import datetime
from typing import Dict, List

import structlog
from sqlalchemy import func

from .database import get_db_session
from .models import LexiconEntry

logger = structlog.get_logger(__name__)


class DictionaryVersion:
    """
    Manage dictionary versions for reproducibility.

    Dictionary versions track changes to lexicon entries,
    allowing pipeline runs to freeze dictionary state and reproduce results.

    As specified in brainstorming v2 section 5.1.
    """

    @staticmethod
    def get_current_version() -> int:
        """
        Get current dictionary version.

        Returns:
            Current version number

        Examples:
            >>> version = DictionaryVersion.get_current_version()
            >>> version
            1
        """
        from eml_classificator.config import settings

        return settings.current_dictionary_version

    @staticmethod
    def create_new_version() -> int:
        """
        Increment dictionary version.

        Call this when promoting new entries to active status.

        Returns:
            New version number

        Examples:
            >>> new_version = DictionaryVersion.create_new_version()
            >>> new_version
            2
        """
        current = DictionaryVersion.get_current_version()
        new_version = current + 1

        logger.info(
            "dictionary_version_incremented",
            old_version=current,
            new_version=new_version
        )

        # NOTE: In production, this should update a version table or config store
        # For now, it returns the incremented version
        return new_version

    @staticmethod
    def freeze_version(version: int) -> Dict[str, any]:
        """
        Freeze dictionary state at given version.

        Returns snapshot of all active entries at or before the specified version.

        Args:
            version: Dictionary version to freeze

        Returns:
            Dictionary with version metadata and entry counts

        Examples:
            >>> snapshot = DictionaryVersion.freeze_version(1)
            >>> snapshot['version']
            1
            >>> snapshot['regex_entries']
            45
        """
        with get_db_session() as session:
            # Query active regex entries at or before this version
            regex_lexicon = (
                session.query(LexiconEntry)
                .filter(
                    LexiconEntry.dict_type == "regex",
                    LexiconEntry.status == "active",
                    LexiconEntry.dict_version_added <= version,
                )
                .all()
            )

            # Query active NER entries at or before this version
            ner_lexicon = (
                session.query(LexiconEntry)
                .filter(
                    LexiconEntry.dict_type == "ner",
                    LexiconEntry.status == "active",
                    LexiconEntry.dict_version_added <= version,
                )
                .all()
            )

            snapshot = {
                "version": version,
                "frozen_at": datetime.utcnow().isoformat(),
                "regex_entries": len(regex_lexicon),
                "ner_entries": len(ner_lexicon),
                "total_entries": len(regex_lexicon) + len(ner_lexicon),
            }

            logger.info(
                "dictionary_version_frozen",
                version=version,
                regex_entries=snapshot["regex_entries"],
                ner_entries=snapshot["ner_entries"],
            )

            return snapshot

    @staticmethod
    def get_active_lexicon_for_version(
        version: int, dict_type: str = "regex"
    ) -> Dict[str, List[Dict]]:
        """
        Get active lexicon entries for a specific version.

        Args:
            version: Dictionary version
            dict_type: "regex" or "ner"

        Returns:
            Dictionary mapping label_id to list of entries

        Examples:
            >>> lexicon = DictionaryVersion.get_active_lexicon_for_version(1, "regex")
            >>> lexicon["FATTURAZIONE"]
            [{'lemma': 'fattura', 'regex_pattern': '...', ...}]
        """
        with get_db_session() as session:
            entries = (
                session.query(LexiconEntry)
                .filter(
                    LexiconEntry.dict_type == dict_type,
                    LexiconEntry.status == "active",
                    LexiconEntry.dict_version_added <= version,
                )
                .all()
            )

            # Group by label_id
            lexicon = {}
            for entry in entries:
                if entry.label_id not in lexicon:
                    lexicon[entry.label_id] = []

                lexicon[entry.label_id].append(
                    {
                        "entry_id": entry.entry_id,
                        "lemma": entry.lemma,
                        "surface_forms": entry.surface_forms,
                        "regex_pattern": entry.regex_pattern,
                        "doc_freq": entry.doc_freq,
                        "total_count": entry.total_count,
                        "embedding_score": entry.embedding_score,
                    }
                )

            logger.debug(
                "lexicon_retrieved_for_version",
                version=version,
                dict_type=dict_type,
                labels_count=len(lexicon),
                total_entries=len(entries),
            )

            return lexicon
