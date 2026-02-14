"""Dictionary management module.

Provides regex generation, database models, keyword promotion,
and dictionary versioning functionality.

As specified in brainstorming v2 section 5 (lines 1333-1896).
"""

from .collision import detect_collisions, get_collision_index, resolve_collision_by_score
from .database import create_all_tables, drop_all_tables, get_db_session, get_engine
from .models import KeywordObservation, LabelRegistry, LexiconEntry
from .observations import (
    collect_observations_from_classification,
    get_recent_observations,
    mark_observations_promoted,
)
from .promoter import KeywordPromoter
from .regex_generator import generate_safe_regex, validate_regex_pattern
from .versioning import DictionaryVersion

__all__ = [
    # Regex generation
    "generate_safe_regex",
    "validate_regex_pattern",
    # Database models
    "LabelRegistry",
    "LexiconEntry",
    "KeywordObservation",
    # Database connection
    "get_engine",
    "get_db_session",
    "create_all_tables",
    "drop_all_tables",
    # Observations
    "collect_observations_from_classification",
    "get_recent_observations",
    "mark_observations_promoted",
    # Promotion
    "KeywordPromoter",
    # Collision detection
    "detect_collisions",
    "get_collision_index",
    "resolve_collision_by_score",
    # Versioning
    "DictionaryVersion",
]
