"""
Keyword promoter with deterministic rules.

Implements keyword promotion as specified in brainstorming v2 section 5.2 (lines 1398-1578).
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set

import structlog

from .collision import get_collision_index
from .regex_generator import generate_safe_regex

logger = structlog.get_logger(__name__)


class KeywordPromoter:
    """
    Manages keyword promotion to dictionaries with deterministic rules.

    Analyzes keyword observations and decides which keywords should be
    promoted to active regex/NER dictionaries based on:
    - Document frequency (doc_freq)
    - Total occurrence count (total_count)
    - Average embedding score (avg_embedding_score)
    - Label collisions

    As specified in brainstorming v2 section 5.2.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize promoter with configuration.

        Args:
            config: Optional configuration dict. If None, uses defaults from settings.

        Examples:
            >>> promoter = KeywordPromoter()
            >>> promoter.config["regex_min_doc_freq"]
            3
        """
        if config is None:
            from eml_classificator.config import settings

            self.config = {
                "regex_min_doc_freq": settings.promoter_regex_min_doc_freq,
                "regex_min_embedding_score": settings.promoter_regex_min_embedding_score,
                "ner_min_doc_freq": settings.promoter_ner_min_doc_freq,
                "ner_min_embedding_score": settings.promoter_ner_min_embedding_score,
                "max_collision_labels": settings.promoter_max_collision_labels,
                "min_total_count": settings.promoter_min_total_count,
            }
        else:
            self.config = config

        logger.info("keyword_promoter_initialized", config=self.config)

    def compute_stats(self, observations: List[dict]) -> Dict[str, Dict[str, dict]]:
        """
        Calculate aggregate statistics for (label_id, lemma) pairs.

        Computes:
        - doc_freq: Number of unique messages containing this lemma for this label
        - total_count: Sum of all occurrences across messages
        - avg_embedding_score: Average KeyBERT score across observations

        Args:
            observations: List of observation dicts with keys:
                          label_id, lemma, message_id, count, embedding_score

        Returns:
            Nested dict: {label_id: {lemma: {doc_freq, total_count, avg_embedding_score}}}

        Examples:
            >>> observations = [
            ...     {"label_id": "FATTURAZIONE", "lemma": "fattura", "message_id": "msg1",
            ...      "count": 2, "embedding_score": 0.8},
            ...     {"label_id": "FATTURAZIONE", "lemma": "fattura", "message_id": "msg2",
            ...      "count": 1, "embedding_score": 0.75},
            ... ]
            >>> stats = promoter.compute_stats(observations)
            >>> stats["FATTURAZIONE"]["fattura"]["doc_freq"]
            2
        """
        stats = defaultdict(
            lambda: defaultdict(lambda: {"doc_freq": 0, "total_count": 0, "embedding_scores": []})
        )

        # Track unique messages per (label, lemma) for doc_freq
        message_lemma_label = defaultdict(set)

        for obs in observations:
            label = obs["label_id"]
            lemma = obs["lemma"]
            msg_id = obs["message_id"]
            count = obs.get("count", 1)
            emb_score = obs.get("embedding_score", 0.0)

            # Doc freq: unique messages
            message_lemma_label[(label, lemma)].add(msg_id)

            # Total count
            stats[label][lemma]["total_count"] += count

            # Embedding scores
            if emb_score and emb_score > 0:
                stats[label][lemma]["embedding_scores"].append(emb_score)

        # Finalize doc_freq and average embedding score
        for (label, lemma), msg_ids in message_lemma_label.items():
            stats[label][lemma]["doc_freq"] = len(msg_ids)

        for label in stats:
            for lemma in stats[label]:
                scores = stats[label][lemma]["embedding_scores"]
                stats[label][lemma]["avg_embedding_score"] = sum(scores) / len(scores) if scores else 0.0
                del stats[label][lemma]["embedding_scores"]  # Cleanup

        logger.debug(
            "promoter_stats_computed", labels_count=len(stats), total_lemmas=sum(len(lemmas) for lemmas in stats.values())
        )

        return dict(stats)

    def promote_keywords(
        self, observations: List[dict], existing_lexicon: Optional[Dict[str, List[dict]]] = None
    ) -> Dict[str, List[dict]]:
        """
        Decide promotions for regex and NER lexicon using deterministic rules.

        Decision tree:
        1. Reject if total_count < min_total_count (noise)
        2. Quarantine if collisions > max_collision_labels (ambiguous)
        3. Promote to regex_active if:
           - doc_freq >= regex_min_doc_freq AND
           - avg_embedding_score >= regex_min_embedding_score AND
           - collisions <= 1
        4. Promote to ner_active if:
           - doc_freq >= ner_min_doc_freq AND
           - avg_embedding_score >= ner_min_embedding_score AND
           - collisions <= 2
        5. Otherwise quarantine (candidate, needs more evidence)

        Args:
            observations: List of observation dicts
            existing_lexicon: Optional existing lexicon to avoid duplicates

        Returns:
            Dictionary with keys: regex_active, ner_active, quarantined, rejected

        Examples:
            >>> observations = [...]  # List of keyword observations
            >>> updates = promoter.promote_keywords(observations)
            >>> len(updates["regex_active"])
            5
            >>> len(updates["quarantined"])
            12
        """
        stats = self.compute_stats(observations)
        collision_index = get_collision_index(stats)

        updates = {"regex_active": [], "ner_active": [], "quarantined": [], "rejected": []}

        for label_id, lemmas in stats.items():
            for lemma, stat in lemmas.items():
                doc_freq = stat["doc_freq"]
                total_count = stat["total_count"]
                avg_emb_score = stat["avg_embedding_score"]

                collision_labels = collision_index.get(lemma, {label_id})
                num_collisions = len(collision_labels)

                # Collect surface forms from observations
                surface_forms = list(
                    set(
                        [
                            obs["term"]
                            for obs in observations
                            if obs["label_id"] == label_id and obs["lemma"] == lemma
                        ]
                    )
                )

                entry = {
                    "label_id": label_id,
                    "lemma": lemma,
                    "surface_forms": surface_forms,
                    "doc_freq": doc_freq,
                    "total_count": total_count,
                    "avg_embedding_score": avg_emb_score,
                    "collision_labels": list(collision_labels),
                }

                # Decision tree
                # 1. Reject if total count too low (noise)
                if total_count < self.config["min_total_count"]:
                    updates["rejected"].append({**entry, "reason": "low_count"})
                    continue

                # 2. Quarantine if too many collisions
                if num_collisions > self.config["max_collision_labels"]:
                    updates["quarantined"].append({**entry, "reason": "high_collision"})
                    continue

                # 3. Promotion to regex_active (high precision)
                if (
                    doc_freq >= self.config["regex_min_doc_freq"]
                    and avg_emb_score >= self.config["regex_min_embedding_score"]
                    and num_collisions <= 1
                ):
                    # Generate regex pattern
                    try:
                        regex_pattern = generate_safe_regex(surface_forms, case_sensitive=False)

                        updates["regex_active"].append(
                            {
                                **entry,
                                "dict_type": "regex",
                                "regex_pattern": regex_pattern,
                                "status": "active",
                            }
                        )

                        # Also promote to NER (if in regex, also in NER)
                        updates["ner_active"].append({**entry, "dict_type": "ner", "status": "active"})

                    except Exception as e:
                        logger.error(
                            "regex_generation_failed_during_promotion",
                            lemma=lemma,
                            label_id=label_id,
                            error=str(e),
                        )
                        updates["quarantined"].append({**entry, "reason": "regex_generation_failed"})

                # 4. Promotion to ner_active only (lower precision OK)
                elif (
                    doc_freq >= self.config["ner_min_doc_freq"]
                    and avg_emb_score >= self.config["ner_min_embedding_score"]
                ):
                    if num_collisions <= 2:
                        updates["ner_active"].append({**entry, "dict_type": "ner", "status": "active"})
                    else:
                        updates["quarantined"].append({**entry, "reason": "moderate_collision"})

                # 5. Otherwise quarantine (candidate, needs more evidence)
                else:
                    updates["quarantined"].append({**entry, "reason": "insufficient_evidence"})

        logger.info(
            "keyword_promotion_complete",
            regex_active=len(updates["regex_active"]),
            ner_active=len(updates["ner_active"]),
            quarantined=len(updates["quarantined"]),
            rejected=len(updates["rejected"]),
        )

        return updates
