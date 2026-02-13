"""
Italian tokenizer with deterministic n-gram generation and lemmatization (Phase 2).

Provides stable tokenization for Italian text with support for:
- Apostrophes (dell'ordine, l'utente)
- Italian accented characters (àèéìòù)
- N-gram generation (unigrams, bigrams, trigrams)
- Stable ID generation via SHA1
- spaCy-based lemmatization for Italian
"""

import hashlib
import re
from typing import List, Optional

import spacy
from spacy.language import Language

# Version for audit trail
TOKENIZER_VERSION = "tokenizer-1.0.0"
NER_MODEL_VERSION = "it_core_news_lg-3.8.0"

# Singleton for spaCy model
_spacy_model: Optional[Language] = None


def get_spacy_model() -> Language:
    """
    Get or initialize spaCy model (singleton pattern).

    Uses it_core_news_lg for Italian lemmatization and NER.
    Model is loaded once and cached for performance.

    Returns:
        Initialized spaCy Language instance

    Raises:
        OSError: If spaCy model is not installed
    """
    global _spacy_model
    if _spacy_model is None:
        try:
            _spacy_model = spacy.load("it_core_news_lg")
        except OSError as e:
            raise OSError(
                "spaCy model 'it_core_news_lg' not found. "
                "Please install it with: python -m spacy download it_core_news_lg"
            ) from e
    return _spacy_model


def tokenize_it(text: str) -> List[str]:
    """
    Tokenize Italian text deterministically.

    Pattern matches:
    - Italian words with apostrophes: dell'ordine, l'utente
    - Accented characters: àèéìòù
    - Numbers: preserved as tokens

    Args:
        text: Input text (any case)

    Returns:
        List of tokens (lowercased)

    Examples:
        >>> tokenize_it("Buongiorno, ho un problema")
        ['buongiorno', 'ho', 'un', 'problema']
        >>> tokenize_it("dell'ordine numero 12345")
        ["dell'ordine", 'numero', '12345']
    """
    text = text.lower()
    # Pattern: Italian word chars + optional apostrophe + more chars, OR digits
    tokens = re.findall(r"[a-zàèéìòù]+(?:'[a-zàèéìòù]+)?|\d+", text, flags=re.IGNORECASE)
    return tokens


def lemmatize_term(term: str) -> str:
    """
    Lemmatize Italian term using spaCy.

    For multi-word terms (n-grams), each word is lemmatized separately
    and then rejoined with spaces.

    Args:
        term: Input term (can be unigram, bigram, or trigram)

    Returns:
        Lemmatized term (lowercase)

    Examples:
        >>> lemmatize_term("problemi")
        'problema'
        >>> lemmatize_term("ordini")
        'ordine'
        >>> lemmatize_term("problemi gravi")  # bigram
        'problema grave'
    """
    try:
        nlp = get_spacy_model()
        doc = nlp(term.lower())
        lemmas = [token.lemma_ for token in doc]
        return " ".join(lemmas)
    except Exception:
        # Fallback: return lowercase term if lemmatization fails
        return term.lower()


def ngrams(tokens: List[str], n: int) -> List[str]:
    """
    Generate n-grams from token list.

    Args:
        tokens: List of tokens
        n: N-gram size (1=unigram, 2=bigram, 3=trigram)

    Returns:
        List of n-gram strings (space-joined)

    Examples:
        >>> ngrams(["a", "b", "c"], 1)
        ['a', 'b', 'c']
        >>> ngrams(["a", "b", "c"], 2)
        ['a b', 'b c']
        >>> ngrams(["a", "b", "c", "d"], 3)
        ['a b c', 'b c d']
    """
    if n < 1:
        return []
    if n > len(tokens):
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def stable_id(*parts: str) -> str:
    """
    Generate deterministic ID from parts using SHA1.

    Args:
        *parts: String parts to hash (e.g., source, term)

    Returns:
        Hex SHA1 digest (first 12 chars for brevity)

    Examples:
        >>> stable_id("subject", "problema")
        'a1b2c3d4e5f6'  # deterministic
        >>> stable_id("subject", "problema") == stable_id("subject", "problema")
        True
        >>> stable_id("subject", "problema") != stable_id("body", "problema")
        True
    """
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]
