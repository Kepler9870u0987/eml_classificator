# Text canonicalization module

from .text_normalizer import (
    CANONICALIZATION_VERSION,
    canonicalize_text,
    normalize_whitespace,
    remove_email_signatures,
    remove_excessive_newlines,
    remove_reply_chains,
    remove_urls,
)
from .html_converter import clean_html, html_to_text, strip_html_tags

__all__ = [
    "CANONICALIZATION_VERSION",
    "canonicalize_text",
    "normalize_whitespace",
    "remove_email_signatures",
    "remove_reply_chains",
    "remove_urls",
    "remove_excessive_newlines",
    "html_to_text",
    "strip_html_tags",
    "clean_html",
]
