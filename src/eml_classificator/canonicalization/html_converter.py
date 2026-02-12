"""
HTML to text conversion module.

This module handles conversion of HTML email bodies to plain text using html2text.
"""

import re
import html2text


def html_to_text(html: str, preserve_links: bool = False) -> str:
    """
    Convert HTML email body to plain text.

    Uses html2text library to preserve structure while removing formatting.

    Args:
        html: HTML content
        preserve_links: Whether to keep links as markdown [text](url)

    Returns:
        Plain text representation
    """
    if not html:
        return ""

    # Configure html2text
    h = html2text.HTML2Text()
    h.ignore_links = not preserve_links
    h.ignore_images = True
    h.ignore_emphasis = True
    h.body_width = 0  # Don't wrap lines
    h.unicode_snob = True  # Use unicode instead of ASCII replacements

    try:
        text = h.handle(html)
        return text.strip()
    except Exception:
        # Fallback to simple tag stripping if html2text fails
        return strip_html_tags(html)


def strip_html_tags(html: str) -> str:
    """
    Simple HTML tag stripping (fallback if html2text fails).

    Args:
        html: HTML content

    Returns:
        Text with HTML tags removed
    """
    if not html:
        return ""

    # Remove script and style elements
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", html)

    # Decode HTML entities
    import html as html_module

    text = html_module.unescape(text)

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_html(html: str) -> str:
    """
    Clean HTML by removing common email client artifacts.

    Args:
        html: HTML content

    Returns:
        Cleaned HTML
    """
    if not html:
        return ""

    # Remove common email client tracking pixels
    html = re.sub(
        r"<img[^>]*src=['\"][^'\"]*tracking[^'\"]*['\"][^>]*>",
        "",
        html,
        flags=re.IGNORECASE,
    )

    # Remove mailto: links (they'll be in text anyway)
    html = re.sub(r"<a[^>]*href=['\"]mailto:[^'\"]*['\"][^>]*>(.*?)</a>", r"\1", html, flags=re.IGNORECASE)

    return html
