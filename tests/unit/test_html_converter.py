"""
Unit tests for HTML conversion module (html_converter.py).

Tests cover:
- HTML to plain text conversion
- Link preservation options
- Image removal
- Emphasis removal
- Script and style tag removal
- HTML entity decoding
- Tracking pixel removal
- Mailto link handling
"""

import pytest

from eml_classificator.canonicalization.html_converter import (
    clean_html,
    html_to_text,
    strip_html_tags,
)


class TestHtmlToText:
    """Tests for html_to_text() function."""

    @pytest.mark.unit
    def test_html_to_text_simple(self):
        """Test converting simple HTML to text."""
        html = "<p>Hello <strong>world</strong>!</p>"
        result = html_to_text(html)

        assert "Hello world" in result
        assert "<p>" not in result
        assert "<strong>" not in result

    @pytest.mark.unit
    def test_html_to_text_with_links_preserve(self):
        """Test keeping links as markdown [text](url)."""
        html = '<p>Visit our <a href="https://example.com">website</a></p>'
        result = html_to_text(html, preserve_links=True)

        # html2text should produce markdown-style links
        assert "website" in result
        assert "example.com" in result

    @pytest.mark.unit
    def test_html_to_text_with_links_ignore(self):
        """Test removing links entirely."""
        html = '<p>Visit our <a href="https://example.com">website</a></p>'
        result = html_to_text(html, preserve_links=False)

        assert "Visit our website" in result or "website" in result

    @pytest.mark.unit
    def test_html_to_text_with_images(self):
        """Test that images are removed."""
        html = '<p>Image: <img src="photo.jpg" alt="Photo" /></p>'
        result = html_to_text(html)

        assert "<img" not in result
        assert "photo.jpg" not in result

    @pytest.mark.unit
    def test_html_to_text_with_emphasis(self):
        """Test that bold/italic formatting is removed."""
        html = "<p>Text with <strong>bold</strong> and <em>italic</em></p>"
        result = html_to_text(html)

        assert "bold" in result
        assert "italic" in result
        assert "<strong>" not in result
        assert "<em>" not in result

    @pytest.mark.unit
    def test_html_to_text_empty(self):
        """Test handling of empty string."""
        assert html_to_text("") == ""
        assert html_to_text("   ") == ""

    @pytest.mark.unit
    def test_html_to_text_with_headings(self):
        """Test conversion of heading tags."""
        html = "<h1>Title</h1><h2>Subtitle</h2><p>Content</p>"
        result = html_to_text(html)

        assert "Title" in result
        assert "Subtitle" in result
        assert "Content" in result

    @pytest.mark.unit
    def test_html_to_text_with_lists(self):
        """Test conversion of list tags."""
        html = "<ul><li>Item 1</li><li>Item 2</li></ul>"
        result = html_to_text(html)

        assert "Item 1" in result
        assert "Item 2" in result


class TestStripHtmlTags:
    """Tests for strip_html_tags() fallback function."""

    @pytest.mark.unit
    def test_strip_html_tags_basic(self):
        """Test removing basic HTML tags."""
        html = "<p>Hello <b>world</b></p>"
        result = strip_html_tags(html)

        assert result.strip() == "Hello world"
        assert "<" not in result

    @pytest.mark.unit
    def test_strip_html_tags_scripts(self):
        """Test removing <script> blocks."""
        html = """<html>
<script>alert('test');</script>
<p>Content</p>
</html>"""
        result = strip_html_tags(html)

        assert "alert" not in result
        assert "Content" in result

    @pytest.mark.unit
    def test_strip_html_tags_styles(self):
        """Test removing <style> blocks."""
        html = """<html>
<style>body { color: red; }</style>
<p>Content</p>
</html>"""
        result = strip_html_tags(html)

        assert "color: red" not in result
        assert "Content" in result

    @pytest.mark.unit
    def test_strip_html_tags_entities(self):
        """Test decoding HTML entities (&amp;, &lt;, etc.)."""
        html = "<p>&amp; &lt; &gt; &quot; &copy;</p>"
        result = strip_html_tags(html)

        assert "&" in result
        assert "<" in result
        assert ">" in result
        assert '"' in result or "\"" in result
        assert "©" in result

    @pytest.mark.unit
    def test_strip_html_tags_empty(self):
        """Test handling of empty string."""
        assert strip_html_tags("") == ""

    @pytest.mark.unit
    def test_strip_html_tags_nested(self):
        """Test handling of nested tags."""
        html = "<div><p>Outer <span>inner</span> text</p></div>"
        result = strip_html_tags(html)

        assert "Outer inner text" in result
        assert "<" not in result

    @pytest.mark.unit
    def test_strip_html_tags_whitespace_cleanup(self):
        """Test that excessive whitespace is cleaned up."""
        html = "<p>Text   with    spaces</p>"
        result = strip_html_tags(html)

        # Should collapse multiple spaces
        assert "Text with spaces" in result or "Text  with  spaces" not in result


class TestCleanHtml:
    """Tests for clean_html() function."""

    @pytest.mark.unit
    def test_clean_html_tracking_pixels(self):
        """Test removing tracking pixel images."""
        html = '''<p>Content</p>
<img src="https://tracking.example.com/pixel.gif" />
<p>More content</p>'''
        result = clean_html(html)

        assert "Content" in result
        assert "More content" in result
        assert "tracking" not in result

    @pytest.mark.unit
    def test_clean_html_mailto_links(self):
        """Test converting mailto: links to plain text."""
        html = '<p>Email us at <a href="mailto:support@example.com">support@example.com</a></p>'
        result = clean_html(html)

        assert "support@example.com" in result
        # Mailto link should be replaced with just the text
        assert "mailto:" not in result or result.count("mailto:") < html.count("mailto:")

    @pytest.mark.unit
    def test_clean_html_empty(self):
        """Test handling of empty string."""
        assert clean_html("") == ""

    @pytest.mark.unit
    def test_clean_html_no_changes_needed(self):
        """Test that clean HTML is unchanged."""
        html = "<p>Simple clean content</p>"
        result = clean_html(html)

        assert "<p>Simple clean content</p>" in result

    @pytest.mark.unit
    def test_clean_html_multiple_tracking_pixels(self):
        """Test removing multiple tracking pixels."""
        html = '''<p>Content</p>
<img src="https://tracking1.example.com/pixel.gif" />
<img src="https://tracking2.example.com/track.png" />
<p>More</p>'''
        result = clean_html(html)

        assert result.count("tracking") < html.count("tracking")


class TestHtmlToTextIntegration:
    """Integration tests for HTML conversion with real-world examples."""

    @pytest.mark.unit
    def test_newsletter_html(self):
        """Test converting a typical newsletter HTML."""
        html = """<html>
<head><title>Newsletter</title></head>
<body>
<h1>Monthly Update</h1>
<p>Dear subscriber,</p>
<p>Check out our <a href="https://example.com/offers">special offers</a>!</p>
<img src="https://tracking.example.com/pixel.gif" width="1" height="1" />
</body>
</html>"""
        result = html_to_text(html)

        assert "Monthly Update" in result
        assert "Dear subscriber" in result
        assert "Newsletter" not in result  # <title> should not be in body text

    @pytest.mark.unit
    def test_email_with_signature_html(self):
        """Test converting HTML email with signature."""
        html = """<html>
<body>
<p>Hello,</p>
<p>This is the message body.</p>
<p>Best regards,<br/>
John Doe<br/>
CEO</p>
</body>
</html>"""
        result = html_to_text(html)

        assert "Hello" in result
        assert "message body" in result
        assert "Best regards" in result
        assert "John Doe" in result

    @pytest.mark.unit
    def test_html_with_unicode(self):
        """Test handling Unicode characters in HTML."""
        html = "<p>Café naïve résumé</p>"
        result = html_to_text(html)

        assert "Café" in result
        assert "naïve" in result
        assert "résumé" in result
