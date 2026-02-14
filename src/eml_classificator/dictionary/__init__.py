"""Dictionary management module.

Provides regex generation and future dictionary management functionality.
"""

from .regex_generator import generate_safe_regex, test_regex_pattern

__all__ = ["generate_safe_regex", "test_regex_pattern"]
