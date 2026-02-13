"""
Tool definitions for LLM function calling.

Defines the function calling schema that the LLM will use to return
structured classification results.

Based on OpenAI function calling format, compatible with:
- OpenAI API (GPT-4o, etc.)
- DeepSeek API
- Ollama (with tool calling support)
- OpenRouter

References:
- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
- Brainstorming v2 section 3.2: JSON Schema optimization (PARSE)
"""

from typing import Dict, Any, List


# ============================================================================
# SCHEMA VERSION
# ============================================================================

TOOL_CALLING_VERSION = "openai-tool-calling-2026"
SCHEMA_VERSION = "json-schema-v2.2"


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

def get_classification_tool_definition() -> Dict[str, Any]:
    """
   Get the main classification tool definition for function calling.

    Returns the complete schema for email classification with all constraints:
    - Topics: 1-5 topics from closed taxonomy
    - Keywords: 1-15 per topic, MUST reference candidate_id from input
    - Evidence: 1-2 per topic, max 200 chars per quote
    - Sentiment: positive/neutral/negative
    - Priority: low/medium/high/urgent (with justification signals)
    - Customer status: new/existing/unknown

    This schema is optimized following PARSE principles (v2 section 3.2):
    - Reduced maxItems for better LLM adherence
    - Clear descriptions for each field
    - Strict minItems to ensure completeness
    - String length limits to avoid verbosity
    """
    return {
        "type": "function",
        "function": {
            "name": "classify_email",
            "description": "Classify an email for customer service triage. Extract topics, sentiment, priority signals, and relevant keywords from the candidate list.",

            "parameters": {
                "type": "object",
                "required": ["dictionary_version", "topics", "sentiment", "priority", "customer_status"],
                "additionalProperties": False,

                "properties": {
                    "dictionary_version": {
                        "type": "integer",
                        "description": "Version of keyword dictionary used (from input)"
                    },

                    "topics": {
                        "type": "array",
                        "description": "Multi-label topic classification. Select 1-5 topics from the allowed taxonomy. If content is unclear or doesn't fit, use UNKNOWN_TOPIC.",
                        "minItems": 1,
                        "maxItems": 5,

                        "items": {
                            "type": "object",
                            "required": ["label_id", "confidence", "keywords_in_text", "evidence"],
                            "additionalProperties": False,

                            "properties": {
                                "label_id": {
                                    "type": "string",
                                    "enum": [
                                        "FATTURAZIONE",
                                        "ASSISTENZA_TECNICA",
                                        "RECLAMO",
                                        "INFO_COMMERCIALI",
                                        "DOCUMENTI",
                                        "APPUNTAMENTO",
                                        "CONTRATTO",
                                        "GARANZIA",
                                        "SPEDIZIONE",
                                        "UNKNOWN_TOPIC"
                                    ],
                                    "description": "Topic label from closed taxonomy"
                                },

                                "confidence": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "description": "Confidence score for this topic assignment (0.0-1.0)"
                                },

                                "keywords_in_text": {
                                    "type": "array",
                                    "description": "Keywords ONLY from the input candidate list that support this topic. Each keyword MUST have a valid candidate_id from the provided candidates.",
                                    "minItems": 1,
                                    "maxItems": 15,

                                    "items": {
                                        "type": "object",
                                        "required": ["candidate_id", "lemma", "count"],
                                        "additionalProperties": False,

                                        "properties": {
                                            "candidate_id": {
                                                "type": "string",
                                                "description": "MUST match a candidate_id from the input candidate list. Do not invent IDs."
                                            },
                                            "lemma": {
                                                "type": "string",
                                                "description": "Lemmatized form of the keyword"
                                            },
                                            "count": {
                                                "type": "integer",
                                                "minimum": 1,
                                                "description": "How many times this keyword appears in the email"
                                            },
                                            "spans": {
                                                "type": "array",
                                                "description": "Optional: [[start, end], ...] character positions in text where keyword appears",
                                                "items": {
                                                    "type": "array",
                                                    "minItems": 2,
                                                    "maxItems": 2,
                                                    "items": {"type": "integer"}
                                                }
                                            }
                                        }
                                    }
                                },

                                "evidence": {
                                    "type": "array",
                                    "description": "Text quotes from the email that justify this topic assignment. Provide 1-2 short, relevant quotes.",
                                    "minItems": 1,
                                    "maxItems": 2,

                                    "items": {
                                        "type": "object",
                                        "required": ["quote"],
                                        "additionalProperties": False,

                                        "properties": {
                                            "quote": {
                                                "type": "string",
                                                "maxLength": 200,
                                                "description": "Exact quote from email text (max 200 characters)"
                                            },
                                            "span": {
                                                "type": "array",
                                                "description": "Optional: [start, end] character position of quote in text",
                                                "minItems": 2,
                                                "maxItems": 2,
                                                "items": {"type": "integer"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },

                    "sentiment": {
                        "type": "object",
                        "description": "Sentiment analysis of the email",
                        "required": ["value", "confidence"],
                        "additionalProperties": False,

                        "properties": {
                            "value": {
                                "type": "string",
                                "enum": ["positive", "neutral", "negative"],
                                "description": "Overall email sentiment: positive (satisfaction, gratitude), neutral (informational), negative (complaints, frustration)"
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Confidence in sentiment classification (0.0-1.0)"
                            }
                        }
                    },

                    "priority": {
                        "type": "object",
                        "description": "Priority assessment for triage. Will be adjusted by rule-based scoring.",
                        "required": ["value", "confidence", "signals"],
                        "additionalProperties": False,

                        "properties": {
                            "value": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "urgent"],
                                "description": "Priority level: urgent (immediate action, SLA breach, blocking issue), high (same-day response), medium (1-2 days), low (routine, no time pressure)"
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Confidence in priority assessment (0.0-1.0)"
                            },
                            "signals": {
                                "type": "array",
                                "description": "Phrases or keywords from the email that justify this priority level (for audit)",
                                "maxItems": 10,
                                "items": {"type": "string"}
                            }
                        }
                    },

                    "customer_status": {
                        "type": "object",
                        "description": "Customer status. Note: This will be overridden by CRM lookup in post-processing. Provide your best guess based on text signals only.",
                        "required": ["value", "confidence"],
                        "additionalProperties": False,

                        "properties": {
                            "value": {
                                "type": "string",
                                "enum": ["new", "existing", "unknown"],
                                "description": "Customer status: new (first contact), existing (has relationship/contract), unknown (cannot determine from text)"
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Confidence in status determination based on text signals (0.0-1.0)"
                            }
                        }
                    }
                }
            }
        }
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_tool_definitions_list() -> List[Dict[str, Any]]:
    """
    Get the list of all tool definitions for LLM function calling.

    Currently only returns the main classification tool, but structured
    to allow future expansion with additional tools.
    """
    return [get_classification_tool_definition()]


def get_tool_choice() -> str:
    """
    Get the tool_choice parameter for LLM API calls.

    Returns "required" (or equivalent) to force the LLM to use the classification tool.
    This ensures structured output even if the LLM tries to respond in plain text.
    """
    # Note: Some providers use "required", others use "auto" or {"type": "function", "function": {...}}
    # The LLM client should adapt this based on the provider
    return "required"


def format_tool_call_for_provider(provider: str) -> Dict[str, Any]:
    """
    Format tool calling parameters for a specific LLM provider.

    Args:
        provider: Provider name ("openai", "ollama", "deepseek", "openrouter")

    Returns:
        Dict with provider-specific formatting for tool calling
    """
    tools = get_tool_definitions_list()

    if provider in ["openai", "deepseek", "openrouter"]:
        # OpenAI-compatible format
        return {
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "classify_email"}}
        }

    elif provider == "ollama":
        # Ollama format (similar to OpenAI but may have differences)
        return {
            "tools": tools,
            "tool_choice": "required"  # or "auto" depending on Ollama version
        }

    else:
        raise ValueError(f"Unknown provider: {provider}")


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def get_topics_enum() -> List[str]:
    """Get the list of valid topic labels for validation."""
    return  [ "FATTURAZIONE",
        "ASSISTENZA_TECNICA",
        "RECLAMO",
        "INFO_COMMERCIALI",
        "DOCUMENTI",
        "APPUNTAMENTO",
        "CONTRATTO",
        "GARANZIA",
        "SPEDIZIONE",
        "UNKNOWN_TOPIC"
    ]


def get_sentiment_enum() -> List[str]:
    """Get the list of valid sentiment values for validation."""
    return ["positive", "neutral", "negative"]


def get_priority_enum() -> List[str]:
    """Get the list of valid priority values for validation."""
    return ["low", "medium", "high", "urgent"]


def get_customer_status_enum() -> List[str]:
    """Get the list of valid customer status values for validation."""
    return ["new", "existing", "unknown"]
