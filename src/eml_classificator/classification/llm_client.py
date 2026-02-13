"""
LLM Client abstraction layer for Phase 3 classification.

Provides unified interface for multiple LLM providers:
- Ollama (self-hosted models: DeepSeek, Qwen, Llama, etc.)
- OpenAI API (GPT-4o, etc.)
- DeepSeek API (OpenAI-compatible)
- OpenRouter (multiple models via unified API)

Key features:
- Function calling / tool use support
- Retry logic with exponential backoff
- Token usage tracking
- Timeout handling
- Provider detection and format adaptation
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any,List, Optional, Tuple
from dataclasses import dataclass
import structlog

# LLM client imports
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from openai import OpenAI, OpenAIError, APITimeoutError, APIConnectionError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from eml_classificator.classification.tool_definitions import (
    get_classification_tool_definition,
    format_tool_call_for_provider
)
from eml_classificator.config import settings


logger = structlog.get_logger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class LLMResponse:
    """
    Unified LLM response structure.

    Contains the raw tool call result and metadata about the request.
    """
    # Tool call result (parsed JSON)
    tool_result: Dict[str, Any]

    # Metadata
    model: str
    provider: str
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    tokens_total: Optional[int] = None
    latency_ms: int = 0
    finish_reason: str = "unknown"

    # Raw response for debugging
    raw_response: Optional[Any] = None


# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================

class LLMClient(ABC):
    """
    Abstract base class for LLM clients.

    All concrete implementations must provide the classify() method
    that accepts a prompt and returns structured classification results
    via tool calling.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout_seconds: int = 120,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.logger = logger.bind(
            llm_client=self.__class__.__name__,
            model=model
        )

    @abstractmethod
    def classify(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_definitions: Optional[List[Dict[str, Any]]] = None
    ) -> LLMResponse:
        """
        Perform classification using LLM with tool calling.

        Args:
            system_prompt: System instructions
            user_prompt: User content (email data + candidates)
            tool_definitions: Tool definitions for function calling

        Returns:
            LLMResponse with tool call result and metadata

        Raises:
            Exception: On unrecoverable errors (after retries)
        """
        pass

    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute function with exponential backoff retry logic.

        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function

        Returns:
            Function result

        Raises:
            Last exception encountered after all retries exhausted
        """
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.error(
                        "llm_call_failed_after_retries",
                        error=str(e),
                        attempts=self.max_retries
                    )
                    raise

                wait_time = self.retry_delay * (2 ** attempt)
                self.logger.warning(
                    "llm_call_retry",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e),
                    wait_seconds=wait_time
                )
                time.sleep(wait_time)


# ============================================================================
# OLLAMA CLIENT
# ============================================================================

class OllamaClient(LLMClient):
    """
    Ollama client for self-hosted models.

    Supports models like:
    - deepseek-r1:7b, deepseek-r1:14b, deepseek-r1:32b
    - qwen2.5:7b, qwen2.5:14b, qwen2.5:32b
    - llama3.1:8b, llama3.3:70b
    - mistral, mixtral, etc.

    Requires Ollama running locally or accessible via base_url.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "ollama library not installed. Install with: pip install ollama"
            )

        super().__init__(model=model, **kwargs)
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)

        # Check model availability
        try:
            models = self.client.list()
            available_models = [m.model for m in models.get('models', [])]
            if model not in available_models:
                self.logger.warning(
                    "ollama_model_not_found",
                    model=model,
                    available_models=available_models
                )
        except Exception as e:
            self.logger.warning("ollama_model_check_failed", error=str(e))

    def classify(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_definitions: Optional[List[Dict[str, Any]]] = None
    ) -> LLMResponse:
        """
        Classify using Ollama with tool calling.

        Ollama supports tool use via the tools parameter (similar to OpenAI format).
        """
        start_time = time.time()

        if tool_definitions is None:
            tool_definitions = [get_classification_tool_definition()]

        def _call_ollama():
            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Call Ollama with tools
            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=tool_definitions,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            )

            return response

        try:
            response = self._retry_with_backoff(_call_ollama)

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract tool call
            message = response.get('message', {})
            tool_calls = message.get('tool_calls', [])

            if not tool_calls:
                raise ValueError("No tool calls in Ollama response")

            # Get first tool call (should be classify_email)
            tool_call = tool_calls[0]
            function_args = tool_call.get('function', {}).get('arguments', {})

            # Parse if string
            if isinstance(function_args, str):
                function_args = json.loads(function_args)

            # Extract token usage (if available)
            tokens_input = response.get('prompt_eval_count')
            tokens_output = response.get('eval_count')
            tokens_total = (tokens_input or 0) + (tokens_output or 0) if tokens_input and tokens_output else None

            return LLMResponse(
                tool_result=function_args,
                model=self.model,
                provider="ollama",
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                tokens_total=tokens_total,
                latency_ms=latency_ms,
                finish_reason=response.get('done_reason', 'stop'),
                raw_response=response
            )

        except Exception as e:
            self.logger.error(
                "ollama_classification_failed",
                error=str(e),
                model=self.model
            )
            raise


# ============================================================================
# OPENAI-COMPATIBLE CLIENT
# ============================================================================

class OpenAICompatibleClient(LLMClient):
    """
    OpenAI-compatible client for multiple providers.

    Works with:
    - OpenAI API (api.openai.com)
    - DeepSeek API (api.deepseek.com) - OpenAI-compatible
    - OpenRouter (openrouter.ai/api/v1)
    - Any other OpenAI-compatible endpoint

    Uses structured outputs via function calling.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        provider_name: str = "openai",
        **kwargs
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai library not installed. Install with: pip install openai"
            )

        super().__init__(model=model, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self.provider_name = provider_name

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.timeout_seconds
        )

    def classify(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_definitions: Optional[List[Dict[str, Any]]] = None
    ) -> LLMResponse:
        """
        Classify using OpenAI-compatible API with function calling.
        """
        start_time = time.time()

        if tool_definitions is None:
            tool_definitions = [get_classification_tool_definition()]

        def _call_openai():
            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Call with tools
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tool_definitions,
                tool_choice={"type": "function", "function": {"name": "classify_email"}},
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response

        try:
            response = self._retry_with_backoff(_call_openai)

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract tool call
            message = response.choices[0].message
            tool_calls = message.tool_calls

            if not tool_calls:
                raise ValueError("No tool calls in OpenAI response")

            # Get first tool call
            tool_call = tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)

            # Extract token usage
            usage = response.usage
            tokens_input = usage.prompt_tokens if usage else None
            tokens_output = usage.completion_tokens if usage else None
            tokens_total = usage.total_tokens if usage else None

            return LLMResponse(
                tool_result=function_args,
                model=self.model,
                provider=self.provider_name,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                tokens_total=tokens_total,
                latency_ms=latency_ms,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response
            )

        except (APITimeoutError, APIConnectionError) as e:
            self.logger.error(
                "openai_api_error",
                error=str(e),
                error_type=type(e).__name__,
                model=self.model,
                provider=self.provider_name
            )
            raise

        except Exception as e:
            self.logger.error(
                "openai_classification_failed",
                error=str(e),
                model=self.model,
                provider=self.provider_name
            )
            raise


# ============================================================================
# CLIENT FACTORY
# ============================================================================

def create_llm_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **override_kwargs
) -> LLMClient:
    """
    Factory function to create appropriate LLM client based on configuration.

    Priority order for configuration:
    1. Explicit parameters passed to this function
    2. Settings from config

    Args:
        provider: Provider name ("ollama", "openai", "deepseek", "openrouter")
        model: Model name (provider-specific)
        **override_kwargs: Override any client parameters

    Returns:
        Configured LLMClient instance

    Raises:
        ValueError: If provider is unknown or required dependencies missing
    """
    # Use settings as defaults
    provider = provider or settings.llm_provider
    model = model or settings.llm_model

    # Extract client parameters
    client_params = {
        "temperature": override_kwargs.get("temperature", settings.llm_temperature),
        "max_tokens": override_kwargs.get("max_tokens", settings.llm_max_tokens),
        "timeout_seconds": override_kwargs.get("timeout_seconds", settings.llm_timeout_seconds),
        "max_retries": override_kwargs.get("max_retries", settings.llm_max_retries),
        "retry_delay": override_kwargs.get("retry_delay", settings.llm_retry_delay_seconds),
    }

    logger.info(
        "creating_llm_client",
        provider=provider,
        model=model,
        temperature=client_params["temperature"]
    )

    # Ollama
    if provider == "ollama":
        return OllamaClient(
            model=model,
            base_url=override_kwargs.get("base_url", settings.llm_api_base_url),
            **client_params
        )

    # OpenAI
    elif provider == "openai":
        api_key = override_kwargs.get("api_key", settings.llm_api_key)
        if not api_key:
            raise ValueError("OpenAI API key required (set LLM_API_KEY env var)")

        return OpenAICompatibleClient(
            model=model,
            api_key=api_key,
            base_url=override_kwargs.get("base_url", "https://api.openai.com/v1"),
            provider_name="openai",
            **client_params
        )

    # DeepSeek API (OpenAI-compatible)
    elif provider == "deepseek":
        api_key = override_kwargs.get("api_key", settings.llm_api_key)
        if not api_key:
            raise ValueError("DeepSeek API key required (set LLM_API_KEY env var)")

        return OpenAICompatibleClient(
            model=model,
            api_key=api_key,
            base_url=override_kwargs.get("base_url", "https://api.deepseek.com"),
            provider_name="deepseek",
            **client_params
        )

    # OpenRouter (OpenAI-compatible)
    elif provider == "openrouter":
        api_key = override_kwargs.get("api_key", settings.llm_api_key)
        if not api_key:
            raise ValueError("OpenRouter API key required (set LLM_API_KEY env var)")

        return OpenAICompatibleClient(
            model=model,
            api_key=api_key,
            base_url=override_kwargs.get("base_url", "https://openrouter.ai/api/v1"),
            provider_name="openrouter",
            **client_params
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Supported: ollama, openai, deepseek, openrouter"
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_model_string(model_string: str) -> Tuple[str, str]:
    """
    Parse model string in format "provider/model-name".

    Examples:
        "ollama/deepseek-r1:7b" → ("ollama", "deepseek-r1:7b")
        "openai/gpt-4o" → ("openai", "gpt-4o")
        "deepseek-r1:7b" → (None, "deepseek-r1:7b")  # No provider prefix

    Args:
        model_string: Model specification

    Returns:
        (provider, model_name) tuple. provider is None if no prefix.
    """
    if "/" in model_string:
        parts = model_string.split("/", 1)
        return parts[0], parts[1]
    else:
        return None, model_string


def create_llm_client_from_model_string(
    model_string: str,
    **override_kwargs
) -> LLMClient:
    """
    Create LLM client from model string (e.g., "ollama/deepseek-r1:7b").

    This is the main entry point for creating clients with flexible configuration.

    Args:
        model_string: Model specification (may include provider prefix)
        **override_kwargs: Override any client parameters

    Returns:
        Configured LLMClient instance
    """
    provider_prefix, model_name = parse_model_string(model_string)

    # If provider specified in model string, use it; otherwise use config
    provider = provider_prefix if provider_prefix else settings.llm_provider

    return create_llm_client(
        provider=provider,
        model=model_name,
        **override_kwargs
    )
