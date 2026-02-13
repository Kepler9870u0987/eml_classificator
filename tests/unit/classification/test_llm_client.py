"""
Unit tests for LLM client abstraction layer.

Tests client initialization, model string parsing, retry logic,
and mock LLM responses without requiring actual API calls.
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import asdict

from eml_classificator.classification.llm_client import (
    LLMResponse,
    LLMClient,
    OllamaClient,
    OpenAICompatibleClient,
    parse_model_string,
    create_llm_client,
    create_llm_client_from_model_string,
    OLLAMA_AVAILABLE,
    OPENAI_AVAILABLE
)


class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_llm_response_creation(self):
        """Test creating LLMResponse with all fields."""
        response = LLMResponse(
            tool_result={"topics": [], "sentiment": {"value": "neutral"}},
            model="deepseek-r1:7b",
            provider="ollama",
            tokens_input=100,
            tokens_output=50,
            tokens_total=150,
            latency_ms=2500,
            finish_reason="stop"
        )

        assert response.model == "deepseek-r1:7b"
        assert response.provider == "ollama"
        assert response.tokens_total == 150
        assert response.latency_ms == 2500

    def test_llm_response_minimal(self):
        """Test LLMResponse with minimal required fields."""
        response = LLMResponse(
            tool_result={},
            model="gpt-4o",
            provider="openai"
        )

        assert response.tool_result == {}
        assert response.tokens_input is None
        assert response.latency_ms == 0


class TestModelStringParsing:
    """Test model string parsing utilities."""

    def test_parse_model_string_with_provider(self):
        """Test parsing model string with provider prefix."""
        provider, model = parse_model_string("ollama/deepseek-r1:7b")
        assert provider == "ollama"
        assert model == "deepseek-r1:7b"

    def test_parse_model_string_without_provider(self):
        """Test parsing model string without provider prefix."""
        provider, model = parse_model_string("deepseek-r1:7b")
        assert provider is None
        assert model == "deepseek-r1:7b"

    def test_parse_model_string_openai(self):
        """Test parsing OpenAI model string."""
        provider, model = parse_model_string("openai/gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_parse_model_string_with_version(self):
        """Test parsing model string with version tag."""
        provider, model = parse_model_string("ollama/qwen2.5:32b")
        assert provider == "ollama"
        assert model == "qwen2.5:32b"

    def test_parse_model_string_multiple_slashes(self):
        """Test parsing model string with multiple slashes (edge case)."""
        provider, model = parse_model_string("openrouter/anthropic/claude-3-opus")
        assert provider == "openrouter"
        assert model == "anthropic/claude-3-opus"


class TestOllamaClient:
    """Test OllamaClient implementation."""

    @pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="ollama library not installed")
    @patch('eml_classificator.classification.llm_client.ollama.Client')
    def test_ollama_client_initialization(self, mock_client_class):
        """Test OllamaClient initialization."""
        mock_client = Mock()
        mock_client.list.return_value = {'models': [
            Mock(model='deepseek-r1:7b'),
            Mock(model='qwen2.5:7b')
        ]}
        mock_client_class.return_value = mock_client

        client = OllamaClient(
            model="deepseek-r1:7b",
            base_url="http://localhost:11434"
        )

        assert client.model == "deepseek-r1:7b"
        assert client.base_url == "http://localhost:11434"
        mock_client_class.assert_called_once_with(host="http://localhost:11434")

    @pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="ollama library not installed")
    @patch('eml_classificator.classification.llm_client.ollama.Client')
    def test_ollama_client_classify_success(self, mock_client_class):
        """Test successful Ollama classification."""
        # Mock Ollama response
        mock_response = {
            'message': {
                'tool_calls': [
                    {
                        'function': {
                            'name': 'classify_email',
                            'arguments': {
                                'dictionary_version': 1,
                                'topics': [
                                    {
                                        'label_id': 'FATTURAZIONE',
                                        'confidence': 0.9,
                                        'keywords_in_text': [
                                            {'candidate_id': 'abc123', 'lemma': 'fattura', 'count': 3}
                                        ],
                                        'evidence': [
                                            {'quote': 'ho ricevuto la fattura'}
                                        ]
                                    }
                                ],
                                'sentiment': {'value': 'negative', 'confidence': 0.85},
                                'priority': {'value': 'high', 'confidence': 0.8, 'signals': []},
                                'customer_status': {'value': 'existing', 'confidence': 1.0}
                            }
                        }
                    }
                ]
            },
            'prompt_eval_count': 150,
            'eval_count': 100,
            'done_reason': 'stop'
        }

        mock_client = Mock()
        mock_client.chat.return_value = mock_response
        mock_client.list.return_value = {'models': [Mock(model='deepseek-r1:7b')]}
        mock_client_class.return_value = mock_client

        client = OllamaClient(model="deepseek-r1:7b")

        response = client.classify(
            system_prompt="You are a classifier",
            user_prompt="Classify this email"
        )

        assert isinstance(response, LLMResponse)
        assert response.provider == "ollama"
        assert response.model == "deepseek-r1:7b"
        assert response.tokens_input == 150
        assert response.tokens_output == 100
        assert response.tokens_total == 250
        assert 'topics' in response.tool_result

    @pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="ollama library not installed")
    @patch('eml_classificator.classification.llm_client.ollama.Client')
    def test_ollama_client_classify_no_tool_calls(self, mock_client_class):
        """Test Ollama classification when no tool calls returned."""
        mock_response = {
            'message': {
                'content': 'Some text response',
                'tool_calls': []  # Empty tool calls
            }
        }

        mock_client = Mock()
        mock_client.chat.return_value = mock_response
        mock_client.list.return_value = {'models': [Mock(model='deepseek-r1:7b')]}
        mock_client_class.return_value = mock_client

        client = OllamaClient(model="deepseek-r1:7b")

        with pytest.raises(ValueError, match="No tool calls"):
            client.classify(
                system_prompt="System",
                user_prompt="User"
            )

    @pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="ollama library not installed")
    @patch('eml_classificator.classification.llm_client.ollama.Client')
    def test_ollama_client_retry_logic(self, mock_client_class):
        """Test Ollama client retry with exponential backoff."""
        mock_client = Mock()
        # First two calls fail, third succeeds
        mock_client.chat.side_effect = [
            Exception("Connection error"),
            Exception("Timeout"),
            {
                'message': {
                    'tool_calls': [{
                        'function': {
                            'name': 'classify_email',
                            'arguments': {'dictionary_version': 1, 'topics': []}
                        }
                    }]
                },
                'prompt_eval_count': 100,
                'eval_count': 50
            }
        ]
        mock_client.list.return_value = {'models': [Mock(model='test-model')]}
        mock_client_class.return_value = mock_client

        client = OllamaClient(
            model="test-model",
            max_retries=3,
            retry_delay=0.01  # Fast retry for testing
        )

        response = client.classify(
            system_prompt="System",
            user_prompt="User"
        )

        assert isinstance(response, LLMResponse)
        assert mock_client.chat.call_count == 3


class TestOpenAICompatibleClient:
    """Test OpenAICompatibleClient implementation."""

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai library not installed")
    @patch('eml_classificator.classification.llm_client.OpenAI')
    def test_openai_client_initialization(self, mock_openai_class):
        """Test OpenAICompatibleClient initialization."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        client = OpenAICompatibleClient(
            model="gpt-4o",
            api_key="sk-test123",
            base_url="https://api.openai.com/v1",
            provider_name="openai"
        )

        assert client.model == "gpt-4o"
        assert client.api_key == "sk-test123"
        assert client.provider_name == "openai"
        mock_openai_class.assert_called_once()

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai library not installed")
    @patch('eml_classificator.classification.llm_client.OpenAI')
    def test_openai_client_classify_success(self, mock_openai_class):
        """Test successful OpenAI classification."""
        # Mock OpenAI response structure
        mock_tool_call = Mock()
        mock_tool_call.function.name = "classify_email"
        mock_tool_call.function.arguments = json.dumps({
            'dictionary_version': 1,
            'topics': [],
            'sentiment': {'value': 'neutral', 'confidence': 0.9},
            'priority': {'value': 'medium', 'confidence': 0.7, 'signals': []},
            'customer_status': {'value': 'unknown', 'confidence': 0.5}
        })

        mock_message = Mock()
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_usage = Mock()
        mock_usage.prompt_tokens = 200
        mock_usage.completion_tokens = 150
        mock_usage.total_tokens = 350

        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        client = OpenAICompatibleClient(
            model="gpt-4o",
            api_key="sk-test123"
        )

        response = client.classify(
            system_prompt="System",
            user_prompt="User"
        )

        assert isinstance(response, LLMResponse)
        assert response.provider == "openai"
        assert response.tokens_input == 200
        assert response.tokens_output == 150
        assert response.tokens_total == 350

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai library not installed")
    @patch('eml_classificator.classification.llm_client.OpenAI')
    def test_openai_client_classify_no_tool_calls(self, mock_openai_class):
        """Test OpenAI classification when no tool calls returned."""
        mock_message = Mock()
        mock_message.tool_calls = None

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        client = OpenAICompatibleClient(
            model="gpt-4o",
            api_key="sk-test123"
        )

        with pytest.raises(ValueError, match="No tool calls"):
            client.classify(
                system_prompt="System",
                user_prompt="User"
            )

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai library not installed")
    @patch('eml_classificator.classification.llm_client.OpenAI')
    def test_openai_client_deepseek_provider(self, mock_openai_class):
        """Test DeepSeek API client (OpenAI-compatible)."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        client = OpenAICompatibleClient(
            model="deepseek-chat",
            api_key="sk-deepseek123",
            base_url="https://api.deepseek.com",
            provider_name="deepseek"
        )

        assert client.provider_name == "deepseek"
        assert client.base_url == "https://api.deepseek.com"


class TestClientFactory:
    """Test LLM client factory functions."""

    @patch('eml_classificator.classification.llm_client.OllamaClient')
    def test_create_llm_client_ollama(self, mock_ollama_class):
        """Test factory creates OllamaClient."""
        mock_client = Mock(spec=OllamaClient)
        mock_ollama_class.return_value = mock_client

        client = create_llm_client(
            provider="ollama",
            model="deepseek-r1:7b"
        )

        assert mock_ollama_class.called
        # Verify it was called with correct model
        call_kwargs = mock_ollama_class.call_args[1]
        assert call_kwargs.get('model') == "deepseek-r1:7b" or mock_ollama_class.call_args[0][0] == "deepseek-r1:7b"

    @patch('eml_classificator.classification.llm_client.OpenAICompatibleClient')
    def test_create_llm_client_openai(self, mock_openai_class):
        """Test factory creates OpenAI client."""
        mock_client = Mock(spec=OpenAICompatibleClient)
        mock_openai_class.return_value = mock_client

        client = create_llm_client(
            provider="openai",
            model="gpt-4o",
            api_key="sk-test123"
        )

        assert mock_openai_class.called

    @patch('eml_classificator.classification.llm_client.OpenAICompatibleClient')
    def test_create_llm_client_deepseek(self, mock_openai_class):
        """Test factory creates DeepSeek client."""
        mock_client = Mock(spec=OpenAICompatibleClient)
        mock_openai_class.return_value = mock_client

        client = create_llm_client(
            provider="deepseek",
            model="deepseek-chat",
            api_key="sk-deepseek123"
        )

        assert mock_openai_class.called
        # Check that provider_name is set to deepseek
        call_kwargs = mock_openai_class.call_args[1]
        assert call_kwargs.get('provider_name') == "deepseek"

    @patch('eml_classificator.classification.llm_client.OpenAICompatibleClient')
    def test_create_llm_client_openrouter(self, mock_openai_class):
        """Test factory creates OpenRouter client."""
        mock_client = Mock(spec=OpenAICompatibleClient)
        mock_openai_class.return_value = mock_client

        client = create_llm_client(
            provider="openrouter",
            model="anthropic/claude-3-opus",
            api_key="sk-or-test123"
        )

        assert mock_openai_class.called
        call_kwargs = mock_openai_class.call_args[1]
        assert call_kwargs.get('provider_name') == "openrouter"

    def test_create_llm_client_unknown_provider(self):
        """Test factory raises error for unknown provider."""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_client(
                provider="unknown_provider",
                model="some-model"
            )

    def test_create_llm_client_missing_api_key(self):
        """Test factory raises error when API key missing for cloud providers."""
        with pytest.raises(ValueError, match="API key required"):
            create_llm_client(
                provider="openai",
                model="gpt-4o"
                # No api_key provided
            )

    @patch('eml_classificator.classification.llm_client.create_llm_client')
    def test_create_llm_client_from_model_string_with_provider(self, mock_create):
        """Test creating client from model string with provider."""
        mock_create.return_value = Mock()

        client = create_llm_client_from_model_string(
            "ollama/deepseek-r1:7b"
        )

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs['provider'] == "ollama"
        assert call_kwargs['model'] == "deepseek-r1:7b"

    @patch('eml_classificator.classification.llm_client.create_llm_client')
    @patch('eml_classificator.classification.llm_client.settings')
    def test_create_llm_client_from_model_string_without_provider(self, mock_settings, mock_create):
        """Test creating client from model string without provider (uses config)."""
        mock_settings.llm_provider = "ollama"
        mock_create.return_value = Mock()

        client = create_llm_client_from_model_string(
            "qwen2.5:32b"
        )

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs['provider'] == "ollama"  # From settings
        assert call_kwargs['model'] == "qwen2.5:32b"


class TestRetryLogic:
    """Test retry logic in LLM clients."""

    def test_retry_with_backoff_success_first_try(self):
        """Test retry logic when first attempt succeeds."""
        mock_func = Mock(return_value="success")

        # Create a mock client
        client = Mock(spec=LLMClient)
        client.max_retries = 3
        client.retry_delay = 0.01
        client.logger = Mock()

        # Call the actual retry method
        result = LLMClient._retry_with_backoff(client, mock_func)

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_with_backoff_success_after_retries(self):
        """Test retry logic succeeds after failures."""
        mock_func = Mock(side_effect=[
            Exception("Error 1"),
            Exception("Error 2"),
            "success"
        ])

        client = Mock(spec=LLMClient)
        client.max_retries = 3
        client.retry_delay = 0.01
        client.logger = Mock()

        result = LLMClient._retry_with_backoff(client, mock_func)

        assert result == "success"
        assert mock_func.call_count == 3

    def test_retry_with_backoff_all_failures(self):
        """Test retry logic exhausts all retries."""
        mock_func = Mock(side_effect=Exception("Persistent error"))

        client = Mock(spec=LLMClient)
        client.max_retries = 3
        client.retry_delay = 0.01
        client.logger = Mock()

        with pytest.raises(Exception, match="Persistent error"):
            LLMClient._retry_with_backoff(client, mock_func)

        assert mock_func.call_count == 3


class TestCustomClientParameters:
    """Test custom client parameter overrides."""

    @pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="ollama library not installed")
    @patch('eml_classificator.classification.llm_client.ollama.Client')
    def test_custom_temperature(self, mock_client_class):
        """Test setting custom temperature."""
        mock_client = Mock()
        mock_client.list.return_value = {'models': []}
        mock_client_class.return_value = mock_client

        client = OllamaClient(
            model="test-model",
            temperature=0.7
        )

        assert client.temperature == 0.7

    @pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="ollama library not installed")
    @patch('eml_classificator.classification.llm_client.ollama.Client')
    def test_custom_max_tokens(self, mock_client_class):
        """Test setting custom max tokens."""
        mock_client = Mock()
        mock_client.list.return_value = {'models': []}
        mock_client_class.return_value = mock_client

        client = OllamaClient(
            model="test-model",
            max_tokens=4096
        )

        assert client.max_tokens == 4096

    @pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="ollama library not installed")
    @patch('eml_classificator.classification.llm_client.ollama.Client')
    def test_custom_timeout(self, mock_client_class):
        """Test setting custom timeout."""
        mock_client = Mock()
        mock_client.list.return_value = {'models': []}
        mock_client_class.return_value = mock_client

        client = OllamaClient(
            model="test-model",
            timeout_seconds=300
        )

        assert client.timeout_seconds == 300
