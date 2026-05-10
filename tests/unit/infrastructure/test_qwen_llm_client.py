"""Unit tests for QwenLLMClient.

All HTTP calls are mocked — no real DashScope API requests are made.
The mock chain mirrors the real langchain call path:

    ChatOpenAI(model=..., api_key=..., base_url=...)   ← patched
    llm.bind(temperature=X)                            → bound_llm (mock)
    bound_llm.invoke([SystemMessage, HumanMessage])    → AIMessage (mock)
    response.content                                   → str
    response.usage_metadata                            → {"input_tokens": N, "output_tokens": M}
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from ego_crawler.application.ports.llm_client import LLMResponse
from ego_crawler.infrastructure.qwen.qwen_llm_client import QwenLLMClient

_MODULE = "ego_crawler.infrastructure.qwen.qwen_llm_client"

_SAMPLE_TOOLS = [
    {"name": "web_search", "description": "Search", "parameters": {"query": "string"}},
    {"name": "navigate",   "description": "Navigate", "parameters": {"url": "string"}},
]

_THOUGHT_RESPONSE = (
    "THOUGHT: Надо проверить ещё один источник перед тем, как начать.\n"
    'ACTION: web_search | {"query": "ЕГЭ математика 2026"}'
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_ai_message(content: str, input_tokens: int = 100, output_tokens: int = 50):
    """Builds a minimal mock that looks like langchain AIMessage."""
    msg = MagicMock()
    msg.content = content
    msg.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    return msg


def _mock_llm_chain(mock_cls: MagicMock, content: str = "THOUGHT: OK",
                    input_tokens: int = 100, output_tokens: int = 50) -> MagicMock:
    """Wires mock_cls so that llm.bind(...).invoke(...) returns an AIMessage mock."""
    mock_llm = MagicMock(name="llm_instance")
    mock_bound = MagicMock(name="bound_llm")
    mock_llm.bind.return_value = mock_bound
    mock_bound.invoke.return_value = _make_ai_message(content, input_tokens, output_tokens)
    mock_cls.return_value = mock_llm
    return mock_llm


# ── Initialization ─────────────────────────────────────────────────────────────

class TestQwenLLMClientInit:
    def test_explicit_api_key_accepted(self):
        with patch(f"{_MODULE}.ChatOpenAI"):
            client = QwenLLMClient(api_key="sk-explicit")
        assert client is not None

    def test_env_api_key_accepted(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-from-env")
        with patch(f"{_MODULE}.ChatOpenAI"):
            client = QwenLLMClient()
        assert client is not None

    def test_no_api_key_raises_value_error(self, monkeypatch):
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
            QwenLLMClient()

    def test_explicit_key_takes_priority_over_env(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-from-env")
        with patch(f"{_MODULE}.ChatOpenAI") as mock_cls:
            QwenLLMClient(api_key="sk-explicit")
            _, kwargs = mock_cls.call_args
            assert kwargs["api_key"] == "sk-explicit"

    def test_default_model_is_qwen_plus(self):
        with patch(f"{_MODULE}.ChatOpenAI"):
            client = QwenLLMClient(api_key="sk-test")
        assert client._model == "qwen-plus"

    def test_custom_model_accepted(self):
        with patch(f"{_MODULE}.ChatOpenAI"):
            client = QwenLLMClient(api_key="sk-test", model="qwen-max")
        assert client._model == "qwen-max"

    def test_international_base_url_passed_to_chat_openai(self):
        with patch(f"{_MODULE}.ChatOpenAI") as mock_cls:
            QwenLLMClient(api_key="sk-test")
            _, kwargs = mock_cls.call_args
            assert "dashscope-intl" in kwargs["base_url"]


# ── generate — happy path ──────────────────────────────────────────────────────

class TestQwenLLMClientGenerate:
    @pytest.fixture
    def client_and_mock(self):
        with patch(f"{_MODULE}.ChatOpenAI") as mock_cls:
            mock_llm = _mock_llm_chain(mock_cls)
            client = QwenLLMClient(api_key="sk-test")
            yield client, mock_llm

    def _set_response(self, mock_llm, content="THOUGHT: OK",
                      input_tokens=100, output_tokens=50):
        mock_llm.bind.return_value.invoke.return_value = _make_ai_message(
            content, input_tokens, output_tokens
        )

    def test_returns_llm_response(self, client_and_mock):
        client, mock_llm = client_and_mock
        self._set_response(mock_llm, _THOUGHT_RESPONSE)

        result = client.generate("system", "user")

        assert isinstance(result, LLMResponse)

    def test_content_matches_api_response(self, client_and_mock):
        client, mock_llm = client_and_mock
        self._set_response(mock_llm, _THOUGHT_RESPONSE)

        result = client.generate("system", "user")

        assert result.content == _THOUGHT_RESPONSE

    def test_usage_tokens_extracted_correctly(self, client_and_mock):
        client, mock_llm = client_and_mock
        self._set_response(mock_llm, "THOUGHT: OK", input_tokens=847, output_tokens=89)

        result = client.generate("system", "user")

        assert result.usage["prompt_tokens"] == 847
        assert result.usage["completion_tokens"] == 89

    def test_latency_ms_is_non_negative_integer(self, client_and_mock):
        client, mock_llm = client_and_mock
        self._set_response(mock_llm)

        result = client.generate("system", "user")

        assert isinstance(result.latency_ms, int)
        assert result.latency_ms >= 0

    def test_tool_calls_is_empty_list(self, client_and_mock):
        client, mock_llm = client_and_mock
        self._set_response(mock_llm)

        result = client.generate("system", "user")

        assert result.tool_calls == []

    def test_temperature_passed_via_bind(self, client_and_mock):
        client, mock_llm = client_and_mock
        self._set_response(mock_llm)

        client.generate("sys", "usr", temperature=0.3)

        mock_llm.bind.assert_called_with(temperature=0.3)

    def test_none_content_normalized_to_empty_string(self, client_and_mock):
        client, mock_llm = client_and_mock
        ai_msg = _make_ai_message("anything")
        ai_msg.content = None
        mock_llm.bind.return_value.invoke.return_value = ai_msg

        result = client.generate("sys", "usr")

        assert result.content == ""

    def test_missing_usage_metadata_returns_zeros(self, client_and_mock):
        client, mock_llm = client_and_mock
        ai_msg = MagicMock()
        ai_msg.content = "THOUGHT: OK"
        ai_msg.usage_metadata = None
        mock_llm.bind.return_value.invoke.return_value = ai_msg

        result = client.generate("sys", "usr")

        assert result.usage["prompt_tokens"] == 0
        assert result.usage["completion_tokens"] == 0

    def test_invoke_exception_returns_empty_response(self, client_and_mock):
        client, mock_llm = client_and_mock
        mock_llm.bind.return_value.invoke.side_effect = Exception("API error")

        result = client.generate("sys", "usr")

        assert isinstance(result, LLMResponse)
        assert result.content == ""
        assert result.latency_ms == 0


# ── Tool injection ─────────────────────────────────────────────────────────────

class TestQwenLLMClientToolInjection:
    @pytest.fixture
    def client_and_mock(self):
        with patch(f"{_MODULE}.ChatOpenAI") as mock_cls:
            mock_llm = _mock_llm_chain(mock_cls)
            client = QwenLLMClient(api_key="sk-test")
            yield client, mock_llm

    def _get_system_content(self, mock_llm) -> str:
        """Extracts the content of the SystemMessage passed to invoke."""
        messages = mock_llm.bind.return_value.invoke.call_args[0][0]
        return messages[0].content  # SystemMessage is first

    def _get_user_content(self, mock_llm) -> str:
        messages = mock_llm.bind.return_value.invoke.call_args[0][0]
        return messages[1].content  # HumanMessage is second

    def test_tools_injected_into_system_prompt(self, client_and_mock):
        client, mock_llm = client_and_mock
        client.generate("sys", "usr", available_tools=_SAMPLE_TOOLS)

        system = self._get_system_content(mock_llm)
        assert "web_search" in system
        assert "navigate" in system

    def test_tool_descriptions_included(self, client_and_mock):
        client, mock_llm = client_and_mock
        client.generate("sys", "usr", available_tools=_SAMPLE_TOOLS)

        system = self._get_system_content(mock_llm)
        assert "Search" in system
        assert "Navigate" in system

    def test_original_system_prompt_preserved(self, client_and_mock):
        client, mock_llm = client_and_mock
        client.generate("MY_SYSTEM_PROMPT", "usr", available_tools=_SAMPLE_TOOLS)

        system = self._get_system_content(mock_llm)
        assert "MY_SYSTEM_PROMPT" in system

    def test_no_tools_system_prompt_unchanged(self, client_and_mock):
        client, mock_llm = client_and_mock
        client.generate("CLEAN_SYSTEM", "usr", available_tools=None)

        system = self._get_system_content(mock_llm)
        assert system == "CLEAN_SYSTEM"

    def test_empty_tools_system_prompt_unchanged(self, client_and_mock):
        client, mock_llm = client_and_mock
        client.generate("CLEAN_SYSTEM", "usr", available_tools=[])

        system = self._get_system_content(mock_llm)
        assert system == "CLEAN_SYSTEM"

    def test_user_prompt_passed_correctly(self, client_and_mock):
        client, mock_llm = client_and_mock
        client.generate("sys", "MY_USER_PROMPT")

        user = self._get_user_content(mock_llm)
        assert user == "MY_USER_PROMPT"

    def test_invoke_receives_two_messages(self, client_and_mock):
        client, mock_llm = client_and_mock
        client.generate("sys", "usr")

        messages = mock_llm.bind.return_value.invoke.call_args[0][0]
        assert len(messages) == 2
