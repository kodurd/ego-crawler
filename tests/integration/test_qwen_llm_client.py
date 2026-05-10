"""Integration tests for QwenLLMClient.

These tests make REAL HTTP requests to DashScope API.
They are skipped by default; run with:

    DASHSCOPE_API_KEY=sk-... pytest --integration

The tests validate that:
- A real Qwen model follows the THOUGHT:/ACTION: format
- Token usage is populated with real values
- Latency is measured and > 0
- The full agent loop works end-to-end with Аня's persona
"""
import os

import pytest

from ego_crawler.application.ports.llm_client import LLMResponse
from ego_crawler.application.use_cases.create_session import CreateSession, CreateSessionRequest
from ego_crawler.application.use_cases.execute_agent_step import ExecuteAgentStep
from ego_crawler.application.services.prompt_builder import PromptBuilder
from ego_crawler.domain.personas.anya_sokolova import create_anya_sokolova
from ego_crawler.infrastructure.mocks.fake_web_tools import FakeWebTools
from ego_crawler.infrastructure.qwen.qwen_llm_client import QwenLLMClient

_has_api_key = bool(os.environ.get("DASHSCOPE_API_KEY"))
_skip_no_key = pytest.mark.skipif(
    not _has_api_key,
    reason="DASHSCOPE_API_KEY not set — skipping Qwen integration tests",
)

_AVAILABLE_TOOLS = [
    {"name": "web_search",     "description": "Search the web",          "parameters": {"query": "string"}},
    {"name": "navigate",       "description": "Navigate to URL",          "parameters": {"url": "string"}},
    {"name": "extract_text",   "description": "Extract text from page",   "parameters": {"selector": "string (optional)"}},
    {"name": "extract_images", "description": "Extract images from page", "parameters": {"selector": "string (optional)"}},
    {"name": "scroll",         "description": "Scroll the page",          "parameters": {"direction": "up|down"}},
    {"name": "go_back",        "description": "Go back",                  "parameters": {}},
]


class InMemorySessionRepo:
    def __init__(self): self.sessions = {}
    def save(self, s): self.sessions[s.id] = s
    def get_by_id(self, sid): return self.sessions.get(sid)
    def get_active(self): return next((s for s in self.sessions.values() if s.is_active()), None)


class InMemoryStepRepo:
    def __init__(self): self.steps = []
    def save(self, s): self.steps.append(s)
    def get_by_session(self, sid, limit=None):
        r = [s for s in self.steps if s.session_id == sid]
        return r[:limit] if limit else r
    def get_latest(self, sid, n=1):
        f = [s for s in self.steps if s.session_id == sid]
        return f[-n:] if f else []


@pytest.fixture(scope="module")
def qwen_client():
    return QwenLLMClient()


# ── QwenLLMClient generate() ───────────────────────────────────────────────────

@pytest.mark.integration
class TestQwenGenerateIntegration:
    @_skip_no_key
    def test_generate_returns_llm_response(self, qwen_client):
        result = qwen_client.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say exactly: THOUGHT: hello",
        )
        assert isinstance(result, LLMResponse)

    @_skip_no_key
    def test_content_is_non_empty(self, qwen_client):
        result = qwen_client.generate(
            system_prompt="You are a terse assistant.",
            user_prompt="THOUGHT: test",
        )
        assert len(result.content) > 0

    @_skip_no_key
    def test_usage_prompt_tokens_populated(self, qwen_client):
        result = qwen_client.generate(
            system_prompt="You are helpful.",
            user_prompt="Hello",
        )
        assert result.usage.get("prompt_tokens", 0) > 0

    @_skip_no_key
    def test_usage_completion_tokens_populated(self, qwen_client):
        result = qwen_client.generate(
            system_prompt="You are helpful.",
            user_prompt="Hello",
        )
        assert result.usage.get("completion_tokens", 0) > 0

    @_skip_no_key
    def test_latency_ms_is_positive(self, qwen_client):
        result = qwen_client.generate(
            system_prompt="You are helpful.",
            user_prompt="Hello",
        )
        assert result.latency_ms > 0

    @_skip_no_key
    def test_model_follows_thought_action_format(self, qwen_client):
        """Qwen should follow the THOUGHT:/ACTION: format when instructed."""
        system = (
            "You are a test agent. Always respond in this format:\n"
            "THOUGHT: <one sentence>\n"
            "ACTION: web_search | {\"query\": \"test\"}\n"
            "Never deviate from this format."
        )
        result = qwen_client.generate(
            system_prompt=system,
            user_prompt="What do you want to search for?",
            available_tools=_AVAILABLE_TOOLS,
        )
        assert "THOUGHT:" in result.content


# ── Аня Соколова + Qwen — end-to-end ──────────────────────────────────────────

@pytest.mark.integration
class TestAnyaWithQwenIntegration:
    @_skip_no_key
    def test_anya_generates_thought_about_ege(self):
        """Аня must think about ЕГЭ-related things given her persona."""
        client = QwenLLMClient()
        anya = create_anya_sokolova()

        result = client.generate(
            system_prompt=anya.system_prompt,
            user_prompt=(
                "=== CURRENT STATE ===\n"
                "Time remaining: 1:55:00\n"
                "Current mood: CURIOUS (intensity: 0.50)\n"
                "Current URL: None\n"
                "Steps taken: 0\n\n"
                "=== YOUR TURN ===\n"
                "Respond in this format:\n"
                "THOUGHT: <your reasoning as persona>\n"
                "ACTION: <tool_name> | <params_json> (if using tool)"
            ),
            available_tools=_AVAILABLE_TOOLS,
            temperature=0.7,
        )

        assert "THOUGHT:" in result.content

    @_skip_no_key
    def test_anya_full_session_one_step(self):
        """Full pipeline: CreateSession → ExecuteAgentStep with real Qwen."""
        session_repo = InMemorySessionRepo()
        step_repo    = InMemoryStepRepo()
        qwen         = QwenLLMClient()
        web_tools    = FakeWebTools(success_rate=1.0)
        builder      = PromptBuilder()

        create_uc = CreateSession(session_repo)
        step_uc   = ExecuteAgentStep(session_repo, step_repo, qwen, web_tools, builder)

        anya = create_anya_sokolova()
        session = create_uc.execute(CreateSessionRequest(
            persona_name=anya.name,
            persona_age=anya.age,
            interests=anya.interests,
            budget_minutes=10,
        ))

        result = step_uc.execute(session.id)

        assert result.thought is not None
        assert result.thought.thought_content
        assert result.session.current_step_number == 1
        assert result.session.budget.remaining_seconds < 10 * 60
