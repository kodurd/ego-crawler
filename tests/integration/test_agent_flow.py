import pytest

from ego_crawler.application.use_cases.create_session import (
    CreateSession, CreateSessionRequest
)
from ego_crawler.application.use_cases.execute_agent_step import ExecuteAgentStep
from ego_crawler.application.services.prompt_builder import PromptBuilder
from ego_crawler.infrastructure.mocks.fake_llm_client import FakeLLMClient
from ego_crawler.infrastructure.mocks.fake_web_tools import FakeWebTools


class InMemorySessionRepo:
    def __init__(self):
        self.sessions = {}

    def save(self, session):
        self.sessions[session.id] = session

    def get_by_id(self, sid):
        return self.sessions.get(sid)

    def get_active(self):
        return next((s for s in self.sessions.values() if s.is_active()), None)


class InMemoryStepRepo:
    def __init__(self):
        self.steps = []

    def save(self, step):
        self.steps.append(step)

    def get_by_session(self, sid, limit=None):
        res = [s for s in self.steps if s.session_id == sid]
        return res[:limit] if limit else res

    def get_latest(self, sid, n=1):
        filtered = [s for s in self.steps if s.session_id == sid]
        return filtered[-n:] if filtered else []


class TestAgentFlow:
    """Integration test: full agent session lifecycle"""

    def test_full_session_lifecycle(self):
        # Setup
        session_repo = InMemorySessionRepo()
        step_repo = InMemoryStepRepo()
        llm_client = FakeLLMClient(seed=42)
        web_tools = FakeWebTools()

        create_uc = CreateSession(session_repo)
        step_uc = ExecuteAgentStep(
            session_repo, step_repo, llm_client,
            web_tools, PromptBuilder()
        )

        # Create session with teen persona
        session = create_uc.execute(CreateSessionRequest(
            persona_name="TeenGirl",
            persona_age=15,
            interests=["unicorns", "pink", "memes"],
            budget_minutes=5  # Short for test
        ))

        assert session.is_active()

        # Execute multiple steps
        for i in range(3):
            result = step_uc.execute(session.id)
            session = result.session

            assert result.thought is not None
            assert session.current_step_number == i + 1

            # Update repo reference
            session_repo.save(session)

        # Verify history
        steps = step_repo.get_by_session(session.id)
        thoughts = [s for s in steps if s.step_type.name == "THOUGHT"]
        actions = [s for s in steps if s.step_type.name == "ACTION"]

        assert len(thoughts) == 3
        assert len(actions) >= 1  # At least some actions

        # Verify budget consumed
        assert session.budget.remaining_seconds < 5 * 60

    def test_session_with_unicorn_persona_searches_unicorns(self):
        session_repo = InMemorySessionRepo()
        step_repo = InMemoryStepRepo()
        llm_client = FakeLLMClient(seed=42)  # Will generate unicorn queries
        web_tools = FakeWebTools()

        create_uc = CreateSession(session_repo)
        step_uc = ExecuteAgentStep(
            session_repo, step_repo, llm_client,
            web_tools, PromptBuilder()
        )

        session = create_uc.execute(CreateSessionRequest(
            persona_name="UnicornLover",
            interests=["unicorns", "pink"],
            budget_minutes=10
        ))

        # Execute step
        result = step_uc.execute(session.id)

        assert "unicorn" in result.thought.thought_content.lower() or \
               "pink" in result.thought.thought_content.lower()