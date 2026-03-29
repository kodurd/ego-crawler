import pytest
from uuid import uuid4

from ego_crawler.application.use_cases.execute_agent_step import (
    ExecuteAgentStep, StepResult
)
from ego_crawler.application.services.prompt_builder import PromptBuilder
from ego_crawler.domain.entities.session import Session
from ego_crawler.domain.entities.agent_step import AgentStep
from ego_crawler.domain.entities.persona import Persona
from ego_crawler.infrastructure.mocks.fake_llm_client import FakeLLMClient
from ego_crawler.infrastructure.mocks.fake_web_tools import FakeWebTools


class FakeSessionRepository:
    """Fake repo defined in test file"""

    def __init__(self):
        self.sessions = {}

    def save(self, session: Session) -> None:
        self.sessions[session.id] = session

    def get_by_id(self, session_id):
        return self.sessions.get(session_id)

    def get_active(self):
        for s in self.sessions.values():
            if s.is_active():
                return s
        return None


class FakeStepRepository:
    """Fake repo defined in test file"""

    def __init__(self):
        self.steps = []

    def save(self, step: AgentStep) -> None:
        self.steps.append(step)

    def get_by_session(self, session_id, limit=None):
        res = [s for s in self.steps if s.session_id == session_id]
        return res[:limit] if limit else res

    def get_latest(self, session_id, n=1):
        filtered = [s for s in self.steps if s.session_id == session_id]
        return filtered[-n:] if filtered else []


class TestExecuteAgentStep:
    def setup_method(self):
        self.session_repo = FakeSessionRepository()
        self.step_repo = FakeStepRepository()
        self.llm_client = FakeLLMClient(seed=42)
        self.web_tools = FakeWebTools()
        self.prompt_builder = PromptBuilder()

        self.use_case = ExecuteAgentStep(
            session_repo=self.session_repo,
            step_repo=self.step_repo,
            llm_client=self.llm_client,
            web_tools=self.web_tools,
            prompt_builder=self.prompt_builder
        )

        # Create test session
        self.session = Session(persona=Persona(name="Test", interests=["test"]))
        self.session_repo.save(self.session)

    def test_execute_step_creates_thought(self):
        result = self.use_case.execute(self.session.id)

        assert result.thought is not None
        assert result.thought.step_type.name == "THOUGHT"
        assert len(result.thought.thought_content) > 0

    def test_execute_step_with_action_creates_observation(self):
        result = self.use_case.execute(self.session.id)

        if result.action:
            assert result.observation is not None
            assert result.observation.step_type.name == "OBSERVATION"

    def test_execute_step_saves_to_repository(self):
        initial_count = len(self.step_repo.steps)

        self.use_case.execute(self.session.id)

        assert len(self.step_repo.steps) > initial_count
        assert any(s.step_type.name == "THOUGHT" for s in self.step_repo.steps)

    def test_execute_step_consumes_budget(self):
        initial_budget = self.session.budget.remaining_seconds

        result = self.use_case.execute(self.session.id)

        assert result.session.budget.remaining_seconds < initial_budget

    def test_execute_inactive_session_raises(self):
        # Создаем завершенную сессию напрямую
        from ego_crawler.domain.value_objects.budget import Budget
        from ego_crawler.domain.entities.session import SessionStatus
        from datetime import datetime, timezone

        completed_session = Session(
            budget=Budget.from_minutes(10),
            status=SessionStatus.COMPLETED,
            end_time=datetime.now(timezone.utc)
        )
        self.session_repo.save(completed_session)

        with pytest.raises(ValueError, match="not active"):
            self.use_case.execute(completed_session.id)

    def test_execute_nonexistent_session_raises(self):
        with pytest.raises(ValueError, match="not found"):
            self.use_case.execute(uuid4())

    def test_parse_response_extracts_thought_only(self):
        content = "THOUGHT: I am thinking about something"
        thought, action, params = self.use_case._parse_response(content)

        assert thought == "I am thinking about something"
        assert action is None
        assert params is None

    def test_parse_response_extracts_thought_and_action(self):
        content = 'THOUGHT: I want to search\nACTION: web_search | {"query": "test"}'
        thought, action, params = self.use_case._parse_response(content)

        assert thought == "I want to search"
        assert action == "web_search"
        assert params == {"query": "test"}