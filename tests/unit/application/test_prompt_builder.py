import pytest
from uuid import uuid4

from ego_crawler.application.services.prompt_builder import PromptBuilder
from ego_crawler.domain.entities.session import Session
from ego_crawler.domain.entities.persona import Persona
from ego_crawler.domain.entities.agent_step import AgentStep


class TestPromptBuilder:
    def test_build_system_prompt_includes_persona(self):
        builder = PromptBuilder()
        persona = Persona(name="Alice", age=15, interests=["unicorns"])
        session = Session(persona=persona)

        prompt = builder.build_system_prompt(session)

        assert "Alice" in prompt
        assert "15" in prompt
        assert "unicorns" in prompt
        assert "CONSTRAINTS" in prompt

    def test_build_user_prompt_includes_state(self):
        builder = PromptBuilder()
        session = Session()

        prompt = builder.build_user_prompt(session)

        assert "CURRENT STATE" in prompt
        assert "RECENT HISTORY" in prompt
        assert "YOUR TURN" in prompt

    def test_build_user_prompt_with_recent_steps(self):
        builder = PromptBuilder()
        session = Session()

        steps = [
            AgentStep.create_thought(uuid4(), 1, "Thinking..."),
            AgentStep.create_action(uuid4(), 2, "search", {"q": "test"})
        ]

        prompt = builder.build_user_prompt(session, steps)

        assert "[1] Thought" in prompt
        assert "[2] Action" in prompt

    def test_format_step_truncates_long_content(self):
        builder = PromptBuilder()
        step = AgentStep.create_thought(
            uuid4(), 1, "A" * 200
        )

        formatted = builder._format_step(step)

        assert len(formatted) < 150  # Truncated
        assert "..." in formatted