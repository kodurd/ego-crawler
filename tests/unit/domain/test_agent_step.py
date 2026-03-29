import pytest
from uuid import uuid4

from ego_crawler.domain.entities.agent_step import AgentStep
from ego_crawler.domain.value_objects.step_type import StepType


class TestAgentStep:
    def test_create_thought(self):
        session_id = uuid4()
        step = AgentStep.create_thought(
            session_id=session_id,
            step_number=1,
            content="I want to search for unicorns"
        )

        assert step.step_type == StepType.THOUGHT
        assert step.thought_content == "I want to search for unicorns"
        assert step.session_id == session_id

    def test_create_action(self):
        session_id = uuid4()
        step = AgentStep.create_action(
            session_id=session_id,
            step_number=2,
            tool="web_search",
            params={"query": "unicorns"}
        )

        assert step.step_type == StepType.ACTION
        assert step.action_tool == "web_search"
        assert step.action_params == {"query": "unicorns"}

    def test_create_observation(self):
        session_id = uuid4()
        step = AgentStep.create_observation(
            session_id=session_id,
            step_number=2,
            raw_data="Found 10 results",
            summary="10 results found"
        )

        assert step.step_type == StepType.OBSERVATION
        assert step.observation_raw == "Found 10 results"
        assert step.observation_summary == "10 results found"

    def test_negative_step_number_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            AgentStep(step_number=-1)

    def test_is_tool_related(self):
        action = AgentStep(step_type=StepType.ACTION)
        observation = AgentStep(step_type=StepType.OBSERVATION)
        thought = AgentStep(step_type=StepType.THOUGHT)

        assert action.is_tool_related()
        assert observation.is_tool_related()
        assert not thought.is_tool_related()