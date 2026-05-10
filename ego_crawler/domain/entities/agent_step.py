from typing import Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4

from ego_crawler.domain.value_objects.step_type import StepType


@dataclass
class AgentStep:
    id: UUID = field(default_factory=uuid4)
    session_id: UUID = field(default_factory=uuid4)
    step_number: int = 0
    step_type: StepType = StepType.INTERNAL
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Content
    thought_content: str | None = None
    action_tool: str | None = None
    action_params: dict[str, Any] | None = None
    observation_raw: str | None = None
    observation_summary: str | None = None

    # Metadata
    latency_ms: int | None = None
    token_count_input: int | None = None
    token_count_output: int | None = None
    observation_success: bool | None = None

    def __post_init__(self):
        if self.step_number < 0:
            raise ValueError("Step number must be non-negative")

    @classmethod
    def create_thought(
            cls,
            session_id: UUID,
            step_number: int,
            content: str,
            **metadata
    ) -> "AgentStep":
        return cls(
            session_id=session_id,
            step_number=step_number,
            step_type=StepType.THOUGHT,
            thought_content=content,
            **metadata
        )

    @classmethod
    def create_action(
            cls,
            session_id: UUID,
            step_number: int,
            tool: str,
            params: dict[str, Any],
            **metadata
    ) -> "AgentStep":
        return cls(
            session_id=session_id,
            step_number=step_number,
            step_type=StepType.ACTION,
            action_tool=tool,
            action_params=params,
            **metadata
        )

    @classmethod
    def create_observation(
            cls,
            session_id: UUID,
            step_number: int,
            raw_data: str,
            summary: str | None = None,
            **metadata
    ) -> "AgentStep":
        return cls(
            session_id=session_id,
            step_number=step_number,
            step_type=StepType.OBSERVATION,
            observation_raw=raw_data,
            observation_summary=summary,
            **metadata
        )

    def is_tool_related(self) -> bool:
        return self.step_type in {StepType.ACTION, StepType.OBSERVATION}