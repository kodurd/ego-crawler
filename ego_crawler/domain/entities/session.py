from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional
from uuid import UUID, uuid4

from ego_crawler.domain.value_objects.budget import Budget
from ego_crawler.domain.value_objects.emotion import Emotion, Mood
from ego_crawler.domain.entities.persona import Persona


class SessionStatus(Enum):
    ACTIVE = auto()
    PAUSED = auto()
    COMPLETED = auto()
    ERROR = auto()


@dataclass
class Session:
    id: UUID = field(default_factory=uuid4)
    persona: Persona = field(default_factory=Persona)
    status: SessionStatus = SessionStatus.ACTIVE

    # Time management
    budget: Budget = field(default_factory=lambda: Budget.from_minutes(120))
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None

    # State
    current_emotion: Emotion = field(
        default_factory=lambda: Emotion(Mood.CURIOUS, 0.5)
    )
    current_step_number: int = 0
    current_url: Optional[str] = None

    # History (in-memory, для быстрого доступа)
    recent_steps: list = field(default_factory=list, repr=False)

    MAX_RECENT_STEPS: int = 10  # Для контекста в промпт

    def next_step_number(self) -> int:
        self.current_step_number += 1
        return self.current_step_number

    def consume_budget(self, seconds: int) -> "Session":
        new_budget = self.budget.consume(seconds)
        return Session(
            id=self.id,
            persona=self.persona,
            status=self.status,
            budget=new_budget,
            start_time=self.start_time,
            end_time=self.end_time,
            current_emotion=self.current_emotion,
            current_step_number=self.current_step_number,
            current_url=self.current_url,
            recent_steps=self.recent_steps
        )

    def update_emotion(self, emotion: Emotion) -> "Session":
        return Session(
            id=self.id,
            persona=self.persona,
            status=self.status,
            budget=self.budget,
            start_time=self.start_time,
            end_time=self.end_time,
            current_emotion=emotion,
            current_step_number=self.current_step_number,
            current_url=self.current_url,
            recent_steps=self.recent_steps
        )

    def add_step_to_context(self, step) -> None:
        self.recent_steps.append(step)
        if len(self.recent_steps) > self.MAX_RECENT_STEPS:
            self.recent_steps.pop(0)

    def complete(self, reason: str = "") -> "Session":
        return Session(
            id=self.id,
            persona=self.persona,
            status=SessionStatus.COMPLETED,
            budget=self.budget,
            start_time=self.start_time,
            end_time=datetime.now(timezone.utc),
            current_emotion=self.current_emotion,
            current_step_number=self.current_step_number,
            current_url=self.current_url,
            recent_steps=self.recent_steps
        )

    def is_active(self) -> bool:
        return self.status == SessionStatus.ACTIVE and not self.budget.is_depleted()

    def get_context_for_prompt(self) -> str:
        lines = [
            f"Time remaining: {self.budget.as_timedelta()}",
            f"Current mood: {self.current_emotion.mood.name} "
            f"(intensity: {self.current_emotion.intensity:.2f})",
            f"Current URL: {self.current_url or 'None'}",
            f"Steps taken: {self.current_step_number}",
        ]
        return "\n".join(lines)