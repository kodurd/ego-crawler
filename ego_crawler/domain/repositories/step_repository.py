from abc import ABC, abstractmethod
from uuid import UUID

from ego_crawler.domain.entities.agent_step import AgentStep


class StepRepository(ABC):
    @abstractmethod
    def save(self, step: AgentStep) -> None:
        """Save agent step (append-only)"""
        pass

    @abstractmethod
    def get_by_session(
            self,
            session_id: UUID,
            limit: int | None = None
    ) -> list[AgentStep]:
        """Get steps for session, ordered by step_number"""
        pass

    @abstractmethod
    def get_latest(self, session_id: UUID, n: int = 1) -> list[AgentStep]:
        """Get latest N steps for context"""
        pass
