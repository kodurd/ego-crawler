from abc import ABC, abstractmethod
from uuid import UUID

from ego_crawler.domain.entities.session import Session


class SessionRepository(ABC):
    @abstractmethod
    def save(self, session: Session) -> None:
        """Save session state"""
        pass

    @abstractmethod
    def get_by_id(self, session_id: UUID) -> Session | None:
        """Get session by ID"""
        pass

    @abstractmethod
    def get_active(self) -> Session | None:
        """Get currently active session (if any)"""
        pass
