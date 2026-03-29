import pytest
from uuid import UUID

from ego_crawler.application.use_cases.create_session import (
    CreateSession, CreateSessionRequest
)
from ego_crawler.domain.repositories.session_repository import SessionRepository
from ego_crawler.domain.entities.session import Session


class FakeSessionRepository(SessionRepository):
    def __init__(self):
        self.sessions = {}

    def save(self, session: Session) -> None:
        self.sessions[session.id] = session

    def get_by_id(self, session_id: UUID):
        return self.sessions.get(session_id)

    def get_active(self):
        for s in self.sessions.values():
            if s.is_active():
                return s
        return None


class TestCreateSession:
    def test_create_default_session(self):
        repo = FakeSessionRepository()
        use_case = CreateSession(repo)

        request = CreateSessionRequest()
        session = use_case.execute(request)

        assert isinstance(session.id, UUID)
        assert session.budget.total_minutes == 120
        assert session.persona.name == "Anonymous"
        assert session.id in repo.sessions

    def test_create_custom_session(self):
        repo = FakeSessionRepository()
        use_case = CreateSession(repo)

        request = CreateSessionRequest(
            persona_name="TeenGirl",
            persona_age=15,
            interests=["unicorns", "pink"],
            budget_minutes=60,
            context_vars={"location": "home"}
        )
        session = use_case.execute(request)

        assert session.persona.name == "TeenGirl"
        assert session.persona.age == 15
        assert "unicorns" in session.persona.interests
        assert session.budget.total_minutes == 60
        assert session.persona.context_vars["location"] == "home"