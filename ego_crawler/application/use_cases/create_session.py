from dataclasses import dataclass

from ego_crawler.domain.entities.persona import Persona
from ego_crawler.domain.entities.session import Session
from ego_crawler.domain.value_objects.budget import Budget
from ego_crawler.domain.repositories.session_repository import SessionRepository


@dataclass
class CreateSessionRequest:
    persona_name: str = "Anonymous"
    persona_age: int = 25
    interests: list = None
    budget_minutes: int = 120
    context_vars: dict = None


class CreateSession:
    def __init__(self, session_repo: SessionRepository):
        self.session_repo = session_repo

    def execute(self, request: CreateSessionRequest) -> Session:
        persona = Persona(
            name=request.persona_name,
            age=request.persona_age,
            interests=request.interests or [],
            context_vars=request.context_vars or {}
        )

        session = Session(
            persona=persona,
            budget=Budget.from_minutes(request.budget_minutes)
        )

        self.session_repo.save(session)
        return session
