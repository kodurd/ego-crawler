import pytest
from datetime import timedelta

from ego_crawler.domain.entities.session import Session, SessionStatus
from ego_crawler.domain.entities.persona import Persona
from ego_crawler.domain.value_objects.budget import Budget
from ego_crawler.domain.value_objects.emotion import Emotion, Mood


class TestSession:
    def test_create_default_session(self):
        session = Session()

        assert session.status == SessionStatus.ACTIVE
        assert session.budget.total_minutes == 120
        assert session.is_active()

    def test_create_custom_session(self):
        persona = Persona(name="Test")
        budget = Budget.from_minutes(60)

        session = Session(
            persona=persona,
            budget=budget
        )

        assert session.persona.name == "Test"
        assert session.budget.total_minutes == 60

    def test_depleted_budget_not_active(self):
        budget = Budget(total_minutes=10, remaining_seconds=0)
        session = Session(budget=budget)

        assert not session.is_active()

    def test_consume_budget_reduces_remaining(self):
        session = Session(budget=Budget.from_minutes(10))
        updated = session.consume_budget(300)  # 5 minutes

        assert updated.budget.remaining_seconds == 300  # 5 minutes left
        assert session.budget.remaining_seconds == 600  # Original unchanged

    def test_next_step_number_increments(self):
        session = Session()
        assert session.next_step_number() == 1
        assert session.next_step_number() == 2
        assert session.current_step_number == 2

    def test_update_emotion(self):
        session = Session()
        new_emotion = Emotion(Mood.EXCITED, 0.9)

        updated = session.update_emotion(new_emotion)

        assert updated.current_emotion.mood == Mood.EXCITED
        assert updated.current_emotion.intensity == 0.9

    def test_complete_session(self):
        session = Session()
        completed = session.complete(reason="time_up")

        assert completed.status == SessionStatus.COMPLETED
        assert completed.end_time is not None
        assert not completed.is_active()

    def test_add_step_to_context_limits_history(self):
        session = Session()

        # Add more than MAX_RECENT_STEPS
        for i in range(15):
            session.add_step_to_context(f"step_{i}")

        assert len(session.recent_steps) == session.MAX_RECENT_STEPS
        assert session.recent_steps[-1] == "step_14"
        assert session.recent_steps[0] == "step_5"  # Oldest kept

    def test_get_context_for_prompt(self):
        session = Session(
            persona=Persona(name="Test"),
            current_url="https://example.com"
        )
        context = session.get_context_for_prompt()

        assert "Time remaining" in context
        assert "Current mood" in context
        assert "https://example.com" in context