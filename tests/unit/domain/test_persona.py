import pytest
from uuid import UUID

from ego_crawler.domain.entities.persona import Persona


class TestPersona:
    def test_create_default_persona(self):
        persona = Persona()

        assert isinstance(persona.id, UUID)
        assert persona.name == "Anonymous"
        assert persona.age == 25
        assert "Anonymous" in persona.system_prompt

    def test_create_custom_persona(self):
        persona = Persona(
            name="Alice",
            age=15,
            interests=["unicorns", "pink"],
            personality_traits={"curiosity": 0.9}
        )

        assert persona.name == "Alice"
        assert persona.age == 15
        assert "unicorns" in persona.interests
        assert "curiosity=0.9" in persona.system_prompt

    def test_negative_age_raises(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            Persona(age=-5)

    def test_with_context_creates_new_instance(self):
        persona = Persona(name="Bob")
        updated = persona.with_context(location="home", time="evening")

        assert updated.context_vars == {"location": "home", "time": "evening"}
        assert persona.context_vars == {}  # Original unchanged

    def test_get_full_prompt_includes_context(self):
        persona = Persona(
            name="Test",
            context_vars={"mood": "happy"}
        )
        prompt = persona.get_full_prompt()

        assert "Test" in prompt
        assert "mood: happy" in prompt