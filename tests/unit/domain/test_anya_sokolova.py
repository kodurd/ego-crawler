import pytest

from ego_crawler.domain.entities.persona import Persona
from ego_crawler.domain.personas.anya_sokolova import (
    ANYA_FEARS,
    ANYA_INTERESTS,
    ANYA_SYSTEM_PROMPT,
    ANYA_TRAITS,
    create_anya_sokolova,
)


@pytest.fixture
def anya() -> Persona:
    return create_anya_sokolova()


class TestAnyaSokolovaFactory:
    def test_returns_persona_instance(self, anya):
        assert isinstance(anya, Persona)

    def test_name_is_anya_sokolova(self, anya):
        assert anya.name == "Аня Соколова"

    def test_age_is_sixteen(self, anya):
        assert anya.age == 16

    def test_archetype_is_perfectionist_student(self, anya):
        assert anya.archetype == "perfectionist_student"

    def test_has_system_prompt(self, anya):
        assert anya.system_prompt
        assert len(anya.system_prompt) > 100

    def test_has_interests(self, anya):
        assert len(anya.interests) > 0

    def test_has_fears(self, anya):
        assert len(anya.fears) > 0

    def test_has_personality_traits(self, anya):
        assert len(anya.personality_traits) > 0

    def test_factory_creates_independent_instances(self):
        a1 = create_anya_sokolova()
        a2 = create_anya_sokolova()
        assert a1.id != a2.id


class TestAnyaPersonalityTraits:
    def test_perfectionism_is_very_high(self, anya):
        assert anya.personality_traits["perfectionism"] >= 0.90

    def test_anxiety_is_high(self, anya):
        assert anya.personality_traits["anxiety"] >= 0.75

    def test_conscientiousness_is_high(self, anya):
        assert anya.personality_traits["conscientiousness"] >= 0.85

    def test_self_criticism_is_high(self, anya):
        assert anya.personality_traits["self_criticism"] >= 0.80

    def test_all_traits_in_valid_range(self, anya):
        for name, value in anya.personality_traits.items():
            assert 0.0 <= value <= 1.0, f"Trait '{name}' = {value} out of [0, 1]"

    def test_procrastination_is_moderate(self, anya):
        assert anya.personality_traits["procrastination"] < 0.6


class TestAnyaInterests:
    def test_interests_include_ege(self, anya):
        combined = " ".join(anya.interests).lower()
        assert "егэ" in combined

    def test_interests_include_olympiads(self, anya):
        combined = " ".join(anya.interests).lower()
        assert "олимп" in combined

    def test_interests_include_university_ratings(self, anya):
        combined = " ".join(anya.interests).lower()
        assert any(uni in combined for uni in ["мфти", "мгу", "вшэ", "вуз"])

    def test_interests_list_not_empty(self, anya):
        assert len(anya.interests) >= 4


class TestAnyaFears:
    def test_fears_include_bad_grade(self, anya):
        combined = " ".join(anya.fears).lower()
        assert any(word in combined for word in ["четвёрку", "четверку", "оценк", "балл"])

    def test_fears_include_mfti_or_mgu(self, anya):
        combined = " ".join(anya.fears).lower()
        assert any(uni in combined for uni in ["мфти", "мгу"])

    def test_fears_include_parents(self, anya):
        combined = " ".join(anya.fears).lower()
        assert any(word in combined for word in ["мам", "родител", "пап"])

    def test_fears_list_not_empty(self, anya):
        assert len(anya.fears) >= 4


class TestAnyaSystemPrompt:
    def test_prompt_contains_name(self):
        assert "Аня" in ANYA_SYSTEM_PROMPT

    def test_prompt_contains_age(self):
        assert "16" in ANYA_SYSTEM_PROMPT

    def test_prompt_contains_ege(self):
        assert "ЕГЭ" in ANYA_SYSTEM_PROMPT

    def test_prompt_contains_mfti_or_mgu(self):
        assert any(uni in ANYA_SYSTEM_PROMPT for uni in ["МФТИ", "МГУ"])

    def test_prompt_contains_competitor_masha(self):
        assert "Маша" in ANYA_SYSTEM_PROMPT

    def test_prompt_contains_few_shot_example(self):
        assert "THOUGHT:" in ANYA_SYSTEM_PROMPT
        assert "ACTION:" in ANYA_SYSTEM_PROMPT

    def test_prompt_contains_imposter_syndrome_theme(self):
        assert any(phrase in ANYA_SYSTEM_PROMPT for phrase in [
            "самозванц", "не умная", "зубрила", "синдром"
        ])

    def test_prompt_is_substantial(self):
        assert len(ANYA_SYSTEM_PROMPT) > 500


class TestAnyaPersonaIntegration:
    def test_with_context_returns_new_instance(self, anya):
        updated = anya.with_context(current_task="подготовка к контрольной")

        assert updated.context_vars["current_task"] == "подготовка к контрольной"
        assert anya.context_vars == {}

    def test_full_prompt_includes_system_prompt(self, anya):
        prompt = anya.get_full_prompt()
        assert "Аня" in prompt

    def test_full_prompt_with_context_includes_context_vars(self, anya):
        updated = anya.with_context(mood="паника перед ЕГЭ")
        prompt = updated.get_full_prompt()

        assert "паника перед ЕГЭ" in prompt
