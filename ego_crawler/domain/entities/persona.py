from typing import Any
from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass
class Persona:
    id: UUID = field(default_factory=uuid4)
    name: str = "Anonymous"
    age: int = 25
    archetype: str = "default"
    system_prompt: str = ""
    interests: list[str] = field(default_factory=list)
    fears: list[str] = field(default_factory=list)
    personality_traits: dict[str, float] = field(default_factory=dict)
    context_vars: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.age < 0:
            raise ValueError("Age cannot be negative")
        if not self.system_prompt:
            self.system_prompt = self._generate_default_prompt()

    def _generate_default_prompt(self) -> str:
        traits = ", ".join(f"{k}={v}" for k, v in self.personality_traits.items())
        return (
            f"You are {self.name}, {self.age} years old. "
            f"Interests: {', '.join(self.interests)}. "
            f"Traits: {traits}."
        )

    def with_context(self, **kwargs) -> "Persona":
        new_context = {**self.context_vars, **kwargs}
        return Persona(
            id=self.id,
            name=self.name,
            age=self.age,
            archetype=self.archetype,
            system_prompt=self.system_prompt,
            interests=self.interests,
            fears=self.fears,
            personality_traits=self.personality_traits,
            context_vars=new_context
        )

    def get_full_prompt(self) -> str:
        context = "\n".join(f"{k}: {v}" for k, v in self.context_vars.items())
        return f"{self.system_prompt}\n\nCurrent context:\n{context}"
