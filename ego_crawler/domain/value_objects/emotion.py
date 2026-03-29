from dataclasses import dataclass
from enum import Enum, auto


class Mood(Enum):
    CURIOUS = auto()
    BORED = auto()
    EXCITED = auto()
    ANXIOUS = auto()
    HAPPY = auto()
    FRUSTRATED = auto()


@dataclass(frozen=True)
class Emotion:
    mood: Mood
    intensity: float  # 0.0 to 1.0
    trigger: str | None = None

    def __post_init__(self):
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError(f"Intensity must be between 0.0 and 1.0, got {self.intensity}")

    def with_trigger(self, trigger: str) -> "Emotion":
        return Emotion(self.mood, self.intensity, trigger)

    def is_positive(self) -> bool:
        return self.mood in {Mood.CURIOUS, Mood.EXCITED, Mood.HAPPY}
