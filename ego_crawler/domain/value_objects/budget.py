from dataclasses import dataclass
from datetime import timedelta


@dataclass(frozen=True)
class Budget:
    total_minutes: int
    remaining_seconds: int

    def __post_init__(self):
        if self.total_minutes <= 0:
            raise ValueError("Total minutes must be positive")
        if self.remaining_seconds < 0:
            raise ValueError("Remaining seconds cannot be negative")
        if self.remaining_seconds > self.total_minutes * 60:
            raise ValueError("Remaining seconds cannot exceed total")

    @classmethod
    def from_minutes(cls, minutes: int) -> "Budget":
        return cls(total_minutes=minutes, remaining_seconds=minutes * 60)

    def is_depleted(self) -> bool:
        return self.remaining_seconds <= 0

    def consume(self, seconds: int) -> "Budget":
        new_remaining = max(0, self.remaining_seconds - seconds)
        return Budget(self.total_minutes, new_remaining)

    def as_timedelta(self) -> timedelta:
        return timedelta(seconds=self.remaining_seconds)
