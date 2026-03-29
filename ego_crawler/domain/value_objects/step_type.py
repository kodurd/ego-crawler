from enum import Enum, auto


class StepType(Enum):
    THOUGHT = auto()      # Внутреннее рассуждение агента
    ACTION = auto()       # Вызов инструмента
    OBSERVATION = auto()  # Результат инструмента
    INTERNAL = auto()     # Внутреннее действие без инструмента
    ERROR = auto()        # Ошибка выполнения

    def __eq__(self, other):
        if isinstance(other, StepType):
            return self.value == other.value
        return NotImplemented

    def __hash__(self):
        return hash(self.value)
