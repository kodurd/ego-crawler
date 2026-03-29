from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class LLMResponse:
    content: str
    tool_calls: List[Dict[str, Any]]
    usage: Dict[str, int]  # prompt_tokens, completion_tokens
    latency_ms: int


class LLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        available_tools: Optional[List[Dict]] = None,
        temperature: float = 0.7
    ) -> LLMResponse:
        """Generate response from LLM"""
        pass