import random
import time
from typing import List, Dict, Optional

from ego_crawler.application.ports.llm_client import LLMClient, LLMResponse


class FakeLLMClient(LLMClient):
    """Fake LLM client for testing - simulates persona behavior"""

    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
        self.call_count = 0

    def generate(
            self,
            system_prompt: str,
            user_prompt: str,
            available_tools: Optional[List[Dict]] = None,
            temperature: float = 0.7
    ) -> LLMResponse:
        self.call_count += 1
        time.sleep(0.01)  # Simulate latency

        # Simple parsing of persona from system prompt
        persona_type = "default"
        if "unicorn" in system_prompt.lower():
            persona_type = "unicorn_lover"
        elif "15" in system_prompt and "school" in system_prompt.lower():
            persona_type = "teen"

        # Generate response based on persona
        content = self._generate_persona_response(persona_type, user_prompt)

        return LLMResponse(
            content=content,
            tool_calls=[],
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            latency_ms=10
        )

    def _generate_persona_response(self, persona_type: str, context: str) -> str:
        responses = {
            "unicorn_lover": [
                "THOUGHT: Oh! I wonder if there are any pink unicorns online today? Let me search!\nACTION: web_search | {\"query\": \"pink unicorns cute images\"}",
                "THOUGHT: That blue unicorn was ugly, I want pink only!\nACTION: web_search | {\"query\": \"pink unicorn only\"}",
                "THOUGHT: I'm so happy when I see pink unicorns! Let me scroll for more.\nACTION: scroll | {\"direction\": \"down\"}",
            ],
            "teen": [
                "THOUGHT: Ugh, school was so hard today. Let me check some memes.\nACTION: web_search | {\"query\": \"funny memes 2026\"}",
                "THOUGHT: I hope the bullies aren't online... Let me look at something fun.\nACTION: navigate | {\"url\": \"https://example.com/fun\"}",
                "THOUGHT: Mom said only 2 hours, I should hurry!\nACTION: extract_text | {}",
            ],
            "default": [
                "THOUGHT: Let me explore what's available.\nACTION: web_search | {\"query\": \"interesting content\"}",
                "THOUGHT: I need to think about this.\nACTION: navigate | {\"url\": \"https://example.com\"}",
            ]
        }

        return random.choice(responses.get(persona_type, responses["default"]))
