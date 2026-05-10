from __future__ import annotations

import logging
import os
import time
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ego_crawler.application.ports.llm_client import LLMClient, LLMResponse
from ego_crawler.infrastructure.qwen.tool_formatter import ToolFormatter

logger = logging.getLogger(__name__)

# International endpoint — works outside mainland China
_DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
_DEFAULT_MODEL = "qwen-plus"
_DEFAULT_MAX_TOKENS = 512


class QwenLLMClient(LLMClient):
    """LLMClient backed by Qwen via DashScope's OpenAI-compatible API (langchain).

    Tool descriptions are injected into the system prompt so the model responds
    in the THOUGHT:/ACTION: text format — no changes to use cases required.

    Usage:
        client = QwenLLMClient()              # reads DASHSCOPE_API_KEY from env
        client = QwenLLMClient(api_key="sk-...")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        base_url: str = _DASHSCOPE_BASE_URL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        resolved_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Qwen API key is required. "
                "Pass api_key= or set the DASHSCOPE_API_KEY environment variable."
            )
        self._model = model
        self._llm = ChatOpenAI(
            model=model,
            api_key=resolved_key,
            base_url=base_url,
            max_tokens=max_tokens,
        )
        self._tool_formatter = ToolFormatter()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        available_tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        enriched_system = self._build_system(system_prompt, available_tools)
        messages = [
            SystemMessage(content=enriched_system),
            HumanMessage(content=user_prompt),
        ]

        start = time.time()
        try:
            response = self._llm.bind(temperature=temperature).invoke(messages)
        except Exception:
            logger.exception("LLM invocation failed model=%s", self._model)
            return LLMResponse(content="", tool_calls=[], usage={}, latency_ms=0)
        latency_ms = int((time.time() - start) * 1000)

        content = (response.content or "").strip()

        # usage_metadata: {"input_tokens": N, "output_tokens": M, "total_tokens": K}
        meta = getattr(response, "usage_metadata", None) or {}
        usage = {
            "prompt_tokens":     meta.get("input_tokens", 0),
            "completion_tokens": meta.get("output_tokens", 0),
        }

        return LLMResponse(
            content=content,
            tool_calls=[],
            usage=usage,
            latency_ms=latency_ms,
        )

    def _build_system(
        self,
        system_prompt: str,
        available_tools: Optional[list[dict]],
    ) -> str:
        if not available_tools:
            return system_prompt
        tool_block = self._tool_formatter.format_for_prompt(available_tools)
        return f"{system_prompt}\n\n{tool_block}"
