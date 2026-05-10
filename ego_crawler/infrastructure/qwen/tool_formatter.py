from __future__ import annotations

from typing import Any


class ToolFormatter:
    """Converts the internal AVAILABLE_TOOLS list into a human-readable text
    block suitable for injection into a system prompt.

    Kept dependency-free so it can be unit-tested without any LLM SDK.
    """

    _HEADER = "Available tools — use the exact name in your ACTION field:"
    _SEPARATOR = "-" * 48

    def format_for_prompt(self, tools: list[dict[str, Any]]) -> str:
        """Returns a formatted multi-line string describing all tools.

        Example output:
            Available tools — use the exact name in your ACTION field:
            ------------------------------------------------
              web_search(query: string)
                → Search the web for information
              navigate(url: string)
                → Navigate to a specific URL
            ------------------------------------------------
        """
        if not tools:
            return ""

        lines = [self._HEADER, self._SEPARATOR]
        for tool in tools:
            params_str = self._format_params(tool.get("parameters", {}))
            signature = f"  {tool['name']}({params_str})"
            lines.append(signature)
            lines.append(f"    → {tool['description']}")
        lines.append(self._SEPARATOR)
        return "\n".join(lines)

    def _format_params(self, parameters: dict[str, str]) -> str:
        if not parameters:
            return ""
        return ", ".join(f"{k}: {v}" for k, v in parameters.items())
