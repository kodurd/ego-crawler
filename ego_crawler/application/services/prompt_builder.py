from typing import List, Optional

from ego_crawler.domain.entities.session import Session
from ego_crawler.domain.entities.agent_step import AgentStep


class PromptBuilder:

    def build_system_prompt(self, session: Session) -> str:
        base = session.persona.get_full_prompt()
        constraints = self._build_constraints(session)
        return f"{base}\n\n{constraints}"

    def build_user_prompt(
            self,
            session: Session,
            recent_steps: Optional[List[AgentStep]] = None
    ) -> str:
        lines = [
            "=== CURRENT STATE ===",
            session.get_context_for_prompt(),
            "",
            "=== RECENT HISTORY ===",
        ]

        if recent_steps:
            for step in recent_steps:
                lines.append(self._format_step(step))
        else:
            lines.append("No previous actions.")

        lines.extend([
            "",
            "=== YOUR TURN ===",
            "Based on your personality and current state, decide what to do next.",
            "You can:",
            "1. Think internally (output thought)",
            "2. Use a tool (output action)",
            "",
            "Respond in this format:",
            "THOUGHT: <your reasoning as persona>",
            "ACTION: <tool_name> | <params_json> (if using tool)",
            "OR",
            "THOUGHT: <your reasoning>",
            "(no action if just thinking)",
        ])

        return "\n".join(lines)

    def _build_constraints(self, session: Session) -> str:
        return (
            f"CONSTRAINTS:\n"
            f"- You have {session.budget.as_timedelta()} remaining\n"
            f"- You cannot exceed your time budget\n"
            f"- Stay in character at all times\n"
            f"- You may get bored, excited, or frustrated based on results"
        )

    def _format_step(self, step: AgentStep) -> str:
        if step.step_type.name == "THOUGHT":
            return f"[{step.step_number}] Thought: {step.thought_content[:100]}..."
        elif step.step_type.name == "ACTION":
            return f"[{step.step_number}] Action: {step.action_tool}({step.action_params})"
        elif step.step_type.name == "OBSERVATION":
            summary = step.observation_summary or step.observation_raw[:100]
            return f"[{step.step_number}] Result: {summary}..."
        return f"[{step.step_number}] {step.step_type.name}"