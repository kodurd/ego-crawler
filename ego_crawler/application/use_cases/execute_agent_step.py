import json
import time
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from ego_crawler.domain.entities.session import Session
from ego_crawler.domain.entities.agent_step import AgentStep
from ego_crawler.domain.value_objects.emotion import Emotion, Mood
from ego_crawler.domain.repositories.session_repository import SessionRepository
from ego_crawler.domain.repositories.step_repository import StepRepository
from ego_crawler.application.ports.llm_client import LLMClient
from ego_crawler.application.ports.web_tools import WebTools
from ego_crawler.application.services.prompt_builder import PromptBuilder


@dataclass
class StepResult:
    thought: AgentStep
    action: Optional[AgentStep] = None
    observation: Optional[AgentStep] = None
    session: Session = None


class ExecuteAgentStep:
    AVAILABLE_TOOLS = [
        {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {"query": "string"}
        },
        {
            "name": "navigate",
            "description": "Navigate to a specific URL",
            "parameters": {"url": "string"}
        },
        {
            "name": "extract_text",
            "description": "Extract text from current page",
            "parameters": {"selector": "string (optional)"}
        },
        {
            "name": "extract_images",
            "description": "Extract images from current page",
            "parameters": {"selector": "string (optional)"}
        },
        {
            "name": "scroll",
            "description": "Scroll the page",
            "parameters": {"direction": "up|down"}
        },
        {
            "name": "go_back",
            "description": "Go back to previous page",
            "parameters": {}
        }
    ]

    def __init__(
            self,
            session_repo: SessionRepository,
            step_repo: StepRepository,
            llm_client: LLMClient,
            web_tools: WebTools,
            prompt_builder: PromptBuilder
    ):
        self.session_repo = session_repo
        self.step_repo = step_repo
        self.llm_client = llm_client
        self.web_tools = web_tools
        self.prompt_builder = prompt_builder

    def execute(self, session_id: UUID) -> StepResult:
        session = self.session_repo.get_by_id(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        if not session.is_active():
            raise ValueError(f"Session {session_id} is not active")

        recent_steps = self.step_repo.get_latest(session_id, n=5)

        system_prompt = self.prompt_builder.build_system_prompt(session)
        user_prompt = self.prompt_builder.build_user_prompt(session, recent_steps)

        start_time = time.time()
        llm_response = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            available_tools=self.AVAILABLE_TOOLS,
            temperature=0.8
        )
        latency_ms = int((time.time() - start_time) * 1000)

        thought_content, action_tool, action_params = self._parse_response(
            llm_response.content
        )

        step_number = session.next_step_number()
        thought_step = AgentStep.create_thought(
            session_id=session_id,
            step_number=step_number,
            content=thought_content,
            latency_ms=latency_ms,
            token_count_input=llm_response.usage.get("prompt_tokens"),
            token_count_output=llm_response.usage.get("completion_tokens")
        )
        self.step_repo.save(thought_step)
        session.add_step_to_context(thought_step)

        action_step = None
        observation_step = None

        if action_tool:
            action_step = AgentStep.create_action(
                session_id=session_id,
                step_number=step_number,  # Same number as thought
                tool=action_tool,
                params=action_params
            )
            self.step_repo.save(action_step)

            tool_result = self._execute_tool(action_tool, action_params)

            observation_step = AgentStep.create_observation(
                session_id=session_id,
                step_number=step_number,
                raw_data=str(tool_result.data)[:2000],
                summary=tool_result.error_message if not tool_result.success else None,
                observation_success=tool_result.success,
            )
            self.step_repo.save(observation_step)
            session.add_step_to_context(observation_step)

            new_emotion = self._derive_emotion(session, tool_result)
            session = session.update_emotion(new_emotion)

        session = session.consume_budget(30)
        self.session_repo.save(session)

        return StepResult(
            thought=thought_step,
            action=action_step,
            observation=observation_step,
            session=session
        )

    def _parse_response(self, content: str) -> tuple:
        lines = content.strip().split("\n")
        thought = ""
        action_tool = None
        action_params = None

        for line in lines:
            line = line.strip()
            if line.startswith("THOUGHT:"):
                thought = line[8:].strip()
            elif line.startswith("ACTION:"):
                action_part = line[7:].strip()
                if "|" in action_part:
                    parts = action_part.split("|", 1)
                    action_tool = parts[0].strip()
                    try:
                        action_params = json.loads(parts[1].strip())
                    except json.JSONDecodeError:
                        action_params = {"raw": parts[1].strip()}
                else:
                    action_tool = action_part
                    action_params = {}

        return thought, action_tool, action_params

    def _execute_tool(self, tool: str, params: dict):
        tools_map = {
            "web_search": self.web_tools.search,
            "navigate": self.web_tools.navigate,
            "extract_text": self.web_tools.extract_text,
            "extract_images": self.web_tools.extract_images,
            "scroll": self.web_tools.scroll,
            "go_back": self.web_tools.go_back,
        }

        if tool not in tools_map:
            from ego_crawler.application.ports.web_tools import ToolResult
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Unknown tool: {tool}"
            )

        try:
            return tools_map[tool](**params)
        except Exception as e:
            from ego_crawler.application.ports.web_tools import ToolResult
            return ToolResult(
                success=False,
                data=None,
                error_message=str(e)
            )

    def _derive_emotion(self, session: Session, tool_result):
        if tool_result.success:
            return Emotion(Mood.CURIOUS, 0.7, "found something interesting")
        else:
            return Emotion(Mood.FRUSTRATED, 0.6, tool_result.error_message)
