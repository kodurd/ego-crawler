import pytest

from ego_crawler.infrastructure.qwen.tool_formatter import ToolFormatter

_SAMPLE_TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {"query": "string"},
    },
    {
        "name": "navigate",
        "description": "Navigate to a specific URL",
        "parameters": {"url": "string"},
    },
    {
        "name": "go_back",
        "description": "Go back to previous page",
        "parameters": {},
    },
]


@pytest.fixture
def formatter() -> ToolFormatter:
    return ToolFormatter()


class TestToolFormatterFormatForPrompt:
    def test_empty_tools_returns_empty_string(self, formatter):
        assert formatter.format_for_prompt([]) == ""

    def test_output_contains_tool_names(self, formatter):
        result = formatter.format_for_prompt(_SAMPLE_TOOLS)

        assert "web_search" in result
        assert "navigate" in result
        assert "go_back" in result

    def test_output_contains_descriptions(self, formatter):
        result = formatter.format_for_prompt(_SAMPLE_TOOLS)

        assert "Search the web for information" in result
        assert "Navigate to a specific URL" in result
        assert "Go back to previous page" in result

    def test_output_contains_parameter_names(self, formatter):
        result = formatter.format_for_prompt(_SAMPLE_TOOLS)

        assert "query" in result
        assert "url" in result

    def test_tool_with_no_parameters_has_empty_parens(self, formatter):
        tools = [{"name": "go_back", "description": "Go back", "parameters": {}}]
        result = formatter.format_for_prompt(tools)

        assert "go_back()" in result

    def test_tool_with_single_parameter(self, formatter):
        tools = [
            {"name": "search", "description": "Search", "parameters": {"query": "string"}}
        ]
        result = formatter.format_for_prompt(tools)

        assert "search(query: string)" in result

    def test_tool_with_multiple_parameters(self, formatter):
        tools = [
            {
                "name": "click",
                "description": "Click element",
                "parameters": {"selector": "string", "timeout": "int"},
            }
        ]
        result = formatter.format_for_prompt(tools)

        assert "selector: string" in result
        assert "timeout: int" in result

    def test_output_contains_header(self, formatter):
        result = formatter.format_for_prompt(_SAMPLE_TOOLS)

        assert "Available tools" in result

    def test_output_is_multiline(self, formatter):
        result = formatter.format_for_prompt(_SAMPLE_TOOLS)

        assert "\n" in result

    def test_all_six_agent_tools(self, formatter):
        six_tools = [
            {"name": "web_search",     "description": "Search",          "parameters": {"query": "string"}},
            {"name": "navigate",       "description": "Navigate",         "parameters": {"url": "string"}},
            {"name": "extract_text",   "description": "Extract text",     "parameters": {"selector": "string (optional)"}},
            {"name": "extract_images", "description": "Extract images",   "parameters": {"selector": "string (optional)"}},
            {"name": "scroll",         "description": "Scroll",           "parameters": {"direction": "up|down"}},
            {"name": "go_back",        "description": "Go back",          "parameters": {}},
        ]
        result = formatter.format_for_prompt(six_tools)

        for tool in six_tools:
            assert tool["name"] in result
