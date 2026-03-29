import random
from ego_crawler.application.ports.web_tools import WebTools, ToolResult


class FakeWebTools(WebTools):
    """Fake web tools for testing"""

    def __init__(self, success_rate: float = 0.9):
        self.success_rate = success_rate
        self._current_url = "https://example.com"
        self._history = []

    def search(self, query: str) -> ToolResult:
        if random.random() > self.success_rate:
            return ToolResult(
                success=False,
                data=None,
                error_message="Search timeout"
            )

        results = [
            {"title": f"Result for {query} #1", "url": f"https://result1.com/{query}"},
            {"title": f"Result for {query} #2", "url": f"https://result2.com/{query}"},
        ]
        return ToolResult(
            success=True,
            data=results,
            metadata={"query": query, "result_count": len(results)}
        )

    def navigate(self, url: str) -> ToolResult:
        self._history.append(self._current_url)
        self._current_url = url
        return ToolResult(
            success=True,
            data={"url": url, "title": f"Page at {url}"},
            metadata={"previous": self._history[-1] if self._history else None}
        )

    def extract_text(self, selector: str = None) -> ToolResult:
        return ToolResult(
            success=True,
            data=f"This is some text content from {self._current_url}",
            metadata={"selector": selector, "length": 100}
        )

    def extract_images(self, selector: str = None) -> ToolResult:
        images = [
            {"ego_crawler": "https://example.com/img1.jpg", "alt": "Image 1"},
            {"ego_crawler": "https://example.com/img2.jpg", "alt": "Image 2"},
        ]
        return ToolResult(
            success=True,
            data=images,
            metadata={"count": len(images)}
        )

    def scroll(self, direction: str = "down") -> ToolResult:
        return ToolResult(
            success=True,
            data={"scrolled": direction, "new_position": "middle"},
            metadata={}
        )

    def go_back(self) -> ToolResult:
        if not self._history:
            return ToolResult(
                success=False,
                data=None,
                error_message="No history to go back to"
            )
        previous = self._history.pop()
        self._current_url = previous
        return ToolResult(
            success=True,
            data={"returned_to": previous},
            metadata={}
        )
