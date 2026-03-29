from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ToolResult:
    success: bool
    data: Any
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class WebTools(ABC):
    @abstractmethod
    def search(self, query: str) -> ToolResult:
        """Execute web search"""
        pass

    @abstractmethod
    def navigate(self, url: str) -> ToolResult:
        """Navigate to URL"""
        pass

    @abstractmethod
    def extract_text(self, selector: Optional[str] = None) -> ToolResult:
        """Extract text from current page"""
        pass

    @abstractmethod
    def extract_images(self, selector: Optional[str] = None) -> ToolResult:
        """Extract images from current page"""
        pass

    @abstractmethod
    def scroll(self, direction: str = "down") -> ToolResult:
        """Scroll page"""
        pass

    @abstractmethod
    def go_back(self) -> ToolResult:
        """Go back to previous page"""
        pass