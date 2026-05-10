from __future__ import annotations

from typing import Optional

from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

from ego_crawler.application.ports.web_tools import ToolResult, WebTools
from ego_crawler.infrastructure.playwright.search_parser import (
    GOOGLE_EXTRACT_JS,
    GoogleSearchParser,
)

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
_SCROLL_PX = 500
_GOOGLE_SEARCH_URL = "https://www.google.com/search?hl=en&q={query}"


class PlaywrightWebTools(WebTools):
    """Production WebTools backed by a headless Playwright Chromium browser.

    Must be used as a context manager:

        with PlaywrightWebTools() as tools:
            result = tools.navigate("https://example.com")
    """

    def __init__(
        self,
        headless: bool = True,
        timeout: int = 30_000,
        user_agent: str = _DEFAULT_USER_AGENT,
    ) -> None:
        self._headless = headless
        self._timeout = timeout
        self._user_agent = user_agent

        self._pw_manager = None
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._search_parser = GoogleSearchParser()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def __enter__(self) -> "PlaywrightWebTools":
        self._pw_manager = sync_playwright()
        self._playwright = self._pw_manager.start()
        self._browser = self._playwright.chromium.launch(headless=self._headless)
        self._context = self._browser.new_context(
            user_agent=self._user_agent,
            viewport={"width": 1280, "height": 720},
            locale="en-US",
        )
        self._page = self._context.new_page()
        self._page.set_default_timeout(self._timeout)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        for resource in (self._page, self._context, self._browser):
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    pass
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
        self._page = self._context = self._browser = self._playwright = None
        return False

    # ── WebTools API ───────────────────────────────────────────────────────

    def navigate(self, url: str) -> ToolResult:
        self._require_page()
        try:
            response = self._page.goto(
                url, wait_until="domcontentloaded", timeout=self._timeout
            )
            self._page.wait_for_load_state("networkidle", timeout=self._timeout)
            return ToolResult(
                success=True,
                data={
                    "url": self._page.url,
                    "title": self._page.title(),
                    "status_code": response.status if response else None,
                },
                metadata={"requested_url": url},
            )
        except PlaywrightTimeoutError as exc:
            return ToolResult(success=False, data=None, error_message=f"Timeout: {exc}")
        except Exception as exc:
            return ToolResult(success=False, data=None, error_message=str(exc))

    def search(self, query: str) -> ToolResult:
        self._require_page()
        try:
            search_url = _GOOGLE_SEARCH_URL.format(query=query)
            self._page.goto(
                search_url, wait_until="domcontentloaded", timeout=self._timeout
            )
            self._page.wait_for_load_state("networkidle", timeout=self._timeout)

            raw: list[dict] = self._page.evaluate(GOOGLE_EXTRACT_JS)
            results = self._search_parser.parse(raw)

            return ToolResult(
                success=True,
                data=[
                    {"title": r.title, "url": r.url, "snippet": r.snippet}
                    for r in results
                ],
                metadata={"query": query, "result_count": len(results)},
            )
        except PlaywrightTimeoutError as exc:
            return ToolResult(success=False, data=None, error_message=f"Timeout: {exc}")
        except Exception as exc:
            return ToolResult(success=False, data=None, error_message=str(exc))

    def extract_text(self, selector: Optional[str] = None) -> ToolResult:
        self._require_page()
        try:
            if selector:
                text = self._page.locator(selector).first.inner_text(
                    timeout=self._timeout
                )
            else:
                text = self._page.locator("body").inner_text(timeout=self._timeout)
            return ToolResult(
                success=True,
                data=text,
                metadata={
                    "selector": selector,
                    "length": len(text),
                    "url": self._page.url,
                },
            )
        except PlaywrightTimeoutError as exc:
            return ToolResult(success=False, data=None, error_message=f"Timeout: {exc}")
        except Exception as exc:
            return ToolResult(success=False, data=None, error_message=str(exc))

    def extract_images(self, selector: Optional[str] = None) -> ToolResult:
        self._require_page()
        try:
            img_selector = f"{selector} img" if selector else "img"
            images: list[dict] = self._page.eval_on_selector_all(
                img_selector,
                "els => els.map(e => ({"
                "src: e.src, alt: e.alt || '', "
                "width: e.naturalWidth, height: e.naturalHeight"
                "}))",
            )
            return ToolResult(
                success=True,
                data=images,
                metadata={
                    "count": len(images),
                    "selector": selector,
                    "url": self._page.url,
                },
            )
        except Exception as exc:
            return ToolResult(success=False, data=None, error_message=str(exc))

    def scroll(self, direction: str = "down") -> ToolResult:
        self._require_page()
        try:
            delta = _SCROLL_PX if direction == "down" else -_SCROLL_PX
            self._page.evaluate(f"window.scrollBy(0, {delta})")
            position_y: int = self._page.evaluate("window.scrollY")
            return ToolResult(
                success=True,
                data={"direction": direction, "px": _SCROLL_PX, "position_y": position_y},
                metadata={},
            )
        except Exception as exc:
            return ToolResult(success=False, data=None, error_message=str(exc))

    def go_back(self) -> ToolResult:
        self._require_page()
        try:
            response = self._page.go_back(
                wait_until="domcontentloaded", timeout=self._timeout
            )
            if response is None:
                return ToolResult(
                    success=False,
                    data=None,
                    error_message="No previous page in history",
                )
            self._page.wait_for_load_state("networkidle", timeout=self._timeout)
            return ToolResult(
                success=True,
                data={"returned_to": self._page.url, "title": self._page.title()},
                metadata={},
            )
        except PlaywrightTimeoutError as exc:
            return ToolResult(success=False, data=None, error_message=f"Timeout: {exc}")
        except Exception as exc:
            return ToolResult(success=False, data=None, error_message=str(exc))

    # ── Internal ───────────────────────────────────────────────────────────

    def _require_page(self) -> None:
        if self._page is None:
            raise RuntimeError(
                "PlaywrightWebTools must be used as a context manager: "
                "`with PlaywrightWebTools() as tools: ...`"
            )
